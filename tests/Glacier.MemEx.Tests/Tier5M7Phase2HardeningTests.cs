namespace Glacier.MemEx.Tests;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Hardware;
using Glacier.MemEx;
using Glacier.MemEx.Models;
using Glacier.MemEx.Recall;
using Xunit;

public sealed class Tier5M7Phase2HardeningTests : IDisposable
{
    private sealed class StubMemExHandler : HttpMessageHandler
    {
        private readonly Func<HttpRequestMessage, int, CancellationToken, Task<HttpResponseMessage>> _responder;
        private int _calls;

        public StubMemExHandler(Func<HttpRequestMessage, int, CancellationToken, Task<HttpResponseMessage>> responder)
        {
            _responder = responder;
        }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            int call = Interlocked.Increment(ref _calls);
            return _responder(request, call, cancellationToken);
        }
    }

    public Tier5M7Phase2HardeningTests() => NpuSidecarDiscovery.InvalidateCache();
    public void Dispose() => NpuSidecarDiscovery.InvalidateCache();

    private static HttpResponseMessage Json(HttpStatusCode status, string body) =>
        new(status) { Content = new StringContent(body, Encoding.UTF8, "application/json") };

    private static string MakeVectorJson(int dims, float seedVal = 1.0f)
    {
        var arr = new float[dims];
        arr[0] = seedVal;
        if (dims > 1) arr[dims - 1] = 0.5f;
        return $"{{\"data\":[{{\"index\":0,\"embedding\":{JsonSerializer.Serialize(arr)}}}]}}";
    }

    [Fact]
    public async Task MemExEngine_ConcurrentStoreQueryWithDisposeRace_SafeWithoutUnhandledCrash()
    {
        using var handler = new StubMemExHandler(async (req, _, ct) =>
        {
            string path = req.RequestUri?.AbsolutePath ?? "";
            if (path.EndsWith("/models", StringComparison.OrdinalIgnoreCase))
            {
                await Task.Delay(15, ct);
                return Json(HttpStatusCode.OK, """{"data":[{"id":"embeddinggemma"}]}""");
            }
            if (path.EndsWith("/embeddings", StringComparison.OrdinalIgnoreCase))
            {
                await Task.Delay(15, ct);
                return Json(HttpStatusCode.OK, MakeVectorJson(128));
            }
            return Json(HttpStatusCode.NotFound, "{}");
        });
        using var http = new HttpClient(handler);

        var memEx = new MemExEngine(new MemExOptions
        {
            EnableNpuEmbeddings = true,
            EmbeddingEndpoint = "http://127.0.0.1:52625/v1",
            HttpClient = http,
            ProbeTimeout = TimeSpan.FromSeconds(1),
            EmbeddingTimeout = TimeSpan.FromSeconds(1)
        });

        var tasks = new List<Task>();
        for (int i = 0; i < 8; i++)
        {
            int id = i;
            tasks.Add(Task.Run(async () =>
            {
                try
                {
                    await memEx.StoreTraceAsync(new EpisodicTrace { Id = $"t{id}", Content = $"race content {id}" });
                    await memEx.QueryAsync($"race {id}");
                }
                catch (Exception ex) when (ex is ObjectDisposedException or OperationCanceledException)
                {
                    // Clean shutdown cancellation
                }
            }));
        }

        await Task.Delay(10);
        memEx.Dispose();
        memEx.Dispose(); // double dispose safe
        await Task.WhenAll(tasks);

        await Assert.ThrowsAsync<ObjectDisposedException>(() => memEx.QueryAsync("after-dispose"));
    }

    [Fact]
    public async Task MemExEngine_NonMultipleOf8NpuDims300_MidSessionDropout_PreservesExactLockedDimension()
    {
        int failEmbeddings = 0;
        using var handler = new StubMemExHandler((req, _, _) =>
        {
            string path = req.RequestUri?.AbsolutePath ?? "";
            if (path.EndsWith("/models", StringComparison.OrdinalIgnoreCase))
            {
                return Task.FromResult(Json(HttpStatusCode.OK, """{"data":[{"id":"embed-300d"}]}"""));
            }
            if (path.EndsWith("/embeddings", StringComparison.OrdinalIgnoreCase))
            {
                if (Volatile.Read(ref failEmbeddings) == 1)
                {
                    return Task.FromResult(Json(HttpStatusCode.InternalServerError, """{"error":"drop"}"""));
                }
                return Task.FromResult(Json(HttpStatusCode.OK, MakeVectorJson(300)));
            }
            return Task.FromResult(Json(HttpStatusCode.NotFound, "{}"));
        });
        using var http = new HttpClient(handler);

        using var memEx = new MemExEngine(new MemExOptions
        {
            EnableNpuEmbeddings = true,
            EmbeddingEndpoint = "http://127.0.0.1:52625/v1",
            HttpClient = http,
            ProbeTimeout = TimeSpan.FromSeconds(1),
            EmbeddingTimeout = TimeSpan.FromSeconds(1)
        });

        var t1 = new EpisodicTrace { Id = "npu-300", Content = "initial non-multiple-of-8 300d trace" };
        await memEx.StoreTraceAsync(t1);
        Assert.NotNull(t1.Embedding);
        Assert.Equal(300, t1.Embedding!.Length);

        // Simulate mid-session NPU outage -> fallback vector MUST match locked 300 dims (300 % 8 != 0)
        Volatile.Write(ref failEmbeddings, 1);
        var t2 = new EpisodicTrace { Id = "fallback-300", Content = "post-dropout trace matching 300d" };
        await memEx.StoreTraceAsync(t2);

        Assert.NotNull(t2.Embedding);
        Assert.Equal(t1.Embedding.Length, t2.Embedding!.Length);
        Assert.NotNull(t2.BinaryEmbedding);

        var recall = await memEx.QueryAsync("dropout trace");
        Assert.NotEmpty(recall.SelectedTraces);
    }

    [Fact]
    public async Task MemEx_EmptyInputs_LargeVectorFilter_And_HippoRagCyclicCaseInsensitive()
    {
        using var memEx = new MemExEngine();
        var emptyTrace = new EpisodicTrace { Id = "empty", Content = "   ", Entities = null!, Metadata = null! };
        await memEx.StoreTraceAsync(emptyTrace);
        var emptyRecall = await memEx.QueryAsync("   ");
        Assert.NotNull(emptyRecall);

        // TwoTierVectorFilter with > CoarseCandidateLimit (500) traces, zero-vectors, and 1e30f magnitudes
        const int dim = 32;
        var traces = new List<EpisodicTrace>(600);
        for (int i = 0; i < 600; i++)
        {
            float[] v = new float[dim];
            if (i == 0) v[0] = 0f; // all-zero embedding
            else if (i == 1) { v[0] = 1e30f; v[1] = 1e30f; } // extreme magnitude (would overflow dot without normalization/finite clamp)
            else { v[i % dim] = 1.0f; }

            traces.Add(new EpisodicTrace { Id = $"c-{i}", Content = $"content {i}", Embedding = v });
        }
        float[] q = new float[dim];
        q[0] = 1.0f;
        var filtered = TwoTierVectorFilter.FilterAndRerank(q, traces, coarseLimit: 256, finalLimit: 10);
        Assert.NotEmpty(filtered);
        Assert.All(filtered, r => Assert.True(float.IsFinite(r.CosineSimilarity)));

        // HippoRagSpreadingActivation: cyclic graph + disconnected + case-insensitive entities
        var graphTraces = new List<EpisodicTrace>
        {
            new() { Id = "g1", Content = "A->B", Entities = ["Alpha", "Beta"] },
            new() { Id = "g2", Content = "B->C", Entities = ["beta", "Gamma"] },
            new() { Id = "g3", Content = "C->A", Entities = ["GAMMA", "alpha"] },
            new() { Id = "g4", Content = "Isolated", Entities = ["Island"] }
        };
        var ppr = HippoRagSpreadingActivation.ComputePersonalizedPageRank(["ALPHA"], graphTraces, dampingAlpha: 0.85, maxIterations: 25);
        Assert.True(ppr.Values.All(double.IsFinite));
        Assert.True(ppr.TryGetValue("Island", out double islandScore) ? islandScore == 0.0 : true);
    }
}
