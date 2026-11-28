namespace Glacier.MemEx.Tests;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Hardware;
using Glacier.MemEx;
using Glacier.MemEx.Models;
using Glacier.Rag.Embeddings;
using Xunit;

public sealed class NpuSemanticEmbeddingTests : IDisposable
{
    public NpuSemanticEmbeddingTests()
    {
        NpuSidecarDiscovery.InvalidateCache();
    }

    public void Dispose()
    {
        NpuSidecarDiscovery.InvalidateCache();
    }

    [Fact]
    public async Task T1_StoreTraceAsync_And_QueryAsync_ExplicitNpuUrl_UsesOpenAiEmbeddings()
    {
        var recordedBodies = new List<string>();
        using var handler = new StubHttpMessageHandler(async (req, ct) =>
        {
            string path = req.RequestUri?.AbsolutePath ?? string.Empty;
            if (req.Method == HttpMethod.Get && path == "/v1/models")
            {
                return JsonResponse("""{"data":[{"id":"embeddinggemma"}]}""");
            }

            if (req.Method == HttpMethod.Post && path == "/v1/embeddings")
            {
                string body = await req.Content!.ReadAsStringAsync(ct).ConfigureAwait(false);
                lock (recordedBodies)
                {
                    recordedBodies.Add(body);
                }

                using var doc = JsonDocument.Parse(body);
                string model = doc.RootElement.GetProperty("model").GetString() ?? "";
                string input = doc.RootElement.GetProperty("input").GetString() ?? "";
                Assert.Equal("embeddinggemma", model);

                float[] vec = BuildSemanticVector(input, 768);
                return JsonResponse(SerializeEmbeddingResponse(vec));
            }

            return new HttpResponseMessage(HttpStatusCode.NotFound);
        });

        using var httpClient = new HttpClient(handler);
        using var engine = new MemExEngine(new MemExOptions
        {
            EnableNpuEmbeddings = true,
            EmbeddingEndpoint = "http://127.0.0.1:18081",
            EmbeddingModelId = "embeddinggemma",
            HttpClient = httpClient
        });

        var cardiacTrace = new EpisodicTrace
        {
            Id = "cardiac",
            Content = "Patient experienced acute cardiac arrest in emergency ward.",
            TimestampUs = 1_000_000L,
            BaseImportance = 1.0f
        };
        var k8sTrace = new EpisodicTrace
        {
            Id = "k8s",
            Content = "Restarted failing kubernetes pod across worker nodes.",
            TimestampUs = 1_000_000L,
            BaseImportance = 1.0f
        };

        await engine.StoreTraceAsync(cardiacTrace);
        await engine.StoreTraceAsync(k8sTrace);

        Assert.True(engine.IsUsingNpuEmbeddings);
        Assert.Equal(768, engine.ActiveEmbeddingDimensions);
        Assert.Equal("embeddinggemma", engine.ActiveEmbeddingModelId);
        Assert.Equal("http://127.0.0.1:18081", engine.ActiveEmbeddingEndpoint);
        Assert.NotNull(cardiacTrace.Embedding);
        Assert.Equal(768, cardiacTrace.Embedding!.Length);
        Assert.NotNull(cardiacTrace.BinaryEmbedding);

        // Zero lexical overlap with "cardiac arrest", but semantically aligned in mocked NPU vector space
        MemoryQueryResult result = await engine.QueryAsync(
            "myocardial infarction treatment",
            currentTimestampUs: 1_000_000L);

        Assert.Equal(2, result.SelectedTraces.Count);
        Assert.Equal("cardiac", result.SelectedTraces[0].Id);
        Assert.True(recordedBodies.Count >= 4); // dimension probe + 2 stores + 1 query
    }

    [Fact]
    public async Task T1b_ExplicitNpuUrl_WorksOverRealLoopbackSocket()
    {
        using var server = new LoopbackEmbeddingServer(768, "embeddinggemma", BuildSemanticVector);
        using var engine = new MemExEngine(new MemExOptions
        {
            EnableNpuEmbeddings = true,
            EmbeddingEndpoint = server.BaseUrl,
            EmbeddingTimeout = TimeSpan.FromSeconds(5)
        });

        bool initialized = await engine.InitializeEmbeddingsAsync();
        Assert.True(initialized);
        Assert.Equal(768, engine.ActiveEmbeddingDimensions);
        Assert.Equal("embeddinggemma", engine.ActiveEmbeddingModelId);

        await engine.StoreTraceAsync(new EpisodicTrace
        {
            Id = "heart",
            Content = "cardiac arrest resuscitation protocol",
            TimestampUs = 5_000_000L
        });
        await engine.StoreTraceAsync(new EpisodicTrace
        {
            Id = "cluster",
            Content = "kubernetes pod autoscaling policy",
            TimestampUs = 5_000_000L
        });

        MemoryQueryResult result = await engine.QueryAsync(
            "myocardial infarction",
            currentTimestampUs: 5_000_000L);

        Assert.Equal("heart", result.SelectedTraces[0].Id);
    }

    [Theory]
    [InlineData("auto-npu")]
    [InlineData("npu")]
    [InlineData(null)]
    public async Task T2_NpuDiscovery_AutoNpuAndAliases_DiscoverMockedSidecar(string? endpointAlias)
    {
        using var handler = new StubHttpMessageHandler(async (req, ct) =>
        {
            string url = req.RequestUri?.ToString() ?? "";
            if (req.Method == HttpMethod.Get && url == "http://127.0.0.1:52625/v1/models")
            {
                return JsonResponse("""{"data":[{"id":"qwen2.5:3b"},{"id":"embeddinggemma:latest"}]}""");
            }

            if (req.Method == HttpMethod.Post && url == "http://127.0.0.1:52625/v1/embeddings")
            {
                string body = await req.Content!.ReadAsStringAsync(ct).ConfigureAwait(false);
                using var doc = JsonDocument.Parse(body);
                string input = doc.RootElement.GetProperty("input").GetString() ?? "";
                string model = doc.RootElement.GetProperty("model").GetString() ?? "";
                Assert.Equal("embeddinggemma:latest", model);

                return JsonResponse(SerializeEmbeddingResponse(BuildSemanticVector(input, 768)));
            }

            throw new HttpRequestException("Simulated connection refused for non-active ports.");
        });

        using var httpClient = new HttpClient(handler);
        using var engine = new MemExEngine(new MemExOptions
        {
            EnableNpuEmbeddings = true,
            EmbeddingEndpoint = endpointAlias,
            HttpClient = httpClient,
            ProbeCacheTtl = TimeSpan.Zero
        });

        await engine.StoreTraceAsync(new EpisodicTrace
        {
            Id = "medical",
            Content = "cardiac arrest emergency",
            TimestampUs = 2_000_000L
        });
        await engine.StoreTraceAsync(new EpisodicTrace
        {
            Id = "infra",
            Content = "kubernetes pod crashloop",
            TimestampUs = 2_000_000L
        });

        Assert.True(engine.IsUsingNpuEmbeddings);
        Assert.Equal("http://127.0.0.1:52625", engine.ActiveEmbeddingEndpoint);
        Assert.Equal("embeddinggemma:latest", engine.ActiveEmbeddingModelId);
        Assert.Equal(768, engine.ActiveEmbeddingDimensions);

        MemoryQueryResult result = await engine.QueryAsync("myocardial infarction", currentTimestampUs: 2_000_000L);
        Assert.Equal("medical", result.SelectedTraces[0].Id);
    }

    [Fact]
    public void T3_SyncStoreTrace_And_SyncQuery_AutoEmbedWhenMissing_AgentContract()
    {
        // Mode A: Mocked NPU with active SynchronizationContext (verifies deadlock-safe sync bridging)
        using var handler = new StubHttpMessageHandler(async (req, ct) =>
        {
            string path = req.RequestUri?.AbsolutePath ?? "";
            if (req.Method == HttpMethod.Get && path == "/v1/models")
            {
                return JsonResponse("""{"data":[{"id":"embeddinggemma"}]}""");
            }

            if (req.Method == HttpMethod.Post && path == "/v1/embeddings")
            {
                string body = await req.Content!.ReadAsStringAsync(ct).ConfigureAwait(false);
                using var doc = JsonDocument.Parse(body);
                string input = doc.RootElement.GetProperty("input").GetString() ?? "";
                return JsonResponse(SerializeEmbeddingResponse(BuildSemanticVector(input, 768)));
            }

            return new HttpResponseMessage(HttpStatusCode.NotFound);
        });

        using var httpClient = new HttpClient(handler);
        using var npuEngine = new MemExEngine(new MemExOptions
        {
            EnableNpuEmbeddings = true,
            EmbeddingEndpoint = "http://127.0.0.1:18083",
            HttpClient = httpClient
        });

        SynchronizationContext? previousContext = SynchronizationContext.Current;
        try
        {
            SynchronizationContext.SetSynchronizationContext(new NonCapturingTestSyncContext());

            var t1 = new EpisodicTrace
            {
                Id = "t-cardiac",
                Content = "Prompt: handle cardiac arrest\nResult: started CPR",
                BaseImportance = 2.0f,
                TimestampUs = 10_000_000L
            };
            var t2 = new EpisodicTrace
            {
                Id = "t-k8s",
                Content = "Prompt: debug kubernetes pod\nResult: checked events",
                BaseImportance = 2.0f,
                TimestampUs = 10_000_000L
            };

            npuEngine.StoreTrace(t1);
            npuEngine.StoreTrace(t2);

            Assert.NotNull(t1.Embedding);
            Assert.NotNull(t1.BinaryEmbedding);

            MemoryQueryResult npuRecall = npuEngine.Query(
                "myocardial infarction",
                ReadOnlySpan<float>.Empty,
                currentTimestampUs: 10_000_000L);

            Assert.Equal("t-cardiac", npuRecall.SelectedTraces[0].Id);
        }
        finally
        {
            SynchronizationContext.SetSynchronizationContext(previousContext);
        }

        // Mode B: Offline hash mode (AgentExecutionEngine contract without NPU)
        using var offlineEngine = new MemExEngine(new MemExOptions
        {
            EnableNpuEmbeddings = false,
            FallbackHashDimensions = 384
        });

        var h1 = new EpisodicTrace
        {
            Id = "h-sql",
            Content = "PostgreSQL vacuum deadlock troubleshooting guide", // 48 chars -> 12 tokens
            BaseImportance = 1.5f,
            TimestampUs = 10_000_000L
        };
        var h2 = new EpisodicTrace
        {
            Id = "h-css",
            Content = "Tailwind flexbox grid layout alignment cheatsheet", // 49 chars -> 12 tokens
            BaseImportance = 1.5f,
            TimestampUs = 10_000_000L
        };

        offlineEngine.StoreTrace(h1);
        offlineEngine.StoreTrace(h2);

        Assert.False(offlineEngine.IsUsingNpuEmbeddings);
        Assert.Equal(384, offlineEngine.ActiveEmbeddingDimensions);
        Assert.NotNull(h1.Embedding);
        Assert.Equal(384, h1.Embedding!.Length);

        MemoryQueryResult hashRecall = offlineEngine.Query(
            "PostgreSQL deadlock",
            ReadOnlySpan<float>.Empty,
            currentTimestampUs: 10_000_000L);

        Assert.Equal("h-sql", hashRecall.SelectedTraces[0].Id);
    }

    [Fact]
    public async Task T4_ExplicitCallerEmbedding_And_QuerySpan_ArePreserved_SleepMemoryContract()
    {
        int httpCalls = 0;
        using var handler = new StubHttpMessageHandler((_, _) =>
        {
            Interlocked.Increment(ref httpCalls);
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.InternalServerError));
        });

        using var httpClient = new HttpClient(handler);
        using var engine = new MemExEngine(new MemExOptions
        {
            EnableNpuEmbeddings = true,
            EmbeddingEndpoint = "http://127.0.0.1:18084",
            HttpClient = httpClient
        });

        float[] custom64Target = CreateUnitBasisVector(64, 3);
        float[] custom64Other = CreateUnitBasisVector(64, 17);
        float[] custom64Query = CreateUnitBasisVector(64, 3);

        var targetTrace = new EpisodicTrace
        {
            Id = "sleep-target",
            Content = "Consolidated turn summary A",
            Embedding = custom64Target,
            TokenCount = 42,
            TimestampUs = 5_000_000L
        };
        var otherTrace = new EpisodicTrace
        {
            Id = "sleep-other",
            Content = "Consolidated turn summary B",
            Embedding = custom64Other,
            TokenCount = 42,
            TimestampUs = 5_000_000L
        };

        engine.StoreTrace(targetTrace);
        await engine.StoreTraceAsync(otherTrace);

        Assert.Same(custom64Target, targetTrace.Embedding);
        Assert.Equal(42, targetTrace.TokenCount);
        Assert.NotNull(targetTrace.BinaryEmbedding);

        MemoryQueryResult syncResult = engine.Query(
            "ignored text because explicit vector is provided",
            custom64Query,
            currentTimestampUs: 5_000_000L);

        MemoryQueryResult asyncResult = await engine.QueryAsync(
            "ignored text because explicit vector is provided",
            custom64Query,
            currentTimestampUs: 5_000_000L);

        Assert.Equal(0, httpCalls);
        Assert.Null(engine.ActiveEmbeddingDimensions);
        Assert.Equal("sleep-target", syncResult.SelectedTraces[0].Id);
        Assert.Equal("sleep-target", asyncResult.SelectedTraces[0].Id);
    }

    [Fact]
    public async Task T5_TwoTierVectorFilter_And_HippoRagSpreadingActivation_CombinedRanking()
    {
        using var handler = new StubHttpMessageHandler(async (req, ct) =>
        {
            string path = req.RequestUri?.AbsolutePath ?? "";
            if (req.Method == HttpMethod.Post && path == "/v1/embeddings")
            {
                string body = await req.Content!.ReadAsStringAsync(ct).ConfigureAwait(false);
                using var doc = JsonDocument.Parse(body);
                string input = doc.RootElement.GetProperty("input").GetString() ?? "";

                // Give neutral uniform similarity for all traces so graph + vector combination is exercised cleanly,
                // while making "timeout" slightly closer than "billing"
                float[] vec = new float[128];
                if (input.Contains("timeout", StringComparison.OrdinalIgnoreCase) ||
                    input.Contains("latency", StringComparison.OrdinalIgnoreCase))
                {
                    vec[0] = 0.9f;
                    vec[1] = 0.4358899f;
                }
                else
                {
                    vec[0] = 0.5f;
                    vec[2] = 0.8660254f;
                }

                return JsonResponse(SerializeEmbeddingResponse(vec));
            }

            return new HttpResponseMessage(HttpStatusCode.NotFound);
        });

        using var httpClient = new HttpClient(handler);
        using var engine = new MemExEngine(new MemExOptions
        {
            EnableNpuEmbeddings = true,
            EmbeddingEndpoint = "http://127.0.0.1:18085",
            EmbeddingModelId = "embeddinggemma",
            EmbeddingDimensions = 128,
            CoarseCandidateLimit = 4, // Exercises Tier 1 1-bit Hamming quantization when > 4 traces exist
            WeightVector = 0.50f,
            WeightGraph = 0.50f,
            WeightDecay = 0.10f,
            WeightImportance = 0.10f,
            HttpClient = httpClient
        });

        await engine.StoreTraceAsync(new EpisodicTrace
        {
            Id = "bridge",
            Content = "ServiceA communicates over Redis cluster",
            Entities = ["ServiceA", "Redis"],
            TimestampUs = 1_000_000L
        });
        await engine.StoreTraceAsync(new EpisodicTrace
        {
            Id = "multi-hop-target",
            Content = "Redis pool exhaustion caused Timeout errors",
            Entities = ["Redis", "Timeout"],
            TimestampUs = 1_000_000L
        });
        await engine.StoreTraceAsync(new EpisodicTrace
        {
            Id = "disconnected",
            Content = "Billing Invoice job completed normally",
            Entities = ["Billing", "Invoice"],
            TimestampUs = 1_000_000L
        });

        for (int i = 0; i < 4; i++)
        {
            await engine.StoreTraceAsync(new EpisodicTrace
            {
                Id = $"noise-{i}",
                Content = $"Unrelated background telemetry event {i}",
                Entities = [$"Noise{i}"],
                TimestampUs = 1_000_000L
            });
        }

        MemoryQueryResult result = await engine.QueryAsync(
            "investigating latency",
            queryEntities: ["ServiceA"],
            currentTimestampUs: 1_000_000L);

        int multiHopIdx = result.SelectedTraces.ToList().FindIndex(t => t.Id == "multi-hop-target");
        int disconnectedIdx = result.SelectedTraces.ToList().FindIndex(t => t.Id == "disconnected");

        Assert.True(multiHopIdx >= 0);
        Assert.True(disconnectedIdx >= 0);
        Assert.True(multiHopIdx < disconnectedIdx);
    }

    [Fact]
    public async Task T6_DimensionMismatchSafety_And_InstanceConsistency()
    {
        // Case A: Explicit 64-dim trace queried with mismatched 128-dim vector -> no crash, no NaN, PPR still works
        using (var engineA = new MemExEngine(new MemExOptions { EnableNpuEmbeddings = false }))
        {
            engineA.StoreTrace(new EpisodicTrace
            {
                Id = "alpha-64",
                Content = "Alpha trace at 64 dims",
                Embedding = CreateUnitBasisVector(64, 0),
                Entities = ["Alpha"],
                TimestampUs = 1_000_000L
            });

            MemoryQueryResult res = engineA.Query(
                "Alpha",
                CreateUnitBasisVector(128, 0),
                queryEntities: ["Alpha"],
                currentTimestampUs: 1_000_000L);

            Assert.Single(res.SelectedTraces);
            Assert.True(double.IsFinite(res.TotalUtility));

            Assert.Throws<ArgumentException>(() =>
                engineA.Query("bad", [float.NaN, 1.0f]));
        }

        // Case B: NPU probes 768 dims and stores Trace 1; mid-session NPU returns wrong length (512) then HTTP 500.
        // Engine must preserve 768-dim consistency via FastHashEmbeddingModel(768) fallback!
        int mode = 0; // 0 = healthy 768, 1 = wrong length 512, 2 = HTTP 500
        using var handler = new StubHttpMessageHandler(async (req, ct) =>
        {
            string path = req.RequestUri?.AbsolutePath ?? "";
            if (req.Method == HttpMethod.Get && path == "/v1/models")
            {
                return JsonResponse("""{"data":[{"id":"embeddinggemma"}]}""");
            }

            if (req.Method == HttpMethod.Post && path == "/v1/embeddings")
            {
                int currentMode = Volatile.Read(ref mode);
                if (currentMode == 0)
                {
                    string body = await req.Content!.ReadAsStringAsync(ct).ConfigureAwait(false);
                    using var doc = JsonDocument.Parse(body);
                    string input = doc.RootElement.GetProperty("input").GetString() ?? "";
                    return JsonResponse(SerializeEmbeddingResponse(BuildSemanticVector(input, 768)));
                }

                if (currentMode == 1)
                {
                    return JsonResponse(SerializeEmbeddingResponse(new float[512]));
                }

                return new HttpResponseMessage(HttpStatusCode.InternalServerError);
            }

            return new HttpResponseMessage(HttpStatusCode.NotFound);
        });

        using var httpClient = new HttpClient(handler);
        using var engineB = new MemExEngine(new MemExOptions
        {
            EnableNpuEmbeddings = true,
            EmbeddingEndpoint = "http://127.0.0.1:18086",
            HttpClient = httpClient
        });

        var trace1 = new EpisodicTrace
        {
            Id = "trace-1",
            Content = "cardiac arrest emergency wards", // 30 chars -> 7 tokens
            TimestampUs = 1_000_000L
        };
        await engineB.StoreTraceAsync(trace1);
        Assert.Equal(768, trace1.Embedding!.Length);

        // Wrong vector length mid-session (512 != 768)
        Volatile.Write(ref mode, 1);
        var trace2 = new EpisodicTrace
        {
            Id = "trace-2",
            Content = "kubernetes deployment manifest", // 30 chars -> 7 tokens
            TimestampUs = 1_000_000L
        };
        await engineB.StoreTraceAsync(trace2);
        Assert.NotNull(trace2.Embedding);
        Assert.Equal(768, trace2.Embedding!.Length);

        // HTTP 500 during query -> falls back cleanly at locked 768 dims without NaN or throwing
        Volatile.Write(ref mode, 2);
        MemoryQueryResult queryRes = await engineB.QueryAsync(
            "kubernetes deployment",
            currentTimestampUs: 1_000_000L);

        Assert.Equal(2, queryRes.SelectedTraces.Count);
        Assert.True(double.IsFinite(queryRes.TotalUtility));
        Assert.Equal("trace-2", queryRes.SelectedTraces[0].Id);

        // Case C: NPU fails on first StoreTraceAsync while _traces is empty -> demotes permanently to FastHashEmbeddingModel
        using var engineC = new MemExEngine(new MemExOptions
        {
            EnableNpuEmbeddings = true,
            EmbeddingEndpoint = "http://127.0.0.1:18086",
            EmbeddingModelId = "embeddinggemma",
            EmbeddingDimensions = 768,
            HttpClient = httpClient
        });
        Volatile.Write(ref mode, 1); // wrong-length 512 on first embedding call (expected 768)
        var traceC = new EpisodicTrace { Id = "c1", Content = "kubernetes deployment", TimestampUs = 1_000_000L };
        await engineC.StoreTraceAsync(traceC);
        Assert.False(engineC.IsUsingNpuEmbeddings);
        Assert.Equal("fast-hash", engineC.ActiveEmbeddingModelId);
        Assert.Equal(768, traceC.Embedding!.Length);
    }

    [Fact]
    public async Task T7_OfflineOrFailingNpuSidecar_FallsBackCleanlyToFastHash()
    {
        // (a) Unreachable closed localhost port
        int closedPort = GetUnusedTcpPort();
        using (var unreachableEngine = new MemExEngine(new MemExOptions
        {
            EnableNpuEmbeddings = true,
            EmbeddingEndpoint = $"http://127.0.0.1:{closedPort}",
            ProbeTimeout = TimeSpan.FromMilliseconds(150),
            EmbeddingTimeout = TimeSpan.FromMilliseconds(200)
        }))
        {
            await AssertFallbackWorksAsync(unreachableEngine);
        }

        // (b) HTTP 500 & invalid JSON
        using (var invalidJsonHandler = new StubHttpMessageHandler((_, _) =>
            Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent("{not valid json", Encoding.UTF8, "application/json")
            })))
        using (var invalidJsonClient = new HttpClient(invalidJsonHandler))
        using (var badJsonEngine = new MemExEngine(new MemExOptions
        {
            EnableNpuEmbeddings = true,
            EmbeddingEndpoint = "http://127.0.0.1:18087",
            HttpClient = invalidJsonClient
        }))
        {
            await AssertFallbackWorksAsync(badJsonEngine);
        }

        // (c) Timeout fallback (without caller cancellation), including caller HttpClient + explicit EmbeddingDimensions
        using (var slowHandler = new StubHttpMessageHandler(async (_, ct) =>
        {
            await Task.Delay(TimeSpan.FromSeconds(5), ct).ConfigureAwait(false);
            return new HttpResponseMessage(HttpStatusCode.OK);
        }))
        using (var slowClient = new HttpClient(slowHandler))
        using (var timeoutEngine = new MemExEngine(new MemExOptions
        {
            EnableNpuEmbeddings = true,
            EmbeddingEndpoint = "http://127.0.0.1:18088",
            ProbeTimeout = TimeSpan.FromMilliseconds(60),
            EmbeddingTimeout = TimeSpan.FromMilliseconds(60),
            HttpClient = slowClient
        }))
        {
            await AssertFallbackWorksAsync(timeoutEngine);
        }

        using (var slowEmbedHandler = new StubHttpMessageHandler(async (_, ct) =>
        {
            await Task.Delay(TimeSpan.FromSeconds(5), ct).ConfigureAwait(false);
            return new HttpResponseMessage(HttpStatusCode.OK);
        }))
        using (var slowEmbedClient = new HttpClient(slowEmbedHandler))
        using (var explicitDimsTimeoutEngine = new MemExEngine(new MemExOptions
        {
            EnableNpuEmbeddings = true,
            EmbeddingEndpoint = "http://127.0.0.1:18090",
            EmbeddingModelId = "embeddinggemma",
            EmbeddingDimensions = 768,
            EmbeddingTimeout = TimeSpan.FromMilliseconds(60),
            HttpClient = slowEmbedClient
        }))
        {
            var t = new EpisodicTrace
            {
                Id = "timeout-match",
                Content = "Distributed consensus raft leader election protocol",
                TimestampUs = 1_000_000L
            };
            await explicitDimsTimeoutEngine.StoreTraceAsync(t);
            Assert.False(explicitDimsTimeoutEngine.IsUsingNpuEmbeddings);
            Assert.Equal("fast-hash", explicitDimsTimeoutEngine.ActiveEmbeddingModelId);
            Assert.Equal(768, t.Embedding!.Length);
        }

        // (d) Invalid URI / unsupported scheme
        using (var invalidUriEngine = new MemExEngine(new MemExOptions
        {
            EnableNpuEmbeddings = true,
            EmbeddingEndpoint = "ftp://127.0.0.1:52625"
        }))
        {
            await AssertFallbackWorksAsync(invalidUriEngine);
        }

        // (e) Caller cancellation MUST still propagate OperationCanceledException
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        using var cancelEngine = new MemExEngine(new MemExOptions { EnableNpuEmbeddings = false });
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
            await cancelEngine.StoreTraceAsync(new EpisodicTrace { Content = "cancelled" }, cts.Token));

        // (f) Parameterless new MemExEngine() works cleanly offline
        using var defaultEngine = new MemExEngine();
        defaultEngine.StoreTrace(new EpisodicTrace { Id = "d1", Content = "vulkan compute shader pipeline", TimestampUs = 1_000_000L });
        defaultEngine.StoreTrace(new EpisodicTrace { Id = "d2", Content = "chocolate cake baking recipe", TimestampUs = 1_000_000L });
        MemoryQueryResult defaultRes = defaultEngine.Query("vulkan shader", ReadOnlySpan<float>.Empty, currentTimestampUs: 1_000_000L);
        Assert.Equal("d1", defaultRes.SelectedTraces[0].Id);
    }

    [Fact]
    public async Task T8_EpisodicTrace_TokenCount_InitializerAndAutoEstimate_NoReflection()
    {
        using var engine = new MemExEngine(new MemExOptions { EnableNpuEmbeddings = false });

        var t1 = new EpisodicTrace
        {
            Id = "explicit",
            Content = new string('a', 400),
            TokenCount = 42,
            BaseImportance = 5.0f,
            TimestampUs = 1_000_000L
        };
        var t2 = new EpisodicTrace
        {
            Id = "estimated",
            Content = new string('b', 400),
            BaseImportance = 4.0f,
            TimestampUs = 1_000_000L
        };
        var t3 = new EpisodicTrace
        {
            Id = "whitespace",
            Content = "   ",
            BaseImportance = 0.1f,
            TimestampUs = 1_000_000L
        };

        engine.StoreTrace(t1);
        await engine.StoreTraceAsync(t2);
        engine.StoreTrace(t3);

        Assert.Equal(42, t1.TokenCount);
        Assert.Equal(100, t2.TokenCount);
        Assert.Equal(0, t3.TokenCount);
        Assert.Null(t3.Embedding);

        MemoryQueryResult budgeted = await engine.QueryAsync(
            "aaaa",
            maxTokens: 120,
            currentTimestampUs: 1_000_000L);

        Assert.True(budgeted.TotalTokens <= 120);
        Assert.Equal(budgeted.SelectedTraces.Sum(t => t.TokenCount), budgeted.TotalTokens);
    }

    [Fact]
    public async Task T9_Concurrent_StoreAndQueryAsync_ThreadSafety_And_Lifecycle()
    {
        using var handler = new StubHttpMessageHandler(async (req, ct) =>
        {
            string path = req.RequestUri?.AbsolutePath ?? "";
            if (req.Method == HttpMethod.Get && path == "/v1/models")
            {
                return JsonResponse("""{"data":[{"id":"embeddinggemma"}]}""");
            }

            if (req.Method == HttpMethod.Post && path == "/v1/embeddings")
            {
                string body = await req.Content!.ReadAsStringAsync(ct).ConfigureAwait(false);
                using var doc = JsonDocument.Parse(body);
                string input = doc.RootElement.GetProperty("input").GetString() ?? "";
                return JsonResponse(SerializeEmbeddingResponse(BuildSemanticVector(input, 768)));
            }

            return new HttpResponseMessage(HttpStatusCode.NotFound);
        });

        using var callerHttpClient = new HttpClient(handler);
        var engine = new MemExEngine(new MemExOptions
        {
            EnableNpuEmbeddings = true,
            EmbeddingEndpoint = "http://127.0.0.1:18089",
            HttpClient = callerHttpClient
        });

        const int workers = 16;
        Task[] tasks = new Task[workers];
        for (int i = 0; i < workers; i++)
        {
            int idx = i;
            tasks[i] = Task.Run(async () =>
            {
                await engine.StoreTraceAsync(new EpisodicTrace
                {
                    Id = $"concurrent-{idx}",
                    Content = idx % 2 == 0
                        ? $"cardiac arrest observation {idx}"
                        : $"kubernetes pod log {idx}",
                    Entities = [$"Entity{idx}"],
                    TimestampUs = 1_000_000L
                });

                MemoryQueryResult res = await engine.QueryAsync(
                    "myocardial infarction",
                    queryEntities: [$"Entity{idx}"],
                    currentTimestampUs: 1_000_000L);

                Assert.True(double.IsFinite(res.TotalUtility));
            });
        }

        await Task.WhenAll(tasks);

        Assert.Equal(workers, engine.Count);
        Assert.Equal(768, engine.ActiveEmbeddingDimensions);
        Assert.True(engine.IsUsingNpuEmbeddings);

        engine.Dispose();

        // Caller-owned HttpClient must remain usable (not disposed by MemExEngine)
        using HttpResponseMessage check = await callerHttpClient.GetAsync("http://127.0.0.1:18089/v1/models");
        Assert.Equal(HttpStatusCode.OK, check.StatusCode);

        Assert.Throws<ObjectDisposedException>(() => engine.StoreTrace(new EpisodicTrace { Content = "after dispose" }));
        await Assert.ThrowsAsync<ObjectDisposedException>(() => engine.QueryAsync("after dispose"));
    }

    private static async Task AssertFallbackWorksAsync(MemExEngine engine)
    {
        var match = new EpisodicTrace
        {
            Id = "match",
            Content = "Distributed consensus raft leader election protocol",
            TimestampUs = 1_000_000L,
            BaseImportance = 1.0f
        };
        var other = new EpisodicTrace
        {
            Id = "other",
            Content = "Watercolor landscape painting brush techniques",
            TimestampUs = 1_000_000L,
            BaseImportance = 1.0f
        };

        await engine.StoreTraceAsync(match);
        await engine.StoreTraceAsync(other);

        Assert.False(engine.IsUsingNpuEmbeddings);
        Assert.Equal(384, engine.ActiveEmbeddingDimensions);
        Assert.Equal("fast-hash", engine.ActiveEmbeddingModelId);
        Assert.Null(engine.ActiveEmbeddingEndpoint);
        Assert.NotNull(match.Embedding);
        Assert.Equal(384, match.Embedding!.Length);

        MemoryQueryResult res = await engine.QueryAsync(
            "raft leader election",
            currentTimestampUs: 1_000_000L);

        Assert.Equal("match", res.SelectedTraces[0].Id);
    }

    private static float[] BuildSemanticVector(string text, int dimensions)
    {
        var vec = new float[dimensions];
        if (text.Contains("cardiac", StringComparison.OrdinalIgnoreCase) ||
            text.Contains("myocardial", StringComparison.OrdinalIgnoreCase))
        {
            vec[0] = 0.98f;
            vec[1] = 0.198997f;
        }
        else if (text.Contains("kubernetes", StringComparison.OrdinalIgnoreCase) ||
                 text.Contains("pod", StringComparison.OrdinalIgnoreCase))
        {
            vec[2] = 1.0f;
        }
        else
        {
            vec[3] = 1.0f;
        }
        return vec;
    }

    private static float[] CreateUnitBasisVector(int dimensions, int activeIndex)
    {
        var vec = new float[dimensions];
        vec[activeIndex] = 1.0f;
        return vec;
    }

    private static HttpResponseMessage JsonResponse(string json) =>
        new(HttpStatusCode.OK)
        {
            Content = new StringContent(json, Encoding.UTF8, "application/json")
        };

    private static string SerializeEmbeddingResponse(float[] vector)
    {
        var sb = new StringBuilder(vector.Length * 6 + 48);
        sb.Append("""{"data":[{"embedding":[""");
        for (int i = 0; i < vector.Length; i++)
        {
            if (i > 0) sb.Append(',');
            sb.Append(vector[i].ToString("G9", System.Globalization.CultureInfo.InvariantCulture));
        }
        sb.Append("]}]}");
        return sb.ToString();
    }

    private static int GetUnusedTcpPort()
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    private sealed class StubHttpMessageHandler : HttpMessageHandler
    {
        private readonly Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> _responder;

        public StubHttpMessageHandler(Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> responder)
        {
            _responder = responder;
        }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            _responder(request, cancellationToken);
    }

    private sealed class NonCapturingTestSyncContext : SynchronizationContext
    {
        public override void Post(SendOrPostCallback d, object? state) =>
            ThreadPool.QueueUserWorkItem(static s =>
            {
                var (cb, arg) = ((SendOrPostCallback, object?))s!;
                cb(arg);
            }, (d, state));

        public override void Send(SendOrPostCallback d, object? state) =>
            throw new InvalidOperationException("Synchronous Wait on captured SynchronizationContext would deadlock.");
    }

    private sealed class LoopbackEmbeddingServer : IDisposable
    {
        private readonly TcpListener _listener;
        private readonly CancellationTokenSource _cts = new();
        private readonly Task _acceptTask;
        private readonly int _dimensions;
        private readonly string _modelId;
        private readonly Func<string, int, float[]> _embedder;

        public string BaseUrl { get; }

        public LoopbackEmbeddingServer(int dimensions, string modelId, Func<string, int, float[]> embedder)
        {
            _dimensions = dimensions;
            _modelId = modelId;
            _embedder = embedder;
            _listener = new TcpListener(IPAddress.Loopback, 0);
            _listener.Start();
            int port = ((IPEndPoint)_listener.LocalEndpoint).Port;
            BaseUrl = $"http://127.0.0.1:{port}";
            _acceptTask = Task.Run(() => AcceptLoopAsync(_cts.Token));
        }

        private async Task AcceptLoopAsync(CancellationToken ct)
        {
            while (!ct.IsCancellationRequested)
            {
                TcpClient client;
                try
                {
                    client = await _listener.AcceptTcpClientAsync(ct).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (ObjectDisposedException)
                {
                    break;
                }

                _ = Task.Run(() => HandleClientAsync(client, ct), ct);
            }
        }

        private async Task HandleClientAsync(TcpClient client, CancellationToken ct)
        {
            using (client)
            using (NetworkStream stream = client.GetStream())
            {
                var buffer = new byte[16384];
                var sb = new StringBuilder();
                int headerEnd = -1;
                int contentLength = 0;

                while (!ct.IsCancellationRequested)
                {
                    int read = await stream.ReadAsync(buffer, ct).ConfigureAwait(false);
                    if (read <= 0)
                        return;

                    sb.Append(Encoding.UTF8.GetString(buffer, 0, read));
                    string current = sb.ToString();

                    if (headerEnd < 0)
                    {
                        headerEnd = current.IndexOf("\r\n\r\n", StringComparison.Ordinal);
                        if (headerEnd >= 0)
                        {
                            string headerText = current[..headerEnd];
                            foreach (string line in headerText.Split("\r\n"))
                            {
                                if (line.StartsWith("Content-Length:", StringComparison.OrdinalIgnoreCase))
                                {
                                    int.TryParse(line.AsSpan("Content-Length:".Length).Trim(), out contentLength);
                                }
                            }
                        }
                    }

                    if (headerEnd >= 0)
                    {
                        int totalExpected = Encoding.UTF8.GetByteCount(current[..(headerEnd + 4)]) + contentLength;
                        if (Encoding.UTF8.GetByteCount(current) >= totalExpected)
                        {
                            string headerSection = current[..headerEnd];
                            string requestLine = headerSection.Split("\r\n")[0];
                            string body = current[(headerEnd + 4)..];

                            string responseJson;
                            string statusLine = "HTTP/1.1 200 OK";

                            if (requestLine.StartsWith("GET /v1/models ", StringComparison.Ordinal))
                            {
                                responseJson = $$"""{"data":[{"id":"{{_modelId}}"}]}""";
                            }
                            else if (requestLine.StartsWith("POST /v1/embeddings ", StringComparison.Ordinal))
                            {
                                using var doc = JsonDocument.Parse(body);
                                string input = doc.RootElement.GetProperty("input").GetString() ?? "";
                                responseJson = SerializeEmbeddingResponse(_embedder(input, _dimensions));
                            }
                            else
                            {
                                statusLine = "HTTP/1.1 404 Not Found";
                                responseJson = """{"error":"not found"}""";
                            }

                            byte[] bodyBytes = Encoding.UTF8.GetBytes(responseJson);
                            byte[] headerBytes = Encoding.ASCII.GetBytes(
                                $"{statusLine}\r\nContent-Type: application/json\r\nContent-Length: {bodyBytes.Length}\r\nConnection: close\r\n\r\n");

                            await stream.WriteAsync(headerBytes, ct).ConfigureAwait(false);
                            await stream.WriteAsync(bodyBytes, ct).ConfigureAwait(false);
                            await stream.FlushAsync(ct).ConfigureAwait(false);
                            return;
                        }
                    }
                }
            }
        }

        public void Dispose()
        {
            _cts.Cancel();
            _listener.Stop();
            try
            {
                _acceptTask.Wait(TimeSpan.FromSeconds(2));
            }
            catch (Exception ex) when (ex is OperationCanceledException or ObjectDisposedException or AggregateException or SocketException)
            {
            }
            _cts.Dispose();
        }
    }
}
