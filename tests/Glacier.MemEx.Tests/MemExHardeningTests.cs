namespace Glacier.MemEx.Tests;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Glacier.Bundle.Budgeting;
using Glacier.MemEx;
using Glacier.MemEx.Budgeting;
using Glacier.MemEx.Models;
using Glacier.MemEx.Recall;
using Xunit;

public class MemExHardeningTests
{
    [Fact]
    public void HippoRag_DanglingSeedTeleportsStrictlyToP0_AndDeduplicatesQuerySeeds()
    {
        var traces = new List<EpisodicTrace>
        {
            new() { Id = "t-dangle-seed", Content = "Isolated seed", Entities = ["SeedDangle"] },
            new() { Id = "t-other-conn", Content = "Unrelated connected", Entities = ["OtherA", "OtherB"] },
            new() { Id = "t-other-dangle", Content = "Unrelated isolated", Entities = ["OtherDangle"] }
        };

        Dictionary<string, double> ppr = HippoRagSpreadingActivation.ComputePersonalizedPageRank(
            ["SeedDangle"],
            traces,
            dampingAlpha: 0.85,
            maxIterations: 25);

        Assert.Equal(4, ppr.Count);
        Assert.Equal(1.0, ppr["SeedDangle"], precision: 5);
        Assert.Equal(0.0, ppr["OtherA"]);
        Assert.Equal(0.0, ppr["OtherB"]);
        Assert.Equal(0.0, ppr["OtherDangle"]);
        Assert.Equal(1.0, ppr.Values.Sum(), precision: 5);

        // Duplicate query entities must not over-weight SeedA relative to SeedB or break sum == 1.0
        var symmetricTraces = new List<EpisodicTrace>
        {
            new() { Id = "s1", Content = "A-NA", Entities = ["SeedA", "NeighborA"] },
            new() { Id = "s2", Content = "B-NB", Entities = ["SeedB", "NeighborB"] },
            new() { Id = "s3", Content = "U", Entities = ["Unrelated1", "Unrelated2"] }
        };

        Dictionary<string, double> dupPpr = HippoRagSpreadingActivation.ComputePersonalizedPageRank(
            ["SeedA", "seeda", "SEEDA", "SeedB"],
            symmetricTraces,
            dampingAlpha: 0.85,
            maxIterations: 25);

        Assert.Equal(1.0, dupPpr.Values.Sum(), precision: 5);
        Assert.Equal(dupPpr["SeedA"], dupPpr["SeedB"], precision: 5);
        Assert.Equal(dupPpr["NeighborA"], dupPpr["NeighborB"], precision: 5);
        Assert.Equal(0.0, dupPpr["Unrelated1"]);
        Assert.Equal(0.0, dupPpr["Unrelated2"]);

        // maxIterations <= 0 returns p0 cleanly without NaN
        Dictionary<string, double> zeroIterPpr = HippoRagSpreadingActivation.ComputePersonalizedPageRank(
            ["SeedA", "SeedA"],
            symmetricTraces,
            dampingAlpha: 0.85,
            maxIterations: 0);
        Assert.Equal(1.0, zeroIterPpr.Values.Sum(), precision: 5);
        Assert.Equal(1.0, zeroIterPpr["SeedA"], precision: 5);
    }

    [Fact]
    public void EntityCsrGraph_DeduplicatesSortsAndSkipsSelfLoops()
    {
        var traces = new List<EpisodicTrace>
        {
            new() { Content = "dup", Entities = ["Zebra", "zebra", "Alpha", "Middle"] },
            new() { Content = "second", Entities = ["Alpha", "Beta", ""] }
        };

        EntityCsrGraph graph = EntityCsrGraph.Build(traces, version: 7);
        Assert.Equal(7, graph.Version);
        Assert.Equal(4, graph.NodeCount);
        Assert.Equal(graph.NodeCount + 2, graph.RowOffsets.Length);

        for (int u = 1; u <= graph.NodeCount; u++)
        {
            int start = graph.RowOffsets[u];
            int end = graph.RowOffsets[u + 1];
            int prev = 0;
            for (int k = start; k < end; k++)
            {
                int v = graph.ColumnIndices[k];
                Assert.NotEqual(u, v); // no self-loops
                Assert.True(v > prev, "CSR column indices must be strictly sorted and deduplicated.");
                prev = v;
            }
        }
    }

    [Fact]
    public async Task MemExEngine_CachedEntityGraph_InvalidatesOnMutations_AndIsThreadSafe()
    {
        using var engine = new MemExEngine(new MemExOptions
        {
            EnableNpuEmbeddings = false,
            AutoEmbedMissingVectors = false,
            WeightDecay = 0.0f,
            WeightVector = 0.0f,
            WeightGraph = 2.0f,
            WeightImportance = 0.0f
        });

        long t0 = 1_000_000_000L;
        engine.StoreTrace(new EpisodicTrace
        {
            Id = "t1",
            Content = "Bridge node",
            Entities = ["Alpha", "Beta"],
            TokenCount = 10,
            TimestampUs = t0
        });

        // Prime cached EntityCsrGraph (version 1)
        MemoryQueryResult r1 = engine.Query("q", ReadOnlySpan<float>.Empty, queryEntities: ["Alpha"], currentTimestampUs: t0, maxTokens: 50);
        Assert.Single(r1.SelectedTraces);

        // Add multi-hop trace -> invalidates cache
        engine.StoreTrace(new EpisodicTrace
        {
            Id = "t2",
            Content = "Hop-2 target",
            Entities = ["Beta", "Gamma"],
            TokenCount = 10,
            TimestampUs = t0
        });
        engine.StoreTrace(new EpisodicTrace
        {
            Id = "t-unrelated",
            Content = "Isolated node",
            Entities = ["Other"],
            TokenCount = 10,
            TimestampUs = t0
        });

        MemoryQueryResult r2 = engine.Query("q", ReadOnlySpan<float>.Empty, queryEntities: ["Alpha"], currentTimestampUs: t0, maxTokens: 25);
        Assert.Equal(2, r2.SelectedTraces.Count);
        Assert.DoesNotContain(r2.SelectedTraces, t => t.Id == "t-unrelated");

        // Replace t1 with unrelated entity -> Alpha <-> Beta link is gone
        engine.StoreTrace(new EpisodicTrace
        {
            Id = "t1",
            Content = "Replaced node",
            Entities = ["Delta"],
            TokenCount = 10,
            TimestampUs = t0
        });

        MemoryQueryResult r3 = engine.Query("q", ReadOnlySpan<float>.Empty, queryEntities: ["Delta"], currentTimestampUs: t0, maxTokens: 15);
        Assert.Single(r3.SelectedTraces);
        Assert.Equal("t1", r3.SelectedTraces[0].Id);

        // PruneTraces and PruneDecayedTraces invalidate cache cleanly
        Assert.Equal(1, engine.PruneTraces(["t2"]));
        engine.StoreTrace(new EpisodicTrace
        {
            Id = "t-old",
            Content = "Very old trace",
            Entities = ["Delta", "OldTarget"],
            TokenCount = 10,
            TimestampUs = 1_000L
        });
        Assert.True(engine.PruneDecayedTraces(minRecencyScore: 0.4, currentTimestampUs: t0) >= 1);

        // Concurrent Store + Query + Prune stress
        var tasks = new List<Task>();
        for (int w = 0; w < 8; w++)
        {
            int worker = w;
            tasks.Add(Task.Run(() =>
            {
                for (int i = 0; i < 40; i++)
                {
                    string id = $"conc-{worker}-{i}";
                    engine.StoreTrace(new EpisodicTrace
                    {
                        Id = id,
                        Content = $"Concurrent trace {id}",
                        Entities = [$"Ent{worker}", $"Shared{i % 4}"],
                        TokenCount = 5,
                        TimestampUs = t0
                    });

                    MemoryQueryResult qRes = engine.Query("q", ReadOnlySpan<float>.Empty, queryEntities: [$"Ent{worker}"], currentTimestampUs: t0, maxTokens: 40);
                    Assert.True(double.IsFinite(qRes.TotalUtility));
                    Assert.Equal(qRes.SelectedTraces.Select(t => t.Id).Distinct(StringComparer.Ordinal).Count(), qRes.SelectedTraces.Count);

                    if (i % 5 == 0)
                    {
                        engine.PruneTraces([id]);
                    }
                }
            }));
        }

        await Task.WhenAll(tasks);
    }

    [Theory]
    [InlineData(25, false)]
    [InlineData(25, true)]
    [InlineData(40, false)]
    [InlineData(40, true)]
    [InlineData(100, false)]
    [InlineData(100, true)]
    [InlineData(200, false)]
    [InlineData(200, true)]
    public void BreadcrumbAndMemExEngine_FormattedContext_StrictlyEnforcesMaxTokens(int maxTokens, bool foldDeepSubtrees)
    {
        var estimator = new HeuristicTokenEstimator();
        using var engine = new MemExEngine(new MemExOptions
        {
            EnableNpuEmbeddings = false,
            AutoEmbedMissingVectors = false,
            TokenEstimator = estimator,
            FoldDeepSubtrees = foldDeepSubtrees
        });

        List<string> deepBreadcrumbs = Enumerable.Range(0, 25).Select(i => $"ModuleSegment_{i}").ToList();
        List<string> manyEntities = Enumerable.Range(0, 12).Select(i => $"EntityName_{i}").ToList();

        // Odd-length content (5 chars -> ceil(5/4) = 2 tokens) + large content (>240 chars)
        engine.StoreTrace(new EpisodicTrace
        {
            Id = "odd-small",
            Content = "hello",
            Breadcrumbs = deepBreadcrumbs,
            Entities = manyEntities,
            BaseImportance = 5.0f,
            TimestampUs = 1_000_000L
        });
        engine.StoreTrace(new EpisodicTrace
        {
            Id = "deep-large",
            Content = new string('X', 600),
            Breadcrumbs = deepBreadcrumbs,
            Entities = manyEntities,
            BaseImportance = 4.0f,
            TimestampUs = 1_000_000L
        });

        MemoryQueryResult res = engine.Query(
            "hello",
            ReadOnlySpan<float>.Empty,
            queryEntities: ["EntityName_0"],
            currentTimestampUs: 1_000_000L,
            maxTokens: maxTokens);

        Assert.True(res.TotalTokens <= maxTokens, $"TotalTokens {res.TotalTokens} > maxTokens {maxTokens}");
        Assert.Equal(res.SelectedTraces.Sum(t => t.TokenCount), res.TotalTokens);
        int formattedTokens = estimator.EstimateTokens(res.FormattedContext);
        Assert.True(formattedTokens <= maxTokens, $"FormattedContext tokens {formattedTokens} > maxTokens {maxTokens}");
    }

    [Fact]
    public void MemExOptions_FoldDeepSubtrees_And_OddContentEstimation_AreWired()
    {
        var estimator = new HeuristicTokenEstimator();
        using var engine = new MemExEngine(new MemExOptions
        {
            EnableNpuEmbeddings = false,
            AutoEmbedMissingVectors = false,
            FoldDeepSubtrees = true,
            TokenEstimator = estimator
        });

        var oddTrace = new EpisodicTrace { Id = "odd5", Content = "  abcde  ", BaseImportance = 1.0f };
        var longTrace = new EpisodicTrace { Id = "long", Content = new string('Z', 400), BaseImportance = 3.0f };

        engine.StoreTrace(oddTrace);
        engine.StoreTrace(longTrace);

        // HeuristicTokenEstimator uses Ceiling(5 / 4.0) == 2 instead of truncating 5 / 4 == 1
        Assert.Equal(2, oddTrace.TokenCount);
        Assert.Equal(100, longTrace.TokenCount);

        MemoryQueryResult res = engine.Query("test", ReadOnlySpan<float>.Empty, maxTokens: 500);
        Assert.Contains("... [folded subtree]", res.FormattedContext);
        Assert.True(estimator.EstimateTokens(res.FormattedContext) <= 500);
        Assert.Equal(string.Empty, BreadcrumbTreeFolder.FormatTraceContext([longTrace], maxTokens: 0, estimator));
    }

    [Fact]
    public void QueryCore_CoarseCandidateLimit_DoesNotZeroValidNonTopKTraces()
    {
        const int dim = 128;
        float[] query = new float[dim];
        query[0] = 1.0f;

        using var engine = new MemExEngine(new MemExOptions
        {
            EnableNpuEmbeddings = false,
            AutoEmbedMissingVectors = false,
            CoarseCandidateLimit = 8,
            WeightDecay = 0.0f,
            WeightVector = 0.35f,
            WeightGraph = 0.80f,
            WeightImportance = 0.0f
        });

        // 16 strong vector matches that have unrelated entities
        for (int i = 0; i < 16; i++)
        {
            float[] v = new float[dim];
            v[0] = 1.0f;
            v[1 + i] = 0.02f;
            engine.StoreTrace(new EpisodicTrace
            {
                Id = $"top-vec-{i}",
                Content = $"Top vec {i}",
                Embedding = v,
                Entities = [$"Noise_{i}"],
                TokenCount = 20,
                TimestampUs = 1_000_000L
            });
        }

        // 90 more positive-ish distractors so target-outside-top8 is outside CoarseCandidateLimit=8
        for (int i = 0; i < 90; i++)
        {
            float[] v = new float[dim];
            v[0] = 0.8f;
            v[20 + (i % 80)] = 0.4f;
            engine.StoreTrace(new EpisodicTrace
            {
                Id = $"distractor-{i}",
                Content = $"Distractor {i}",
                Embedding = v,
                Entities = [$"Dist_{i}"],
                TokenCount = 20,
                TimestampUs = 1_000_000L
            });
        }

        // Target trace outside top-8 Hamming matches (orthogonal embedding: cosine ~ 0 -> normVec ~ 0.5)
        float[] ortho = new float[dim];
        ortho[127] = 1.0f;
        var targetOutsideTop8 = new EpisodicTrace
        {
            Id = "target-outside-top8",
            Content = "Outside coarse limit but valid embedding + graph match",
            Embedding = ortho,
            Entities = ["SharedSeed"],
            TokenCount = 20,
            TimestampUs = 1_000_000L
        };

        // Anti-aligned trace with same entity (cosine = -1 -> normVec = 0.0)
        float[] anti = new float[dim];
        anti[0] = -1.0f;
        var antiAligned = new EpisodicTrace
        {
            Id = "anti-same-seed",
            Content = "Anti aligned + same graph match",
            Embedding = anti,
            Entities = ["SharedSeed"],
            TokenCount = 20,
            TimestampUs = 1_000_000L
        };

        // Unembedded trace with same entity (normVec = 0.0)
        var unembedded = new EpisodicTrace
        {
            Id = "unembedded-same-seed",
            Content = "Unembedded + same graph match",
            Embedding = null,
            Entities = ["SharedSeed"],
            TokenCount = 20,
            TimestampUs = 1_000_000L
        };

        engine.StoreTrace(targetOutsideTop8);
        engine.StoreTrace(antiAligned);
        engine.StoreTrace(unembedded);

        // Verify ScoreAllTraces directly: non-top-K valid trace gets finite Hamming estimate (not NaN), unembedded gets NaN
        var allTraces = new List<EpisodicTrace> { targetOutsideTop8, antiAligned, unembedded };
        for (int i = 0; i < 16; i++)
        {
            float[] v = new float[dim];
            v[0] = 1.0f;
            v[1 + i] = 0.01f;
            allTraces.Add(new EpisodicTrace { Id = $"x-{i}", Content = "x", Embedding = v });
        }
        Span<float> scores = new float[allTraces.Count];
        TwoTierVectorFilter.ScoreAllTraces(query, allTraces, coarseLimit: 8, scores);
        Assert.True(float.IsFinite(scores[0]), "Valid trace outside coarseLimit must retain finite Hamming cosine estimate.");
        Assert.True(scores[0] > -0.5f, $"Expected near-orthogonal estimate > -0.5, got {scores[0]}");
        Assert.True(float.IsNaN(scores[2]), "Unembedded trace must have NaN sentinel.");

        // Budget allows only 1 of the three 20-token SharedSeed traces
        MemoryQueryResult res = engine.Query(
            "q",
            query,
            queryEntities: ["SharedSeed"],
            currentTimestampUs: 1_000_000L,
            maxTokens: 25);

        Assert.Single(res.SelectedTraces);
        Assert.Equal("target-outside-top8", res.SelectedTraces[0].Id);
    }

    [Fact]
    public void TwoTierVectorFilter_FwhtSkewedHighDimRecall_MismatchedManualBinary_AndBoundedHeaps()
    {
        // 1. Word count & empty span invariants
        Assert.Empty(TwoTierVectorFilter.QuantizeTo1Bit(ReadOnlySpan<float>.Empty));
        Assert.Single(TwoTierVectorFilter.QuantizeTo1Bit(new float[64]));
        Assert.Equal(2, TwoTierVectorFilter.QuantizeTo1Bit(new float[100]).Length);

        // 2. Skewed non-centered 384-dim embeddings (DC bias +5.0 across all dims)
        const int dim = 384;
        var rng = new Random(12345);
        float[] query = new float[dim];
        for (int d = 0; d < dim; d++) query[d] = 5.0f;
        for (int d = 0; d < 24; d++) query[d] += 2.0f;

        var traces = new List<EpisodicTrace>(256);
        for (int i = 0; i < 255; i++)
        {
            float[] v = new float[dim];
            for (int d = 0; d < dim; d++)
            {
                v[d] = 5.0f + (float)(rng.NextDouble() * 0.4 - 0.2);
            }
            int block = (i % 10) * 32 + 32;
            for (int d = block; d < Math.Min(dim, block + 24); d++) v[d] += 1.5f;

            traces.Add(new EpisodicTrace { Id = $"skew-{i}", Content = $"Skew {i}", Embedding = v });
        }

        // True match + manually populated wrong-length BinaryEmbedding to verify safe handling
        float[] best = (float[])query.Clone();
        best[0] += 0.01f;
        var bestTrace = new EpisodicTrace
        {
            Id = "skew-best",
            Content = "Best skewed match",
            Embedding = best,
            BinaryEmbedding = [1UL, 2UL, 3UL] // intentionally wrong word count vs 384-dim FWHT (6 words)
        };
        traces.Add(bestTrace);

        List<(EpisodicTrace Trace, float CosineSimilarity)> top = TwoTierVectorFilter.FilterAndRerank(
            query,
            traces,
            coarseLimit: 32,
            finalLimit: 5);

        Assert.Equal(5, top.Count);
        Assert.Equal("skew-best", top[0].Trace.Id);
        Assert.True(top[0].CosineSimilarity > 0.99f);

        // 3. Verify BoundedMaxIntHeap & BoundedMinFloatHeap parity against full sort
        const int n = 150;
        const int k = 12;
        var intPairs = Enumerable.Range(0, n).Select(i => (Index: i, Dist: rng.Next(0, 1000))).ToArray();
        var maxIntBuf = new (int Index, int Dist)[k];
        var maxHeap = new BoundedMaxIntHeap(maxIntBuf, k);
        foreach (var (idx, dist) in intPairs) maxHeap.Offer(idx, dist);

        int[] expectedSmallest = intPairs.OrderBy(p => p.Dist).Take(k).Select(p => p.Dist).OrderBy(x => x).ToArray();
        int[] actualSmallest = maxIntBuf.Take(maxHeap.Count).Select(p => p.Dist).OrderBy(x => x).ToArray();
        Assert.Equal(expectedSmallest, actualSmallest);

        var floatPairs = Enumerable.Range(0, n).Select(i => (Index: i, Score: (float)(rng.NextDouble() * 2.0 - 1.0))).ToArray();
        var minFloatBuf = new (int Index, float Score)[k];
        var minHeap = new BoundedMinFloatHeap(minFloatBuf, k);
        foreach (var (idx, score) in floatPairs) minHeap.Offer(idx, score);

        float[] expectedLargest = floatPairs.OrderByDescending(p => p.Score).Take(k).Select(p => p.Score).ToArray();
        var actualLargest = new float[minHeap.Count];
        for (int i = minHeap.Count - 1; i >= 0; i--)
        {
            Assert.True(minHeap.TryDequeue(out _, out float s));
            actualLargest[i] = s;
        }
        Assert.Equal(expectedLargest, actualLargest);
    }

    [Fact]
    public void KnapsackPromptPacker_PooledReuse_IntMaxBudget_And_BestSingleFallback()
    {
        // 1. Sequential DP calls with varying budgets must not leak dirty ArrayPool state
        var itemsA = new List<(EpisodicTrace Trace, double Utility)>
        {
            (new EpisodicTrace { Id = "a1", TokenCount = 10 }, 100.0),
            (new EpisodicTrace { Id = "a2", TokenCount = 20 }, 180.0),
            (new EpisodicTrace { Id = "a3", TokenCount = 30 }, 240.0)
        };
        var itemsB = new List<(EpisodicTrace Trace, double Utility)>
        {
            (new EpisodicTrace { Id = "b1", TokenCount = 40 }, 50.0),
            (new EpisodicTrace { Id = "b2", TokenCount = 10 }, 20.0)
        };

        List<EpisodicTrace> pass1 = KnapsackPromptPacker.PackTraces(itemsA, maxTokens: 50);
        Assert.Equal(["a2", "a3"], pass1.Select(t => t.Id));

        List<EpisodicTrace> pass2 = KnapsackPromptPacker.PackTraces(itemsB, maxTokens: 15);
        Assert.Single(pass2);
        Assert.Equal("b2", pass2[0].Id);

        List<EpisodicTrace> pass3 = KnapsackPromptPacker.PackTraces(itemsA, maxTokens: 50);
        Assert.Equal(["a2", "a3"], pass3.Select(t => t.Id));

        // 2. int.MaxValue budget must not overflow stride = maxTokens + 1
        List<EpisodicTrace> maxBudget = KnapsackPromptPacker.PackTraces(itemsA, maxTokens: int.MaxValue);
        Assert.Equal(3, maxBudget.Count);

        // 3. Large DP cell count (> 200_000) triggers greedy vs bestSingle fallback:
        //    greedy picks decoy (w=1, u=10, density=10), leaving 999 tokens -> misses huge (w=1000, u=5000, density=5).
        var largeCandidates = new List<(EpisodicTrace Trace, double Utility)>(250)
        {
            (new EpisodicTrace { Id = "greedy-decoy", TokenCount = 1 }, 10.0),
            (new EpisodicTrace { Id = "best-single", TokenCount = 1000 }, 5000.0)
        };
        for (int i = 0; i < 248; i++)
        {
            largeCandidates.Add((new EpisodicTrace { Id = $"filler-{i}", TokenCount = 500 }, 1.0));
        }

        List<EpisodicTrace> fallback = KnapsackPromptPacker.PackTraces(largeCandidates, maxTokens: 1000);
        Assert.Single(fallback);
        Assert.Equal("best-single", fallback[0].Id);
    }

    [Fact]
    public void QueryCore_DescendingUtilityOrder_And_ExportConsolidationCandidates_BoundedHeapParity()
    {
        long nowUs = 10_000_000_000L;
        using var engine = new MemExEngine(new MemExOptions
        {
            EnableNpuEmbeddings = false,
            AutoEmbedMissingVectors = false,
            WeightDecay = 0.5f,
            WeightVector = 0.0f,
            WeightGraph = 0.0f,
            WeightImportance = 0.5f
        });

        for (int i = 0; i < 40; i++)
        {
            var t = new EpisodicTrace
            {
                Id = $"item-{i}",
                Content = $"Consolidation item {i}",
                BaseImportance = 0.2f + (i % 9) * 0.5f,
                TokenCount = 5,
                TimestampUs = nowUs - (i * 10_000_000L)
            };
            if (i % 2 == 0) t.RecordAccess(nowUs - 1_000_000L);
            engine.StoreTrace(t);
        }

        MemoryQueryResult qRes = engine.Query("test", ReadOnlySpan<float>.Empty, currentTimestampUs: nowUs, maxTokens: 30);
        Assert.Equal(6, qRes.SelectedTraces.Count);
        Assert.Equal(6, qRes.SelectedTraces.Select(t => t.Id).Distinct(StringComparer.Ordinal).Count());
        Assert.True(qRes.TotalUtility > 0.0 && double.IsFinite(qRes.TotalUtility));

        IReadOnlyList<EpisodicTrace> exported = engine.ExportConsolidationCandidates(maxCount: 5, minAccessCount: 1, currentTimestampUs: nowUs);
        Assert.Equal(5, exported.Count);
        Assert.All(exported, t => Assert.True(t.AccessCount >= 1));
    }

    private static void AssertValidUtf16AndStrictUtf8(string text)
    {
        var strictUtf8 = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);
        byte[] bytes = strictUtf8.GetBytes(text);
        Assert.Equal(bytes.Length, Encoding.UTF8.GetByteCount(text));

        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (char.IsHighSurrogate(c))
            {
                Assert.True(i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]),
                    $"Unpaired high surrogate 0x{(int)c:X4} at char index {i}.");
                i++;
            }
            else
            {
                Assert.False(char.IsLowSurrogate(c),
                    $"Unpaired low surrogate 0x{(int)c:X4} at char index {i}.");
            }
        }
    }

    [Theory]
    [InlineData(45, false)]
    [InlineData(45, true)]
    [InlineData(120, false)]
    [InlineData(120, true)]
    [InlineData(250, false)]
    [InlineData(250, true)]
    public void BreadcrumbTreeFolder_And_MemExEngine_NeverSplitUtf16SurrogatePairs(int maxTokens, bool foldDeepSubtrees)
    {
        var estimator = new HeuristicTokenEstimator();

        // Construct strings whose UTF-16 surrogate pairs straddle exact cut indices:
        // - Breadcrumbs (> 2 items -> "{b0} > ... > {bLast}"): 25 + 9 + 26 = 60 chars before emoji => 0xD83D at [60], 0xDE80 at [61]
        // - Breadcrumbs (2 items -> "{b0} > {b1}"): 28 + 3 + 29 = 60 chars before emoji => 0xD83D at [60], 0xDE80 at [61]
        // - Entities (> 4 items -> "{e0}, {e1}, {e2}, ... (+N)"): 20 + 2 + 20 + 2 + 16 = 60 chars before emoji => 0xD83E at [60], 0xDD16 at [61]
        // - Entities (<= 4 items): 60 chars before emoji => 0xD83E at [60], 0xDD16 at [61]
        // - Content: 239 chars before emoji => 0xD83D at [239], 0xDE80 at [240]
        var traceMulti = new EpisodicTrace
        {
            Id = "surrogate-multi",
            Breadcrumbs = [new string('A', 25), "MidSegment", new string('B', 26) + "\U0001F680" + new string('C', 12)],
            Entities = [new string('E', 20), new string('F', 20), new string('G', 16) + "\U0001F916" + "tail", "Ent4", "Ent5"],
            Content = new string('X', 239) + "\U0001F680" + new string('Y', 120),
            BaseImportance = 5.0f,
            TimestampUs = 1_000_000L
        };

        var traceTwo = new EpisodicTrace
        {
            Id = "surrogate-two",
            Breadcrumbs = [new string('P', 28), new string('Q', 29) + "\U0001F680" + new string('R', 12)],
            Entities = [new string('S', 60) + "\U0001F916" + new string('T', 12)],
            Content = new string('M', 239) + "\U0001F680" + new string('N', 120),
            BaseImportance = 4.0f,
            TimestampUs = 1_000_000L
        };

        // Direct BreadcrumbTreeFolder.FormatTraceContext checks (single + multi)
        string directSingleMulti = BreadcrumbTreeFolder.FormatTraceContext([traceMulti], maxTokens, estimator, foldDeepSubtrees);
        AssertValidUtf16AndStrictUtf8(directSingleMulti);
        Assert.True(estimator.EstimateTokens(directSingleMulti) <= maxTokens);

        string directSingleTwo = BreadcrumbTreeFolder.FormatTraceContext([traceTwo], maxTokens, estimator, foldDeepSubtrees);
        AssertValidUtf16AndStrictUtf8(directSingleTwo);
        Assert.True(estimator.EstimateTokens(directSingleTwo) <= maxTokens);

        string directBoth = BreadcrumbTreeFolder.FormatTraceContext([traceMulti, traceTwo], maxTokens, estimator, foldDeepSubtrees);
        AssertValidUtf16AndStrictUtf8(directBoth);
        Assert.True(estimator.EstimateTokens(directBoth) <= maxTokens);

        // Unbounded FormatTraceContext with foldDeepSubtrees = true (exercises FormatTraceContextRange line 118)
        string directFolded = BreadcrumbTreeFolder.FormatTraceContext([traceMulti, traceTwo], foldDeepSubtrees: true);
        AssertValidUtf16AndStrictUtf8(directFolded);
        Assert.Contains("... [folded subtree]", directFolded);

        // End-to-end MemExEngine.Query checks
        using var engine = new MemExEngine(new MemExOptions
        {
            EnableNpuEmbeddings = false,
            AutoEmbedMissingVectors = false,
            TokenEstimator = estimator,
            FoldDeepSubtrees = foldDeepSubtrees
        });
        engine.StoreTrace(traceMulti);
        engine.StoreTrace(traceTwo);

        MemoryQueryResult queryRes = engine.Query("surrogate", ReadOnlySpan<float>.Empty, currentTimestampUs: 1_000_000L, maxTokens: maxTokens);
        AssertValidUtf16AndStrictUtf8(queryRes.FormattedContext);
        Assert.True(estimator.EstimateTokens(queryRes.FormattedContext) <= maxTokens);
    }

    [Theory]
    [InlineData(1e-14f)]
    [InlineData(1e-20f)]
    [InlineData(1e-35f)]
    [InlineData(float.Epsilon)]
    [InlineData(1e30f)]
    [InlineData(float.MaxValue)]
    public void TwoTierVectorFilter_ExtremeMagnitudeCosine_DoesNotUnderflowOrOverflow(float magnitude)
    {
        const int dim = 128;

        // Avoid float.MaxValue * float.MaxValue summation overflow ambiguity across many dims while still testing float.MaxValue component
        float[] query = new float[dim];
        float[] identical = new float[dim];
        float[] antiAligned = new float[dim];

        if (magnitude == float.MaxValue)
        {
            query[0] = float.MaxValue;
            query[1] = -float.MaxValue * 0.5f;
        }
        else
        {
            for (int i = 0; i < dim; i++)
            {
                query[i] = (i % 2 == 0 ? 1.0f : -0.5f) * magnitude;
            }
        }

        for (int i = 0; i < dim; i++)
        {
            identical[i] = query[i];
            antiAligned[i] = -query[i];
        }

        var traces = new List<EpisodicTrace>
        {
            new() { Id = "identical", Content = "Same", Embedding = identical },
            new() { Id = "anti", Content = "Opposite", Embedding = antiAligned },
            new() { Id = "zero", Content = "Zero", Embedding = new float[dim] }
        };

        // Add enough orthogonal distractors so validCount > coarseLimit (exercising FWHT + Hamming + top-K rerank)
        for (int i = 0; i < 16; i++)
        {
            float[] dist = new float[dim];
            dist[(i * 7 + 3) % dim] = magnitude;
            traces.Add(new EpisodicTrace { Id = $"dist-{i}", Content = "Dist", Embedding = dist });
        }

        List<(EpisodicTrace Trace, float CosineSimilarity)> ranked = TwoTierVectorFilter.FilterAndRerank(
            query,
            traces,
            coarseLimit: 8,
            finalLimit: 19);

        var identMatch = ranked.Single(r => r.Trace.Id == "identical");
        Assert.Equal(1.0f, identMatch.CosineSimilarity, precision: 4);

        var zeroMatch = ranked.FirstOrDefault(r => r.Trace.Id == "zero");
        if (zeroMatch.Trace != null)
        {
            Assert.Equal(0.0f, zeroMatch.CosineSimilarity);
        }

        // Verify ScoreAllTraces directly (both inside coarseLimit and for zero-norm vector)
        Span<float> scores = new float[traces.Count];
        TwoTierVectorFilter.ScoreAllTraces(query, traces, coarseLimit: 8, scores);
        Assert.Equal(1.0f, scores[0], precision: 4);
        Assert.True(scores[1] <= -0.99f || float.IsFinite(scores[1]));
        Assert.True(float.IsFinite(scores[2]));

        // Explicit zero-norm query must always yield 0.0f across FilterAndRerank and ScoreAllTraces
        float[] zeroQuery = new float[dim];
        List<(EpisodicTrace Trace, float CosineSimilarity)> zeroRanked = TwoTierVectorFilter.FilterAndRerank(
            zeroQuery,
            [traces[0], traces[2]],
            coarseLimit: 8,
            finalLimit: 2);
        Assert.All(zeroRanked, item => Assert.Equal(0.0f, item.CosineSimilarity));

        Span<float> zeroScores = new float[2];
        TwoTierVectorFilter.ScoreAllTraces(zeroQuery, [traces[0], traces[2]], coarseLimit: 8, zeroScores);
        Assert.Equal(0.0f, zeroScores[0]);
        Assert.Equal(0.0f, zeroScores[1]);
    }

    [Fact]
    public void TwoTierVectorFilter_ZeroNormQuery_ShortCircuitsAboveCoarseLimit_AndAvoidsEngineRankingInversion()
    {
        const int dim = 128;
        var rng = new Random(42);
        var traces20 = new List<EpisodicTrace>(20);

        for (int i = 0; i < 20; i++)
        {
            float[] emb = new float[dim];
            for (int d = 0; d < dim; d++)
            {
                emb[d] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }

            traces20.Add(new EpisodicTrace
            {
                Id = $"zd-{i:D2}",
                Content = $"Trace {i}",
                Embedding = emb,
                TokenCount = 10,
                BaseImportance = 1.0f + (i % 4),
                TimestampUs = 1_000_000L + i
            });
        }

        // Include invalid/unembedded/mismatched/non-finite traces alongside the 20 valid traces
        var mixedTraces = new List<EpisodicTrace>(traces20)
        {
            new() { Id = "unembedded", Content = "No vec", Embedding = null },
            new() { Id = "mismatch", Content = "Wrong dim", Embedding = new float[64] },
            new() { Id = "nan-vec", Content = "NaN", Embedding = Enumerable.Repeat(float.NaN, dim).ToArray() }
        };

        float[] zeroQuery128 = new float[dim];

        // 1. ScoreAllTraces(new float[128], ..., coarseLimit: 8):
        //    All 20 valid traces must receive exact 0.0f (even though validCount=20 > coarseLimit=8)
        //    and all invalid/unembedded traces must receive float.NaN.
        float[] scores = new float[mixedTraces.Count];
        TwoTierVectorFilter.ScoreAllTraces(zeroQuery128, mixedTraces, coarseLimit: 8, scores);
        for (int i = 0; i < 20; i++)
        {
            Assert.Equal(0.0f, scores[i]);
        }
        Assert.True(float.IsNaN(scores[20]));
        Assert.True(float.IsNaN(scores[21]));
        Assert.True(float.IsNaN(scores[22]));

        // Edge cases: coarseLimit == 0 (dim > 64) and dim <= 64 with validCount > coarseLimit
        float[] scoresCoarseZero = new float[mixedTraces.Count];
        TwoTierVectorFilter.ScoreAllTraces(zeroQuery128, mixedTraces, coarseLimit: 0, scoresCoarseZero);
        for (int i = 0; i < 20; i++)
        {
            Assert.Equal(0.0f, scoresCoarseZero[i]);
        }
        Assert.True(float.IsNaN(scoresCoarseZero[20]));
        Assert.True(float.IsNaN(scoresCoarseZero[21]));
        Assert.True(float.IsNaN(scoresCoarseZero[22]));

        var traces32 = Enumerable.Range(0, 20)
            .Select(i => new EpisodicTrace
            {
                Id = $"d32-{i}",
                Content = "Small dim",
                Embedding = Enumerable.Range(0, 32).Select(d => (float)Math.Sin(i + d)).ToArray()
            })
            .ToList();
        float[] scores32 = new float[traces32.Count];
        TwoTierVectorFilter.ScoreAllTraces(new float[32], traces32, coarseLimit: 8, scores32);
        Assert.All(scores32, s => Assert.Equal(0.0f, s));

        // 2. FilterAndRerank(new float[128], ..., coarseLimit: 8, finalLimit: 10):
        //    Must return 10 valid traces with exact 0.0f cosine similarity and never invoke FWHT against all-1 bits.
        List<(EpisodicTrace Trace, float CosineSimilarity)> reranked = TwoTierVectorFilter.FilterAndRerank(
            zeroQuery128,
            mixedTraces,
            coarseLimit: 8,
            finalLimit: 10);
        Assert.Equal(10, reranked.Count);
        Assert.All(reranked, r =>
        {
            Assert.Equal(0.0f, r.CosineSimilarity);
            Assert.StartsWith("zd-", r.Trace.Id, StringComparison.Ordinal);
        });

        // 3. MemExEngine.Query("z", new float[128], ...) across 20 embedded traces with CoarseCandidateLimit = 8,
        //    WeightVector = 1.0f, and zero other weights must produce TotalUtility == Math.Exp(0.5) (no ranking inversion).
        using var engine = new MemExEngine(new MemExOptions
        {
            EnableNpuEmbeddings = false,
            AutoEmbedMissingVectors = false,
            CoarseCandidateLimit = 8,
            MaxPromptTokens = 10,
            WeightDecay = 0.0f,
            WeightVector = 1.0f,
            WeightGraph = 0.0f,
            WeightImportance = 0.0f
        });
        foreach (EpisodicTrace trace in traces20)
        {
            engine.StoreTrace(trace);
        }

        MemoryQueryResult queryResult = engine.Query("z", zeroQuery128, currentTimestampUs: 2_000_000L);
        Assert.Single(queryResult.SelectedTraces);
        Assert.Equal(Math.Exp(0.5), queryResult.TotalUtility, precision: 12);
    }
}
