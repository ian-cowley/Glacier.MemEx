namespace Glacier.MemEx.Tests;

using System;
using System.Collections.Generic;
using System.Linq;
using Glacier.MemEx;
using Glacier.MemEx.Budgeting;
using Glacier.MemEx.Decay;
using Glacier.MemEx.Models;
using Glacier.MemEx.Recall;
using Xunit;

public class AdversarialMemExStressTests
{
    // =========================================================================
    // 1. POWER-LAW DECAY BOUNDARY CONDITIONS & NUMERICAL STABILITY
    // =========================================================================

    [Fact]
    public void PowerLawDecay_DeltaZero_ComputesExactFiniteActivation()
    {
        long t0 = 1_000_000_000L;
        var trace = new EpisodicTrace
        {
            Content = "Delta t = 0 test",
            TimestampUs = t0
        };

        // At delta t = 0, deltaSec = 0.
        // sum = (0 + tau)^(-d) = tau^(-d).
        // With tau = 1.0, sum = 1.0, ln(sum) = 0.0.
        double act = PowerLawDecayCalculator.ComputeBaseLevelActivation(trace, t0, d: 0.5, tauSeconds: 1.0);
        Assert.Equal(0.0, act, precision: 6);

        double recency = PowerLawDecayCalculator.ComputeRecencyScore(trace, t0, d: 0.5, tauSeconds: 1.0);
        // Sigmoid(0) = 0.5
        Assert.Equal(0.5, recency, precision: 6);
    }

    [Fact]
    public void PowerLawDecay_MassiveTimeElapsed_ClampsGracefullyWithoutNaNOrUnderflow()
    {
        long t0 = 1_000_000L;
        var trace = new EpisodicTrace
        {
            Content = "Ancient trace",
            TimestampUs = t0
        };

        // 10^15 microseconds is ~31.7 million years
        long hugeTimestamp = t0 + 1_000_000_000_000_000L;

        double act = PowerLawDecayCalculator.ComputeBaseLevelActivation(trace, hugeTimestamp, d: 0.5, tauSeconds: 1.0);
        Assert.False(double.IsNaN(act));
        Assert.False(double.IsInfinity(act));
        Assert.True(act < -10.0, $"Expected very negative activation, got {act}");

        double recency = PowerLawDecayCalculator.ComputeRecencyScore(trace, hugeTimestamp, d: 0.5, tauSeconds: 1.0);
        Assert.False(double.IsNaN(recency));
        Assert.True(recency >= 0.0 && recency < 0.001);
    }

    [Fact]
    public void PowerLawDecay_FutureTimestamp_ClampsDeltaToZeroWithoutThrowing()
    {
        long t0 = 5_000_000_000L;
        var trace = new EpisodicTrace
        {
            Content = "Trace created in future relative to query",
            TimestampUs = t0
        };

        // Query timestamp is in the past relative to trace creation
        long pastQueryTime = t0 - 1_000_000_000L;

        double act = PowerLawDecayCalculator.ComputeBaseLevelActivation(trace, pastQueryTime, d: 0.5, tauSeconds: 1.0);
        Assert.False(double.IsNaN(act));
        // Math.Max(0.0, ...) clamps negative delta to 0, matching delta = 0
        Assert.Equal(0.0, act, precision: 6);
    }

    [Theory]
    [InlineData(0.001, 1.0)]
    [InlineData(0.1, 0.001)]
    [InlineData(0.5, 1.0)]
    [InlineData(2.0, 100.0)]
    [InlineData(10.0, 2.0)]
    public void PowerLawDecay_ParameterSweeps_RemainFiniteAndBounded(double d, double tau)
    {
        long t0 = 100_000_000L;
        var trace = new EpisodicTrace { Content = "Sweep trace", TimestampUs = t0 };
        for (int i = 1; i <= 50; i++)
        {
            trace.RecordAccess(t0 + i * 1_000_000L);
        }

        long now = t0 + 60_000_000L;
        double act = PowerLawDecayCalculator.ComputeBaseLevelActivation(trace, now, d, tau);
        double recency = PowerLawDecayCalculator.ComputeRecencyScore(trace, now, d, tau);

        Assert.False(double.IsNaN(act), $"Activation was NaN for d={d}, tau={tau}");
        Assert.False(double.IsNaN(recency), $"Recency was NaN for d={d}, tau={tau}");
        Assert.True(recency >= 0.0 && recency <= 1.0, $"Recency {recency} not in [0, 1]");
    }

    [Fact]
    public void PowerLawDecay_InvalidParameters_ThrowsExpectedExceptions()
    {
        var trace = new EpisodicTrace { Content = "Test", TimestampUs = 1000L };

        Assert.Throws<ArgumentNullException>(() => PowerLawDecayCalculator.ComputeBaseLevelActivation(null!, 1000L));
        Assert.Throws<ArgumentOutOfRangeException>(() => PowerLawDecayCalculator.ComputeBaseLevelActivation(trace, 1000L, d: 0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => PowerLawDecayCalculator.ComputeBaseLevelActivation(trace, 1000L, d: -0.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => PowerLawDecayCalculator.ComputeBaseLevelActivation(trace, 1000L, d: 0.5, tauSeconds: 0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => PowerLawDecayCalculator.ComputeBaseLevelActivation(trace, 1000L, d: 0.5, tauSeconds: -1.0));
    }

    // =========================================================================
    // 2. KNAPSACK PROMPT PACKER BOUNDARY CONDITIONS
    // =========================================================================

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(-1000)]
    public void KnapsackPromptPacker_ZeroOrNegativeBudget_ReturnsEmpty(int maxTokens)
    {
        var traces = new List<(EpisodicTrace Trace, double Utility)>
        {
            (new EpisodicTrace { Content = "Trace 1", TokenCount = 10 }, 50.0),
            (new EpisodicTrace { Content = "Trace 2", TokenCount = 20 }, 80.0)
        };

        var packed = KnapsackPromptPacker.PackTraces(traces, maxTokens);
        Assert.Empty(packed);
    }

    [Fact]
    public void KnapsackPromptPacker_CandidateExceedsBudget_FilteredOut()
    {
        var traces = new List<(EpisodicTrace Trace, double Utility)>
        {
            (new EpisodicTrace { Content = "Massive Trace", TokenCount = 500 }, 1000.0)
        };

        var packed = KnapsackPromptPacker.PackTraces(traces, maxTokens: 100);
        Assert.Empty(packed);
    }

    [Fact]
    public void KnapsackPromptPacker_ZeroOrNegativeUtilityCandidates_FilteredOut()
    {
        var traces = new List<(EpisodicTrace Trace, double Utility)>
        {
            (new EpisodicTrace { Content = "Zero utility", TokenCount = 10 }, 0.0),
            (new EpisodicTrace { Content = "Negative utility", TokenCount = 10 }, -15.0),
            (new EpisodicTrace { Content = "Valid utility", TokenCount = 10 }, 25.0)
        };

        var packed = KnapsackPromptPacker.PackTraces(traces, maxTokens: 100);
        Assert.Single(packed);
        Assert.Equal("Valid utility", packed[0].Content);
    }

    // =========================================================================
    // 3. HIPPORAG SPREADING ACTIVATION (DISCONNECTED / SPARSE GRAPHS)
    // =========================================================================

    [Fact]
    public void HippoRag_EmptyTracesOrEmptyQueries_ReturnsEmptyDictionary()
    {
        var traces = new List<EpisodicTrace>
        {
            new EpisodicTrace { Content = "T1", Entities = ["EntityA"] }
        };

        Assert.Empty(HippoRagSpreadingActivation.ComputePersonalizedPageRank([], traces));
        Assert.Empty(HippoRagSpreadingActivation.ComputePersonalizedPageRank(["EntityA"], []));
    }

    [Fact]
    public void HippoRag_IsolatedSingleEntityNodes_HandledAsDanglingWithoutCrash()
    {
        // Traces where each trace has only 1 entity -> graph has nodes with 0 edges (dangling)
        var traces = new List<EpisodicTrace>
        {
            new EpisodicTrace { Content = "T1", Entities = ["NodeA"] },
            new EpisodicTrace { Content = "T2", Entities = ["NodeB"] },
            new EpisodicTrace { Content = "T3", Entities = ["NodeC"] }
        };

        var ppr = HippoRagSpreadingActivation.ComputePersonalizedPageRank(["NodeA"], traces, dampingAlpha: 0.85);

        Assert.Equal(3, ppr.Count);
        double sum = ppr.Values.Sum();
        Assert.True(Math.Abs(sum - 1.0) < 1e-6, $"Total probability sum was {sum}, expected 1.0");

        // Seed NodeA should retain the most activation
        Assert.True(ppr["NodeA"] > ppr["NodeB"]);
        Assert.True(ppr["NodeA"] > ppr["NodeC"]);
    }

    [Fact]
    public void HippoRag_MultipleDisconnectedComponents_ActivationStrictlyConfined()
    {
        // Component 1: Alpha <-> Beta
        // Component 2: Gamma <-> Delta
        var traces = new List<EpisodicTrace>
        {
            new EpisodicTrace { Content = "C1", Entities = ["Alpha", "Beta"] },
            new EpisodicTrace { Content = "C2", Entities = ["Gamma", "Delta"] }
        };

        var ppr = HippoRagSpreadingActivation.ComputePersonalizedPageRank(["Alpha"], traces, dampingAlpha: 0.85);

        Assert.Equal(4, ppr.Count);
        // Alpha and Beta have non-zero scores
        Assert.True(ppr["Alpha"] > 0.0);
        Assert.True(ppr["Beta"] > 0.0);

        // Gamma and Delta receive strictly 0.0
        Assert.Equal(0.0, ppr["Gamma"]);
        Assert.Equal(0.0, ppr["Delta"]);

        double sum = ppr.Values.Sum();
        Assert.True(Math.Abs(sum - 1.0) < 1e-6);
    }

    [Fact]
    public void HippoRag_QueryEntitiesNotInGraph_FallsBackToUniformTeleport()
    {
        var traces = new List<EpisodicTrace>
        {
            new EpisodicTrace { Content = "C1", Entities = ["A", "B"] }
        };

        // Query with entity "UnknownEntity"
        var ppr = HippoRagSpreadingActivation.ComputePersonalizedPageRank(["UnknownEntity"], traces);

        Assert.Equal(2, ppr.Count);
        Assert.True(ppr.ContainsKey("A"));
        Assert.True(ppr.ContainsKey("B"));

        double sum = ppr.Values.Sum();
        Assert.True(Math.Abs(sum - 1.0) < 1e-6);
        Assert.Equal(ppr["A"], ppr["B"], precision: 5);
    }

    // =========================================================================
    // 4. MEMEX ENGINE E2E BOUNDARY CONDITIONS
    // =========================================================================

    [Fact]
    public void MemExEngine_ZeroMaxTokens_ReturnsEmptySelectedTraces()
    {
        var engine = new MemExEngine();
        engine.StoreTrace(new EpisodicTrace
        {
            Content = "Sample context",
            TokenCount = 50,
            TimestampUs = 1000L
        });

        var res = engine.Query("query", ReadOnlySpan<float>.Empty, maxTokens: 0);

        Assert.NotNull(res);
        Assert.Empty(res.SelectedTraces);
        Assert.Equal(0, res.TotalTokens);
    }

    [Fact]
    public void MemExEngine_TracesWithoutEmbeddings_HandledCleanlyInQuery()
    {
        var engine = new MemExEngine();
        engine.StoreTrace(new EpisodicTrace
        {
            Content = "Pure text trace with no embedding vector",
            Entities = ["DocTree"],
            TokenCount = 25,
            TimestampUs = 1000L
        });

        float[] queryEmb = [1.0f, 0.0f, -1.0f, 0.5f];
        var res = engine.Query("query", queryEmb, queryEntities: ["DocTree"], maxTokens: 100);

        Assert.NotNull(res);
        Assert.Single(res.SelectedTraces);
        Assert.Contains("DocTree", res.FormattedContext);
    }

    [Fact]
    public void MemExEngine_ConcurrentQueries_OnSharedTraces()
    {
        var engine = new MemExEngine();
        for (int i = 0; i < 50; i++)
        {
            engine.StoreTrace(new EpisodicTrace
            {
                Content = $"Trace {i} content for concurrent testing",
                Entities = [$"Entity_{i % 5}", "SharedEntity"],
                TokenCount = 10,
                TimestampUs = 1000L + i * 10_000L
            });
        }

        Parallel.For(0, 500, iter =>
        {
            var res = engine.Query("test query", ReadOnlySpan<float>.Empty, queryEntities: ["SharedEntity"], maxTokens: 50);
            Assert.NotNull(res);
        });
    }

    [Fact]
    public void HippoRag_SelfLoopsAndDuplicateEntities_ConvergesWithoutNaN()
    {
        var traces = new List<EpisodicTrace>
        {
            new EpisodicTrace { Content = "Duplicate entities", Entities = ["Apple", "Apple", "Banana"] },
            new EpisodicTrace { Content = "Single entity", Entities = ["Banana", "Cherry"] }
        };

        var ppr = HippoRagSpreadingActivation.ComputePersonalizedPageRank(["Apple"], traces);
        Assert.Equal(3, ppr.Count);
        double sum = ppr.Values.Sum();
        Assert.True(Math.Abs(sum - 1.0) < 1e-6);
        Assert.True(ppr["Apple"] > 0.0);
        Assert.False(double.IsNaN(ppr["Apple"]));
    }

    [Fact]
    public void KnapsackPromptPacker_LargeScaleStress_NeverExceedsTokenBudget()
    {
        var candidates = new List<(EpisodicTrace Trace, double Utility)>();
        var rng = new Random(42);
        for (int i = 0; i < 5_000; i++)
        {
            int tokens = rng.Next(1, 100);
            double utility = rng.NextDouble() * 1000.0;
            candidates.Add((new EpisodicTrace { Content = $"Trace {i}", TokenCount = tokens }, utility));
        }

        int maxBudget = 2_500;
        var packed = KnapsackPromptPacker.PackTraces(candidates, maxBudget);

        Assert.NotEmpty(packed);
        int totalTokens = packed.Sum(t => t.TokenCount);
        Assert.True(totalTokens <= maxBudget, $"Total tokens {totalTokens} exceeded budget {maxBudget}");
    }

    [Fact]
    public void BreadcrumbTreeFolder_DeeplyNestedAndMassiveSubtrees_FoldsCorrectly()
    {
        var longBreadcrumbs = Enumerable.Range(0, 50).Select(i => $"Level_{i}").ToList();
        string massiveContent = new string('A', 5_000);

        var trace = new EpisodicTrace
        {
            Content = massiveContent,
            Breadcrumbs = longBreadcrumbs,
            Entities = ["EntityAlpha", "EntityBeta"]
        };

        // Folded
        string folded = BreadcrumbTreeFolder.FormatTraceContext([trace], foldDeepSubtrees: true);
        Assert.Contains("... [folded subtree]", folded);
        Assert.True(folded.Length < 1_000);
        Assert.Contains("Level_0 > Level_1", folded);
        Assert.Contains("EntityAlpha, EntityBeta", folded);

        // Unfolded
        string unfolded = BreadcrumbTreeFolder.FormatTraceContext([trace], foldDeepSubtrees: false);
        Assert.DoesNotContain("... [folded subtree]", unfolded);
        Assert.Contains(massiveContent, unfolded);
    }

    [Fact]
    public void PowerLawDecay_MonotonicityOverTime_ActivationDecreasesStrictly()
    {
        long t0 = 1_000_000L;
        var trace = new EpisodicTrace { Content = "Monotonic trace", TimestampUs = t0 };
        trace.RecordAccess(t0 + 10_000_000L);

        double lastRecency = double.MaxValue;
        for (int step = 1; step <= 20; step++)
        {
            long t = t0 + 10_000_000L + (step * 5_000_000L);
            double recency = PowerLawDecayCalculator.ComputeRecencyScore(trace, t);

            Assert.True(recency <= lastRecency, $"Recency increased at step {step}: {recency} > {lastRecency}");
            Assert.True(recency >= 0.0 && recency <= 1.0);
            lastRecency = recency;
        }
    }
}


