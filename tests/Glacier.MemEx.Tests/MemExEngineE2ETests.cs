namespace Glacier.MemEx.Tests;

using System;
using System.Collections.Generic;
using Glacier.MemEx;
using Glacier.MemEx.Models;
using Xunit;

public class MemExEngineE2ETests
{
    [Fact]
    public void MemExEngine_StoresTraces_AndRetrievesContextWithBreadcrumbs()
    {
        var engine = new MemExEngine(new MemExOptions
        {
            MaxPromptTokens = 500,
            WeightDecay = 0.3f,
            WeightVector = 0.4f,
            WeightGraph = 0.3f
        });

        long nowUs = 2_000_000_000_000L;
        int dim = 32;

        float[] embGlacier = new float[dim];
        for (int i = 0; i < dim; i++) embGlacier[i] = (i % 2 == 0) ? 1.0f : 0.0f;

        float[] embRandom = new float[dim];
        for (int i = 0; i < dim; i++) embRandom[i] = (i % 3 == 0) ? 1.0f : -0.5f;

        engine.StoreTrace(new EpisodicTrace
        {
            Content = "Glacier database buffer pool handles page eviction via Clock-Pro.",
            Breadcrumbs = ["Architecture", "Storage", "BufferPool"],
            Entities = ["BufferPool", "ClockPro", "Storage"],
            Embedding = embGlacier,
            TokenCount = 40,
            TimestampUs = nowUs - 5_000_000L // 5 seconds ago
        });

        engine.StoreTrace(new EpisodicTrace
        {
            Content = "Unrelated cooking recipe for apple pie.",
            Breadcrumbs = ["Culinary", "Recipes", "Dessert"],
            Entities = ["ApplePie", "Baking"],
            Embedding = embRandom,
            TokenCount = 30,
            TimestampUs = nowUs - 500_000_000L // 500 seconds ago
        });

        Assert.Equal(2, engine.Count);

        // Query with query embedding matching embGlacier and query entity "ClockPro"
        var queryResult = engine.Query(
            queryText: "How does buffer pool evict pages?",
            queryEmbedding: embGlacier,
            queryEntities: ["ClockPro"],
            currentTimestampUs: nowUs,
            maxTokens: 200);

        Assert.NotNull(queryResult);
        Assert.NotEmpty(queryResult.SelectedTraces);
        Assert.True(queryResult.TotalTokens <= 200);

        // First retrieved trace must be the storage buffer pool trace
        var top = queryResult.SelectedTraces[0];
        Assert.Contains("buffer pool", top.Content, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("BufferPool", top.Entities);
        Assert.Equal(["Architecture", "Storage", "BufferPool"], top.Breadcrumbs);

        // Formatted context should contain structural breadcrumbs
        Assert.Contains("Architecture > Storage > BufferPool", queryResult.FormattedContext);
        Assert.Contains("Clock-Pro", queryResult.FormattedContext);
    }

    [Fact]
    public void SleepConsolidationBridge_ExportsTopCandidates_AndPrunesTraces()
    {
        var engine = new MemExEngine();
        long nowUs = 2_000_000_000L;

        var hot = new EpisodicTrace
        {
            Id = "hot-1",
            Content = "Vulkan Q3_K kernel fuses block dequantization with warp reduction.",
            BaseImportance = 4.5f,
            TimestampUs = nowUs - 5_000_000L
        };
        hot.RecordAccess(nowUs - 1_000_000L);

        var cold = new EpisodicTrace
        {
            Id = "cold-1",
            Content = "Unrelated stale note from a long time ago.",
            BaseImportance = 0.2f,
            TimestampUs = 1_000_000L // ~2000s ago
        };

        engine.StoreTrace(hot);
        engine.StoreTrace(cold);

        var candidates = engine.ExportConsolidationCandidates(maxCount: 10, minAccessCount: 1, currentTimestampUs: nowUs);
        Assert.Single(candidates);
        Assert.Equal("hot-1", candidates[0].Id);

        int prunedSpecific = engine.PruneTraces(candidates.Select(c => c.Id));
        Assert.Equal(1, prunedSpecific);
        Assert.Equal(1, engine.Count);

        int prunedDecayed = engine.PruneDecayedTraces(minRecencyScore: 0.5, currentTimestampUs: nowUs);
        Assert.Equal(1, prunedDecayed);
        Assert.Equal(0, engine.Count);
    }

    [Fact]
    public void MemExEngine_UnembeddedTrace_DoesNotReceivePhantomZeroVectorBoost()
    {
        using var engine = new MemExEngine(new MemExOptions
        {
            AutoEmbedMissingVectors = false,
            WeightDecay = 0f,
            WeightVector = 10.0f,
            WeightGraph = 0f,
            WeightImportance = 0f,
            MaxPromptTokens = 50
        });

        float[] q = [1f, 0f, 0f, 0f];
        float[] anti = [-1f, 0f, 0f, 0f]; // cosine = -1 -> normVec = 0.0

        engine.StoreTrace(new EpisodicTrace { Id = "unembedded", Content = "No vector here", TokenCount = 40, BaseImportance = 1f });
        engine.StoreTrace(new EpisodicTrace { Id = "embedded", Content = "Has vector", Embedding = [0.2f, 1f, 0f, 0f], TokenCount = 40, BaseImportance = 1f });

        // Query against anti-aligned vector would score anti at 0.0 and previously scored unembedded at 0.5!
        engine.StoreTrace(new EpisodicTrace { Id = "slightly-neg", Content = "Slightly negative", Embedding = [-0.2f, 1f, 0f, 0f], TokenCount = 40, BaseImportance = 1f });
        var res = engine.Query("test", q, maxTokens: 45);
        Assert.Single(res.SelectedTraces);
        Assert.Equal("embedded", res.SelectedTraces[0].Id);

        // Specifically verify unembedded does NOT beat a positive-cosine trace and ties with cosine=-1 (0.0 normVec):
        using var engine2 = new MemExEngine(new MemExOptions
        {
            AutoEmbedMissingVectors = false,
            WeightDecay = 0f,
            WeightVector = 10.0f,
            WeightGraph = 0f,
            WeightImportance = 0f,
            MaxPromptTokens = 50
        });
        engine2.StoreTrace(new EpisodicTrace { Id = "unembedded", Content = "No vector", TokenCount = 40, BaseImportance = 1f });
        engine2.StoreTrace(new EpisodicTrace { Id = "weak-pos", Content = "Weak match", Embedding = [-0.5f, 1f, 0f, 0f], TokenCount = 40, BaseImportance = 1f });
        // cosine([-0.5,1,0,0], [1,0,0,0]) = -0.447 -> normVec = 0.276 > 0.0 (unembedded)
        var res2 = engine2.Query("test", q, maxTokens: 45);
        Assert.Single(res2.SelectedTraces);
        Assert.Equal("weak-pos", res2.SelectedTraces[0].Id);
    }
}
