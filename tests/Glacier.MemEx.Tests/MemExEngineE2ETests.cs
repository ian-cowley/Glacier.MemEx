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
}
