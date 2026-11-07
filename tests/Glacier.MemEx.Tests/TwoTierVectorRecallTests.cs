namespace Glacier.MemEx.Tests;

using System;
using System.Collections.Generic;
using Glacier.MemEx.Models;
using Glacier.MemEx.Recall;
using Xunit;

public class TwoTierVectorRecallTests
{
    [Fact]
    public void TwoTierVectorFilter_ReturnsMostSimilarVector_Accurately()
    {
        int dim = 64;
        float[] query = new float[dim];
        for (int i = 0; i < dim; i++) query[i] = (i % 2 == 0) ? 1.0f : -1.0f;

        // Trace 1: identical direction
        float[] emb1 = (float[])query.Clone();

        // Trace 2: orthogonal direction
        float[] emb2 = new float[dim];
        for (int i = 0; i < dim; i++) emb2[i] = (i % 4 == 0) ? 1.0f : 0.0f;

        // Trace 3: opposite direction
        float[] emb3 = new float[dim];
        for (int i = 0; i < dim; i++) emb3[i] = -query[i];

        var traces = new List<EpisodicTrace>
        {
            new EpisodicTrace { Content = "Target Trace", Embedding = emb1 },
            new EpisodicTrace { Content = "Orthogonal Trace", Embedding = emb2 },
            new EpisodicTrace { Content = "Opposite Trace", Embedding = emb3 }
        };

        var results = TwoTierVectorFilter.FilterAndRerank(query, traces, coarseLimit: 10, finalLimit: 3);

        Assert.Equal(3, results.Count);
        Assert.Equal("Target Trace", results[0].Trace.Content);
        Assert.True(results[0].CosineSimilarity > 0.99f);

        Assert.Equal("Opposite Trace", results[2].Trace.Content);
        Assert.True(results[2].CosineSimilarity < -0.99f);
    }
}
