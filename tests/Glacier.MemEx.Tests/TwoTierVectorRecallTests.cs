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

    [Fact]
    public void TwoTierVectorFilter_AndMemExEngine_RejectNonFiniteQuery_AndFilterNonFiniteCandidates()
    {
        float[] validEmb = new float[64];
        validEmb[0] = 1.0f;

        float[] nanQuery = new float[64];
        nanQuery[0] = float.NaN;
        float[] infQuery = new float[64];
        infQuery[3] = float.PositiveInfinity;

        var validTrace = new EpisodicTrace { Id = "valid", Content = "Valid trace", Embedding = validEmb, TokenCount = 5 };
        var nanCand = new float[64];
        nanCand[0] = float.NaN;
        var infCand = new float[64];
        infCand[0] = float.NegativeInfinity;
        var nanTrace = new EpisodicTrace { Id = "nan-cand", Content = "NaN trace", Embedding = nanCand, TokenCount = 5 };
        var infTrace = new EpisodicTrace { Id = "inf-cand", Content = "Inf trace", Embedding = infCand, TokenCount = 5 };

        var traces = new List<EpisodicTrace> { nanTrace, infTrace, validTrace };

        Assert.Throws<ArgumentException>(() => TwoTierVectorFilter.FilterAndRerank(nanQuery, traces));
        Assert.Throws<ArgumentException>(() => TwoTierVectorFilter.FilterAndRerank(infQuery, traces));

        var filtered = TwoTierVectorFilter.FilterAndRerank(validEmb, traces, coarseLimit: 16, finalLimit: 16);
        Assert.Single(filtered);
        Assert.Equal("valid", filtered[0].Trace.Id);
        Assert.True(float.IsFinite(filtered[0].CosineSimilarity));

        var engine = new MemExEngine();
        engine.StoreTrace(nanTrace);
        engine.StoreTrace(infTrace);
        engine.StoreTrace(validTrace);

        Assert.Throws<ArgumentException>(() => engine.Query("q", nanQuery));
        Assert.Throws<ArgumentException>(() => engine.Query("q", infQuery));

        var queryRes = engine.Query("q", validEmb, maxTokens: 100);
        Assert.Contains(queryRes.SelectedTraces, t => t.Id == "valid");
        Assert.True(double.IsFinite(queryRes.TotalUtility));
    }

    [Theory]
    [InlineData(64)]
    [InlineData(128)]
    public void TwoTierVectorFilter_MismatchedAndEmptyDimensions_DoNotPoisonCoarseLimitSlots(int validDim)
    {
        float[] query = new float[validDim];
        for (int i = 0; i < validDim; i++) query[i] = (i % 2 == 0) ? 1.0f : -0.5f;

        var traces = new List<EpisodicTrace>();
        for (int i = 0; i < 20; i++)
        {
            traces.Add(new EpisodicTrace { Id = $"empty-{i}", Content = $"Empty {i}", Embedding = [] });
            traces.Add(new EpisodicTrace { Id = $"short-{i}", Content = $"Short {i}", Embedding = [1.0f, 0.5f] });
        }
        traces.Add(new EpisodicTrace { Id = "valid-match", Content = "Valid Match", Embedding = (float[])query.Clone() });

        var results = TwoTierVectorFilter.FilterAndRerank(query, traces, coarseLimit: 16, finalLimit: 5);
        Assert.Single(results);
        Assert.Equal("valid-match", results[0].Trace.Id);
        Assert.True(results[0].CosineSimilarity > 0.99f);
    }

    [Fact]
    public void TwoTierVectorFilter_Dim32_HighCosineMatchNotStarvedByDistractors()
    {
        const int dim = 32;
        float[] query = new float[dim];
        query[0] = 1.0f;
        for (int i = 1; i < dim; i++) query[i] = 0.05f;

        // Construct a trueBest vector with cos ~= 0.9867 whose positive tail would distort 64-element mean binarization
        float[] trueBest = new float[dim];
        trueBest[0] = 1.0f;
        for (int i = 1; i < dim; i++) trueBest[i] = 0.08f;

        var traces = new List<EpisodicTrace>();
        for (int i = 0; i < 32; i++)
        {
            float[] distractor = new float[dim];
            distractor[0] = 0.01f;
            distractor[1 + (i % (dim - 1))] = 1.0f;
            traces.Add(new EpisodicTrace { Id = $"distractor-{i}", Content = $"Distractor {i}", Embedding = distractor });
        }
        traces.Add(new EpisodicTrace { Id = "true-best", Content = "True Best", Embedding = trueBest });

        var results = TwoTierVectorFilter.FilterAndRerank(query, traces, coarseLimit: 16, finalLimit: 4);
        Assert.Equal(4, results.Count);
        Assert.Equal("true-best", results[0].Trace.Id);
        Assert.True(results[0].CosineSimilarity > 0.97f, $"Expected > 0.97, got {results[0].CosineSimilarity}");
    }
}
