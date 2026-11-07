namespace Glacier.MemEx;

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Glacier.MemEx.Budgeting;
using Glacier.MemEx.Decay;
using Glacier.MemEx.Models;
using Glacier.MemEx.Recall;

public sealed class MemExEngine
{
    private readonly ConcurrentDictionary<string, EpisodicTrace> _traces = new(StringComparer.Ordinal);
    private readonly MemExOptions _options;

    public MemExEngine(MemExOptions? options = null)
    {
        _options = options ?? new MemExOptions();
    }

    public int Count => _traces.Count;

    public void StoreTrace(EpisodicTrace trace)
    {
        ArgumentNullException.ThrowIfNull(trace);

        if (trace.Embedding != null && trace.BinaryEmbedding == null)
        {
            trace.BinaryEmbedding = TwoTierVectorFilter.QuantizeTo1Bit(trace.Embedding);
        }

        if (trace.TokenCount <= 0 && !string.IsNullOrWhiteSpace(trace.Content))
        {
            // Estimate tokens: ~4 chars per token
            typeof(EpisodicTrace).GetProperty(nameof(trace.TokenCount))?
                .SetValue(trace, Math.Max(1, trace.Content.Length / 4));
        }

        _traces[trace.Id] = trace;
    }

    public MemoryQueryResult Query(
        string queryText,
        ReadOnlySpan<float> queryEmbedding,
        IReadOnlyList<string>? queryEntities = null,
        long? currentTimestampUs = null,
        int? maxTokens = null)
    {
        long nowUs = currentTimestampUs ?? DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1000L;
        int budget = maxTokens ?? _options.MaxPromptTokens;
        var traceList = _traces.Values.ToList();

        if (traceList.Count == 0)
        {
            return new MemoryQueryResult();
        }

        // 1. Two-tier Vector Recall (1-bit coarse filter + FP32 cosine re-ranking)
        var vectorMatches = new Dictionary<string, float>(StringComparer.Ordinal);
        if (queryEmbedding.Length > 0)
        {
            var reranked = TwoTierVectorFilter.FilterAndRerank(
                queryEmbedding,
                traceList,
                coarseLimit: _options.CoarseCandidateLimit,
                finalLimit: traceList.Count);

            foreach (var (t, cos) in reranked)
            {
                vectorMatches[t.Id] = cos;
            }
        }

        // 2. HippoRAG Spreading Activation
        var pprScores = (queryEntities != null && queryEntities.Count > 0)
            ? HippoRagSpreadingActivation.ComputePersonalizedPageRank(queryEntities, traceList)
            : [];

        // 3. Score all candidates using composite activation
        var scoredCandidates = new List<(EpisodicTrace Trace, double Utility)>(traceList.Count);

        foreach (var trace in traceList)
        {
            // A) Power-law temporal decay recency score
            double recency = PowerLawDecayCalculator.ComputeRecencyScore(
                trace,
                nowUs,
                _options.DecayParameterD,
                _options.TimeScaleTauSeconds);

            // B) Vector similarity score
            vectorMatches.TryGetValue(trace.Id, out float vecSim);
            double normVec = (vecSim + 1.0) * 0.5; // Map [-1, 1] to [0, 1]

            // C) HippoRAG graph score
            double graphScore = HippoRagSpreadingActivation.ScoreTrace(trace, pprScores);

            // D) Base importance
            double importance = Math.Clamp(trace.BaseImportance, 0.1f, 5.0f);

            // Composite Activation: A_i(t)
            double activation =
                (_options.WeightDecay * recency) +
                (_options.WeightVector * normVec) +
                (_options.WeightGraph * graphScore) +
                (_options.WeightImportance * (importance / 5.0));

            // Utility u_i = exp(A_i(t))
            double utility = Math.Exp(activation);
            scoredCandidates.Add((trace, utility));
        }

        // 4. 0/1 Knapsack Prompt Budget Optimization
        var selectedTraces = KnapsackPromptPacker.PackTraces(scoredCandidates, budget);

        // Record retrieval access events
        foreach (var t in selectedTraces)
        {
            t.RecordAccess(nowUs);
        }

        int totalTokens = selectedTraces.Sum(t => t.TokenCount);
        double totalUtility = scoredCandidates
            .Where(c => selectedTraces.Contains(c.Trace))
            .Sum(c => c.Utility);

        string formattedContext = BreadcrumbTreeFolder.FormatTraceContext(selectedTraces);

        return new MemoryQueryResult
        {
            SelectedTraces = selectedTraces,
            TotalTokens = totalTokens,
            TotalUtility = totalUtility,
            FormattedContext = formattedContext
        };
    }
}
