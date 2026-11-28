namespace Glacier.MemEx.Recall;

using System;
using System.Buffers;
using System.Collections.Generic;
using Glacier.Graph.Kernels;
using Glacier.MemEx.Models;

public static class HippoRagSpreadingActivation
{
    private const float ConvergenceTolerance = 1e-7f;

    /// <summary>
    /// Executes Personalized PageRank (PPR) spreading activation from query entity seeds
    /// across an associative co-occurrence graph of entities and traces.
    /// </summary>
    public static Dictionary<string, double> ComputePersonalizedPageRank(
        IReadOnlyList<string> queryEntities,
        IReadOnlyList<EpisodicTrace> allTraces,
        double dampingAlpha = 0.85,
        int maxIterations = 20)
    {
        if (queryEntities == null || queryEntities.Count == 0 || allTraces == null || allTraces.Count == 0)
            return [];

        EntityCsrGraph graph = EntityCsrGraph.Build(allTraces);
        if (graph.NodeCount == 0)
            return [];

        int bufLen = graph.NodeCount + 1;
        float[]? rentedScores = bufLen > 1024 ? ArrayPool<float>.Shared.Rent(bufLen) : null;
        Span<float> pprScores = bufLen <= 1024
            ? stackalloc float[bufLen]
            : rentedScores.AsSpan(0, bufLen);

        try
        {
            ComputePersonalizedPageRank(
                graph,
                queryEntities,
                pprScores,
                dampingAlpha,
                maxIterations,
                fallbackToUniformIfNoSeeds: true);

            var result = new Dictionary<string, double>(graph.NodeCount, StringComparer.OrdinalIgnoreCase);
            for (int u = 1; u <= graph.NodeCount; u++)
            {
                result[graph.NodeNames[u]] = pprScores[u];
            }
            return result;
        }
        finally
        {
            if (rentedScores != null) ArrayPool<float>.Shared.Return(rentedScores);
        }
    }

    /// <summary>
    /// Executes contiguous-span <see cref="CsrKernels.PersonalizedPageRank"/> over a prebuilt <see cref="EntityCsrGraph"/>.
    /// Returns <see langword="true"/> if at least one query entity matched the graph (or uniform fallback ran);
    /// otherwise clears <paramref name="pprScoresOut"/> to zeros and returns <see langword="false"/>.
    /// </summary>
    internal static bool ComputePersonalizedPageRank(
        EntityCsrGraph graph,
        IReadOnlyList<string> queryEntities,
        Span<float> pprScoresOut,
        double dampingAlpha = 0.85,
        int maxIterations = 20,
        bool fallbackToUniformIfNoSeeds = false)
    {
        if (graph.NodeCount <= 0)
        {
            pprScoresOut.Clear();
            return false;
        }

        if (pprScoresOut.Length < graph.NodeCount + 1)
        {
            throw new ArgumentException($"pprScoresOut length ({pprScoresOut.Length}) must be >= {graph.NodeCount + 1}.", nameof(pprScoresOut));
        }

        if (queryEntities == null || queryEntities.Count == 0)
        {
            pprScoresOut.Slice(0, graph.NodeCount + 1).Clear();
            return false;
        }

        int maxSeeds = queryEntities.Count;
        int[]? rentedSeeds = maxSeeds > 64 ? ArrayPool<int>.Shared.Rent(maxSeeds) : null;
        Span<int> seedIds = maxSeeds <= 64
            ? stackalloc int[maxSeeds]
            : rentedSeeds.AsSpan(0, maxSeeds);

        int seedCount = 0;
        HashSet<int>? largeSeedSet = null;

        try
        {
            for (int i = 0; i < queryEntities.Count; i++)
            {
                string entity = queryEntities[i];
                if (string.IsNullOrWhiteSpace(entity)) continue;

                if (!graph.EntityToNodeId.TryGetValue(entity, out int nodeId))
                    continue;

                // Deduplicate matched seeds so personalization weights stay strictly uniform across unique query entities
                bool isDuplicate = false;
                if (seedCount <= 32)
                {
                    for (int k = 0; k < seedCount; k++)
                    {
                        if (seedIds[k] == nodeId)
                        {
                            isDuplicate = true;
                            break;
                        }
                    }
                }
                else
                {
                    if (largeSeedSet == null)
                    {
                        largeSeedSet = new HashSet<int>(seedCount);
                        for (int k = 0; k < seedCount; k++) largeSeedSet.Add(seedIds[k]);
                    }
                    isDuplicate = !largeSeedSet.Add(nodeId);
                }

                if (!isDuplicate)
                {
                    seedIds[seedCount++] = nodeId;
                }
            }

            if (seedCount == 0 && !fallbackToUniformIfNoSeeds)
            {
                pprScoresOut.Slice(0, graph.NodeCount + 1).Clear();
                return false;
            }

            float alpha = double.IsFinite(dampingAlpha)
                ? (float)Math.Clamp(dampingAlpha, 0.0, 1.0)
                : 0.85f;
            int iterations = Math.Max(0, maxIterations);

            CsrKernels.PersonalizedPageRank(
                graph.RowOffsets,
                graph.ColumnIndices,
                ReadOnlySpan<float>.Empty,
                graph.NodeCount,
                seedIds.Slice(0, seedCount),
                ReadOnlySpan<float>.Empty,
                pprScoresOut,
                alpha,
                iterations,
                ConvergenceTolerance);

            return true;
        }
        finally
        {
            if (rentedSeeds != null) ArrayPool<int>.Shared.Return(rentedSeeds);
        }
    }

    /// <summary>
    /// Scores traces based on the accumulated PPR spreading activation of their contained entities.
    /// </summary>
    public static double ScoreTrace(EpisodicTrace trace, IReadOnlyDictionary<string, double> pprScores)
    {
        if (trace?.Entities == null || trace.Entities.Count == 0 || pprScores == null || pprScores.Count == 0) return 0.0;

        double sum = 0.0;
        if (trace.Entities.Count == 1)
        {
            string only = trace.Entities[0];
            if (!string.IsNullOrWhiteSpace(only) && pprScores.TryGetValue(only, out double s))
            {
                return s;
            }
            return 0.0;
        }

        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var entity in trace.Entities)
        {
            if (!string.IsNullOrWhiteSpace(entity) && seen.Add(entity) && pprScores.TryGetValue(entity, out double s))
            {
                sum += s;
            }
        }
        return sum;
    }
}
