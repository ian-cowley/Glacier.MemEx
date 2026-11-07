namespace Glacier.MemEx.Recall;

using System;
using System.Collections.Generic;
using System.Linq;
using Glacier.MemEx.Models;

public static class HippoRagSpreadingActivation
{
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
        if (queryEntities.Count == 0 || allTraces.Count == 0) return [];

        // 1. Build entity co-occurrence adjacency list
        var graph = new Dictionary<string, HashSet<string>>(StringComparer.OrdinalIgnoreCase);
        foreach (var trace in allTraces)
        {
            var ents = trace.Entities;
            if (ents.Count == 0) continue;

            for (int i = 0; i < ents.Count; i++)
            {
                if (!graph.TryGetValue(ents[i], out var neighbors))
                {
                    neighbors = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                    graph[ents[i]] = neighbors;
                }

                for (int j = 0; j < ents.Count; j++)
                {
                    if (i != j) neighbors.Add(ents[j]);
                }
            }
        }

        if (graph.Count == 0) return [];

        var allNodes = graph.Keys.ToList();
        int n = allNodes.Count;

        // 2. Initialize teleport distribution vector p0
        var p0 = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
        var activeSeeds = queryEntities.Where(graph.ContainsKey).ToList();

        if (activeSeeds.Count == 0)
        {
            double uniform = 1.0 / n;
            foreach (var node in allNodes) p0[node] = uniform;
        }
        else
        {
            double seedWeight = 1.0 / activeSeeds.Count;
            foreach (var s in activeSeeds) p0[s] = seedWeight;
            foreach (var node in allNodes) if (!p0.ContainsKey(node)) p0[node] = 0.0;
        }

        // 3. Power iteration: p = (1 - alpha) * p0 + alpha * (M * p)
        var p = new Dictionary<string, double>(p0, StringComparer.OrdinalIgnoreCase);
        var pNext = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);

        for (int iter = 0; iter < maxIterations; iter++)
        {
            foreach (var node in allNodes)
            {
                pNext[node] = (1.0 - dampingAlpha) * p0[node];
            }

            foreach (var u in allNodes)
            {
                var neighbors = graph[u];
                if (neighbors.Count == 0)
                {
                    // Dangling node distributes to teleport
                    double contrib = dampingAlpha * p[u] / n;
                    foreach (var v in allNodes) pNext[v] += contrib;
                }
                else
                {
                    double outWeight = dampingAlpha * p[u] / neighbors.Count;
                    foreach (var v in neighbors) pNext[v] += outWeight;
                }
            }

            // Swap buffers
            foreach (var node in allNodes) p[node] = pNext[node];
        }

        return p;
    }

    /// <summary>
    /// Scores traces based on the accumulated PPR spreading activation of their contained entities.
    /// </summary>
    public static double ScoreTrace(EpisodicTrace trace, IReadOnlyDictionary<string, double> pprScores)
    {
        if (trace.Entities.Count == 0 || pprScores.Count == 0) return 0.0;

        double sum = 0.0;
        foreach (var entity in trace.Entities)
        {
            if (pprScores.TryGetValue(entity, out double s))
            {
                sum += s;
            }
        }
        return sum;
    }
}
