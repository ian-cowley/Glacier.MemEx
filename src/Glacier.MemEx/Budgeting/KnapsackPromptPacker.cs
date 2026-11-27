namespace Glacier.MemEx.Budgeting;

using System;
using System.Collections.Generic;
using System.Linq;
using Glacier.MemEx.Models;

public static class KnapsackPromptPacker
{
    /// <summary>
    /// Solves the 0/1 knapsack prompt budget optimization problem:
    /// Selects a subset of episodic traces that maximizes total utility sum(u_i)
    /// subject to total token count sum(w_i) <= maxTokens.
    /// </summary>
    public static List<EpisodicTrace> PackTraces(
        IReadOnlyList<(EpisodicTrace Trace, double Utility)> candidateTraces,
        int maxTokens)
    {
        if (candidateTraces.Count == 0 || maxTokens <= 0) return [];

        // Filter out zero/negative utility items using normalized token cost
        var validCandidates = candidateTraces
            .Where(c => c.Utility > 0.0 && Math.Max(1, c.Trace.TokenCount) <= maxTokens)
            .ToList();

        if (validCandidates.Count == 0) return [];

        int n = validCandidates.Count;
        // Exact 0/1 knapsack DP when state table fits comfortably in memory
        if ((long)n * maxTokens <= 200_000)
        {
            var dp = new double[maxTokens + 1];
            // Track backpointers in a compact bit-packed/bool matrix (only <= 200,000 entries)
            var keep = new bool[n * (maxTokens + 1)];
            int stride = maxTokens + 1;

            for (int i = 0; i < n; i++)
            {
                int w = Math.Max(1, validCandidates[i].Trace.TokenCount);
                double u = validCandidates[i].Utility;
                int rowOffset = i * stride;
                for (int c = maxTokens; c >= w; c--)
                {
                    double withItem = dp[c - w] + u;
                    if (withItem > dp[c])
                    {
                        dp[c] = withItem;
                        keep[rowOffset + c] = true;
                    }
                }
            }

            var optimal = new List<EpisodicTrace>();
            int rem = maxTokens;
            for (int i = n - 1; i >= 0; i--)
            {
                if (keep[i * stride + rem])
                {
                    optimal.Add(validCandidates[i].Trace);
                    rem -= Math.Max(1, validCandidates[i].Trace.TokenCount);
                }
            }
            optimal.Reverse();
            return optimal;
        }

        // For very large budgets/counts, compare greedy density packing vs best single item
        // (guaranteed >= 1/2 approximation ratio vs unboundedly bad density-only greedy).
        var rankedByDensity = validCandidates
            .Select(c => new
            {
                c.Trace,
                c.Utility,
                Tokens = Math.Max(1, c.Trace.TokenCount),
                Density = c.Utility / Math.Max(1, c.Trace.TokenCount)
            })
            .OrderByDescending(x => x.Density)
            .ThenByDescending(x => x.Utility)
            .ToList();

        var selected = new List<EpisodicTrace>();
        int currentTokens = 0;
        double greedyUtility = 0.0;

        foreach (var item in rankedByDensity)
        {
            if (currentTokens + item.Tokens <= maxTokens)
            {
                selected.Add(item.Trace);
                currentTokens += item.Tokens;
                greedyUtility += item.Utility;
            }
        }

        EpisodicTrace? bestSingle = null;
        double bestSingleUtility = double.NegativeInfinity;
        for (int i = 0; i < n; i++)
        {
            if (validCandidates[i].Utility > bestSingleUtility)
            {
                bestSingleUtility = validCandidates[i].Utility;
                bestSingle = validCandidates[i].Trace;
            }
        }

        if (bestSingle != null && bestSingleUtility > greedyUtility)
        {
            return [bestSingle];
        }

        return selected;
    }
}
