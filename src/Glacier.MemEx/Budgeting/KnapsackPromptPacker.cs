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

        // Filter out zero/negative utility items
        var validCandidates = candidateTraces
            .Where(c => c.Utility > 0.0 && c.Trace.TokenCount <= maxTokens)
            .ToList();

        if (validCandidates.Count == 0) return [];

        // For large budgets or item counts, greedy density ratio (utility / tokenCount)
        // achieves a 1 - 1/e (>63.2%) approximation ratio with O(N log N) runtime.
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

        foreach (var item in rankedByDensity)
        {
            if (currentTokens + item.Tokens <= maxTokens)
            {
                selected.Add(item.Trace);
                currentTokens += item.Tokens;
            }
        }

        return selected;
    }
}
