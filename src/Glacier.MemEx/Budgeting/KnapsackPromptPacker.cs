namespace Glacier.MemEx.Budgeting;

using System;
using System.Buffers;
using System.Collections.Generic;
using Glacier.MemEx.Models;

public static class KnapsackPromptPacker
{
    private const int MaxDpCells = 200_000;

    /// <summary>
    /// Solves the 0/1 knapsack prompt budget optimization problem:
    /// Selects a subset of episodic traces that maximizes total utility sum(u_i)
    /// subject to total token count sum(w_i) &lt;= maxTokens.
    /// </summary>
    public static List<EpisodicTrace> PackTraces(
        IReadOnlyList<(EpisodicTrace Trace, double Utility)> candidateTraces,
        int maxTokens)
    {
        ArgumentNullException.ThrowIfNull(candidateTraces);
        if (candidateTraces.Count == 0 || maxTokens <= 0) return [];

        int count = candidateTraces.Count;
        int[] validSourceIndices = ArrayPool<int>.Shared.Rent(count);
        int[] validCosts = ArrayPool<int>.Shared.Rent(count);
        double[] validUtilities = ArrayPool<double>.Shared.Rent(count);
        int[] selectedSourceIndices = ArrayPool<int>.Shared.Rent(count);

        try
        {
            int n = 0;
            for (int i = 0; i < count; i++)
            {
                var (trace, utility) = candidateTraces[i];
                if (trace == null || !(utility > 0.0) || !double.IsFinite(utility))
                    continue;

                int w = Math.Max(1, trace.TokenCount);
                if (w > maxTokens)
                    continue;

                validSourceIndices[n] = i;
                validCosts[n] = w;
                validUtilities[n] = utility;
                n++;
            }

            if (n == 0) return [];

            PackIndicesCore(
                n,
                validSourceIndices.AsSpan(0, n),
                validCosts.AsSpan(0, n),
                validUtilities.AsSpan(0, n),
                maxTokens,
                selectedSourceIndices,
                out int selectedCount,
                out _);

            var result = new List<EpisodicTrace>(selectedCount);
            for (int i = 0; i < selectedCount; i++)
            {
                result.Add(candidateTraces[selectedSourceIndices[i]].Trace);
            }
            return result;
        }
        finally
        {
            ArrayPool<int>.Shared.Return(selectedSourceIndices);
            ArrayPool<double>.Shared.Return(validUtilities);
            ArrayPool<int>.Shared.Return(validCosts);
            ArrayPool<int>.Shared.Return(validSourceIndices);
        }
    }

    internal static void PackTraceIndices(
        IReadOnlyList<EpisodicTrace> traces,
        ReadOnlySpan<double> utilities,
        ReadOnlySpan<int> tokenCosts,
        int maxTokens,
        Span<int> selectedIndicesOut,
        out int selectedCount,
        out double totalUtility)
    {
        selectedCount = 0;
        totalUtility = 0.0;

        int count = traces.Count;
        if (count == 0 || maxTokens <= 0) return;

        int[] validSourceIndices = ArrayPool<int>.Shared.Rent(count);
        int[] validCosts = ArrayPool<int>.Shared.Rent(count);
        double[] validUtilities = ArrayPool<double>.Shared.Rent(count);

        try
        {
            int n = 0;
            for (int i = 0; i < count; i++)
            {
                double utility = utilities[i];
                if (traces[i] == null || !(utility > 0.0) || !double.IsFinite(utility))
                    continue;

                int w = Math.Max(1, tokenCosts[i]);
                if (w > maxTokens)
                    continue;

                validSourceIndices[n] = i;
                validCosts[n] = w;
                validUtilities[n] = utility;
                n++;
            }

            if (n == 0) return;

            PackIndicesCore(
                n,
                validSourceIndices.AsSpan(0, n),
                validCosts.AsSpan(0, n),
                validUtilities.AsSpan(0, n),
                maxTokens,
                selectedIndicesOut,
                out selectedCount,
                out totalUtility);
        }
        finally
        {
            ArrayPool<double>.Shared.Return(validUtilities);
            ArrayPool<int>.Shared.Return(validCosts);
            ArrayPool<int>.Shared.Return(validSourceIndices);
        }
    }

    private static void PackIndicesCore(
        int n,
        ReadOnlySpan<int> validSourceIndices,
        ReadOnlySpan<int> validCosts,
        ReadOnlySpan<double> validUtilities,
        int maxTokens,
        Span<int> selectedSourceIndicesOut,
        out int selectedCount,
        out double totalUtility)
    {
        selectedCount = 0;
        totalUtility = 0.0;

        long strideLong = (long)maxTokens + 1L;
        if (maxTokens < int.MaxValue && (long)n * strideLong <= MaxDpCells)
        {
            int stride = (int)strideLong;
            int keepLen = checked(n * stride);

            double[] dpArr = ArrayPool<double>.Shared.Rent(stride);
            bool[] keepArr = ArrayPool<bool>.Shared.Rent(keepLen);

            try
            {
                Span<double> dp = dpArr.AsSpan(0, stride);
                Span<bool> keep = keepArr.AsSpan(0, keepLen);
                dp.Clear();
                keep.Clear();

                for (int i = 0; i < n; i++)
                {
                    int w = validCosts[i];
                    double u = validUtilities[i];
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

                int rem = maxTokens;
                for (int i = n - 1; i >= 0; i--)
                {
                    if (keep[(i * stride) + rem])
                    {
                        selectedSourceIndicesOut[selectedCount++] = validSourceIndices[i];
                        totalUtility += validUtilities[i];
                        rem -= validCosts[i];
                    }
                }

                selectedSourceIndicesOut.Slice(0, selectedCount).Reverse();
                return;
            }
            finally
            {
                ArrayPool<bool>.Shared.Return(keepArr);
                ArrayPool<double>.Shared.Return(dpArr);
            }
        }

        // Large-budget/count fallback: density-greedy vs best single item (>= 1/2 approximation guarantee)
        (int SourceIndex, int Tokens, double Utility, double Density)[] rankedBuf =
            ArrayPool<(int SourceIndex, int Tokens, double Utility, double Density)>.Shared.Rent(n);

        try
        {
            Span<(int SourceIndex, int Tokens, double Utility, double Density)> ranked = rankedBuf.AsSpan(0, n);
            int bestSingleSourceIdx = -1;
            double bestSingleUtility = double.NegativeInfinity;

            for (int i = 0; i < n; i++)
            {
                int tokens = validCosts[i];
                double utility = validUtilities[i];
                ranked[i] = (validSourceIndices[i], tokens, utility, utility / tokens);

                if (utility > bestSingleUtility)
                {
                    bestSingleUtility = utility;
                    bestSingleSourceIdx = validSourceIndices[i];
                }
            }

            ranked.Sort(static (a, b) =>
            {
                int cmp = b.Density.CompareTo(a.Density);
                return cmp != 0 ? cmp : b.Utility.CompareTo(a.Utility);
            });

            long currentTokens = 0;
            double greedyUtility = 0.0;
            for (int i = 0; i < n; i++)
            {
                ref readonly var item = ref ranked[i];
                if (currentTokens + item.Tokens <= maxTokens)
                {
                    selectedSourceIndicesOut[selectedCount++] = item.SourceIndex;
                    currentTokens += item.Tokens;
                    greedyUtility += item.Utility;
                }
            }

            if (bestSingleSourceIdx >= 0 && bestSingleUtility > greedyUtility)
            {
                selectedSourceIndicesOut[0] = bestSingleSourceIdx;
                selectedCount = 1;
                totalUtility = bestSingleUtility;
                return;
            }

            totalUtility = greedyUtility;
        }
        finally
        {
            ArrayPool<(int SourceIndex, int Tokens, double Utility, double Density)>.Shared.Return(rankedBuf);
        }
    }
}
