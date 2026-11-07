namespace Glacier.MemEx.Decay;

using System;
using System.Collections.Generic;
using Glacier.MemEx.Models;

public static class PowerLawDecayCalculator
{
    /// <summary>
    /// Computes ACT-R base-level activation:
    /// B_i(t) = ln( sum_{k=1}^n (t - t_k + tau)^(-d) ).
    /// </summary>
    public static double ComputeBaseLevelActivation(
        EpisodicTrace trace,
        long currentTimestampUs,
        double d = 0.5,
        double tauSeconds = 1.0)
    {
        ArgumentNullException.ThrowIfNull(trace);
        if (d <= 0.0) throw new ArgumentOutOfRangeException(nameof(d), "Decay parameter d must be positive.");
        if (tauSeconds <= 0.0) throw new ArgumentOutOfRangeException(nameof(tauSeconds), "Time scale tau must be positive.");

        double sum = 0.0;

        // 1. Creation event
        double deltaSecCreation = Math.Max(0.0, (currentTimestampUs - trace.TimestampUs) / 1_000_000.0);
        sum += Math.Pow(deltaSecCreation + tauSeconds, -d);

        // 2. Historical retrieval / access events
        sum += trace.ComputeAccessDecaySum(currentTimestampUs, d, tauSeconds);

        return Math.Log(Math.Max(1e-12, sum));
    }

    /// <summary>
    /// Computes normalized recency score in range [0, 1] derived from base-level activation.
    /// </summary>
    public static double ComputeRecencyScore(
        EpisodicTrace trace,
        long currentTimestampUs,
        double d = 0.5,
        double tauSeconds = 1.0)
    {
        double b = ComputeBaseLevelActivation(trace, currentTimestampUs, d, tauSeconds);
        // Sigmoid mapping from activation space to [0, 1]
        return 1.0 / (1.0 + Math.Exp(-b));
    }
}
