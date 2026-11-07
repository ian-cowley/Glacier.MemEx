namespace Glacier.MemEx.Tests;

using System;
using Glacier.MemEx.Decay;
using Glacier.MemEx.Models;
using Xunit;

public class PowerLawDecayTests
{
    [Fact]
    public void BaseLevelActivation_MonotonicallyDecreases_OverTime()
    {
        long t0 = 1_000_000_000L; // 1000 seconds
        var trace = new EpisodicTrace
        {
            Content = "System test",
            TimestampUs = t0
        };

        double act10s = PowerLawDecayCalculator.ComputeBaseLevelActivation(trace, t0 + 10_000_000L, d: 0.5);
        double act100s = PowerLawDecayCalculator.ComputeBaseLevelActivation(trace, t0 + 100_000_000L, d: 0.5);
        double act1000s = PowerLawDecayCalculator.ComputeBaseLevelActivation(trace, t0 + 1_000_000_000L, d: 0.5);

        Assert.True(act10s > act100s, $"10s activation ({act10s}) should exceed 100s ({act100s})");
        Assert.True(act100s > act1000s, $"100s activation ({act100s}) should exceed 1000s ({act1000s})");
    }

    [Fact]
    public void FrequencyEffect_MultipleRetrievals_IncreasesActivation()
    {
        long t0 = 1_000_000_000L;
        var traceSingle = new EpisodicTrace
        {
            Content = "Single access item",
            TimestampUs = t0
        };

        var traceMulti = new EpisodicTrace
        {
            Content = "Frequently retrieved item",
            TimestampUs = t0
        };
        traceMulti.RecordAccess(t0 + 20_000_000L); // retrieved at 20s
        traceMulti.RecordAccess(t0 + 40_000_000L); // retrieved at 40s
        traceMulti.RecordAccess(t0 + 60_000_000L); // retrieved at 60s

        long queryTime = t0 + 100_000_000L; // 100s
        double actSingle = PowerLawDecayCalculator.ComputeBaseLevelActivation(traceSingle, queryTime, d: 0.5);
        double actMulti = PowerLawDecayCalculator.ComputeBaseLevelActivation(traceMulti, queryTime, d: 0.5);

        Assert.True(actMulti > actSingle, $"Multi-access activation ({actMulti}) must exceed single ({actSingle})");
    }
}
