namespace Glacier.MemEx.Tests;

using System;
using System.Collections.Generic;
using Glacier.MemEx.Budgeting;
using Glacier.MemEx.Decay;
using Glacier.MemEx.Models;
using Xunit;

public class BreadcrumbAndTraceCoverageTests
{
    [Fact]
    public void BreadcrumbTreeFolder_EmptyTraces_ReturnsEmptyString()
    {
        Assert.Equal(string.Empty, BreadcrumbTreeFolder.FormatTraceContext(Array.Empty<EpisodicTrace>()));
    }

    [Fact]
    public void BreadcrumbTreeFolder_FormatsBreadcrumbsEntitiesAndFolding()
    {
        string longBody = new string('A', 300);
        var traces = new[]
        {
            new EpisodicTrace
            {
                Breadcrumbs = ["Session1", "Auth", "TokenRefresh"],
                Entities = ["OAuth", "JWT"],
                Content = "  Short summary  "
            },
            new EpisodicTrace
            {
                Content = longBody
            }
        };

        string unfolded = BreadcrumbTreeFolder.FormatTraceContext(traces, foldDeepSubtrees: false);
        Assert.Contains("### [1] Session1 > Auth > TokenRefresh", unfolded);
        Assert.Contains("*Entities*: OAuth, JWT", unfolded);
        Assert.Contains("### [2] Episodic Trace", unfolded);
        Assert.Contains(longBody, unfolded);

        string folded = BreadcrumbTreeFolder.FormatTraceContext(traces, foldDeepSubtrees: true);
        Assert.Contains("... [folded subtree]", folded);
        Assert.DoesNotContain(longBody, folded);
    }

    [Fact]
    public void EpisodicTrace_BatchRecordAccessAndSnapshot_WorkCorrectly()
    {
        var trace = new EpisodicTrace { TimestampUs = 1_000_000L };
        Assert.Equal(0, trace.AccessCount);
        Assert.Equal(0.0, trace.ComputeAccessDecaySum(2_000_000L, 0.5, 1.0));
        Assert.Empty(trace.GetAccessTimestampsSnapshot());

        Assert.Throws<ArgumentNullException>(() => trace.RecordAccess((IEnumerable<long>)null!));

        trace.RecordAccess(new[] { 1_500_000L, 1_800_000L });
        Assert.Equal(2, trace.AccessCount);
        Assert.Equal(new[] { 1_500_000L, 1_800_000L }, trace.GetAccessTimestampsSnapshot());
        Assert.True(trace.ComputeAccessDecaySum(2_000_000L, 0.5, 1.0) > 0.0);
    }

    [Fact]
    public void PowerLawDecayCalculator_InvalidArgsAndRecencyBounds_WorkCorrectly()
    {
        var trace = new EpisodicTrace { TimestampUs = 1_000_000L };

        Assert.Throws<ArgumentNullException>(() => PowerLawDecayCalculator.ComputeBaseLevelActivation(null!, 2_000_000L));
        Assert.Throws<ArgumentOutOfRangeException>(() => PowerLawDecayCalculator.ComputeBaseLevelActivation(trace, 2_000_000L, d: 0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => PowerLawDecayCalculator.ComputeBaseLevelActivation(trace, 2_000_000L, tauSeconds: -1.0));

        double recency = PowerLawDecayCalculator.ComputeRecencyScore(trace, 2_000_000L);
        Assert.InRange(recency, 0.0, 1.0);
    }
}
