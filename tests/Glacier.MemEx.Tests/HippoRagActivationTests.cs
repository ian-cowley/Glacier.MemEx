namespace Glacier.MemEx.Tests;

using System;
using System.Collections.Generic;
using Glacier.MemEx.Models;
using Glacier.MemEx.Recall;
using Xunit;

public class HippoRagActivationTests
{
    [Fact]
    public void HippoRag_SpreadsActivation_AcrossAssociativeEntityGraph()
    {
        // Graph structure:
        // Trace 1: "Glacier", "Storage"
        // Trace 2: "Storage", "BufferPool"
        // Trace 3: "BufferPool", "ClockPro"
        // Trace 4: "Physics", "Thermodynamics" (unconnected component)

        var traces = new List<EpisodicTrace>
        {
            new EpisodicTrace { Content = "T1", Entities = ["Glacier", "Storage"] },
            new EpisodicTrace { Content = "T2", Entities = ["Storage", "BufferPool"] },
            new EpisodicTrace { Content = "T3", Entities = ["BufferPool", "ClockPro"] },
            new EpisodicTrace { Content = "T4", Entities = ["Physics", "Thermodynamics"] }
        };

        // Seed with "Glacier"
        var ppr = HippoRagSpreadingActivation.ComputePersonalizedPageRank(["Glacier"], traces, dampingAlpha: 0.85);

        Assert.True(ppr.ContainsKey("Glacier"));
        Assert.True(ppr.ContainsKey("Storage"));
        Assert.True(ppr.ContainsKey("BufferPool"));
        Assert.True(ppr.ContainsKey("ClockPro"));

        // Connected path scores are positive and decay with hop distance from Storage bridge
        Assert.True(ppr["Glacier"] > 0.30);
        Assert.True(ppr["Storage"] > ppr["BufferPool"]);
        Assert.True(ppr["BufferPool"] > ppr["ClockPro"]);

        // Disconnected component has 0 activation
        Assert.Equal(0.0, ppr["Physics"]);
        Assert.Equal(0.0, ppr["Thermodynamics"]);
    }
}
