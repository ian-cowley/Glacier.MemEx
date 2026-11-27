namespace Glacier.MemEx.Tests;

using System;
using System.Collections.Generic;
using Glacier.MemEx.Budgeting;
using Glacier.MemEx.Models;
using Xunit;

public class KnapsackPromptPackerTests
{
    [Fact]
    public void KnapsackPromptPacker_EnforcesStrictTokenBudget_AndMaximizesDensity()
    {
        var traces = new List<(EpisodicTrace Trace, double Utility)>
        {
            (new EpisodicTrace { Content = "Small High Utility", TokenCount = 50 }, 100.0),
            (new EpisodicTrace { Content = "Medium High Utility", TokenCount = 100 }, 150.0),
            (new EpisodicTrace { Content = "Huge Low Utility", TokenCount = 300 }, 20.0),
            (new EpisodicTrace { Content = "Tiny Moderate Utility", TokenCount = 20 }, 30.0)
        };

        int maxTokens = 160;
        var selected = KnapsackPromptPacker.PackTraces(traces, maxTokens);

        int totalTokens = 0;
        foreach (var t in selected) totalTokens += t.TokenCount;

        // Must respect budget constraint
        Assert.True(totalTokens <= maxTokens);

        // High density items (Small High Utility, Medium High Utility) should be included
        Assert.Contains(selected, t => t.Content == "Small High Utility");
        Assert.Contains(selected, t => t.Content == "Medium High Utility");

        // Huge Low Utility item (300 tokens) must NOT be included
        Assert.DoesNotContain(selected, t => t.Content == "Huge Low Utility");
    }

    [Fact]
    public void KnapsackPromptPacker_PicksOptimalSubsetOverGreedyDensityDecoy()
    {
        // Classic greedy counterexample: capacity=10
        // Item A: w=1, u=2 (density=2.0) -> decoy
        // Item B: w=10, u=19 (density=1.9) -> optimal
        var decoy = new EpisodicTrace { Content = "Decoy", TokenCount = 1 };
        var optimal = new EpisodicTrace { Content = "Optimal", TokenCount = 10 };

        var selected = KnapsackPromptPacker.PackTraces([(decoy, 2.0), (optimal, 19.0)], maxTokens: 10);
        Assert.Single(selected);
        Assert.Equal("Optimal", selected[0].Content);
    }
}
