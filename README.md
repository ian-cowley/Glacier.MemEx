# Glacier.MemEx

Episodic memory engine for .NET 10 combining ACT-R power-law temporal decay, Fast Walsh-Hadamard (FWHT) 1-bit + FP32 two-tier vector recall, cached Compressed Sparse Row (CSR) HippoRAG Personalized PageRank, and token-budgeted 0/1 knapsack prompt packing.

## Architecture

`MemExEngine` scores every stored `EpisodicTrace` across four normalized signals:

1. **Temporal Recency (`PowerLawDecayCalculator`)**: ACT-R base-level activation $B_i(t) = \ln \sum_k (t - t_k + \tau)^{-d}$ mapped via sigmoid into $[0, 1]$, accounting for both creation time and retrieval history.
2. **Two-Tier Vector Recall (`TwoTierVectorFilter` + `Glacier.Vector`)**:
   - Quantizes embeddings with deterministic randomized Hadamard rotation (`HadamardProjection.RotateAndBinarize`) and SIMD Hamming distance (`DistanceKernels.HammingDistance`) via bounded pooled top-K heaps.
   - Re-ranks the top `CoarseCandidateLimit` candidates with exact FP32 cosine similarity (`DistanceKernels.DotProduct` with overflow-safe `double` fallback).
   - Scores valid-dimension traces outside `CoarseCandidateLimit` using Charikar's angular estimator (`HadamardProjection.EstimateCosineFromHamming`) rather than zeroing them out, while leaving unembedded or dimension-mismatched traces at `0.0`.
3. **Associative Entity Graph (`HippoRagSpreadingActivation` + `Glacier.Graph`)**:
   - Maintains a version-cached 1-based CSR entity co-occurrence graph (`EntityCsrGraph`) invalidated automatically on entity-bearing `StoreTrace`, `StoreTraceAsync`, `PruneTraces`, and `PruneDecayedTraces` mutations.
   - Runs contiguous-span `CsrKernels.PersonalizedPageRank` with dangling-node probability mass teleported strictly to the normalized seed distribution $p_0$.
4. **Base Importance & Composite Packing (`KnapsackPromptPacker` + `BreadcrumbTreeFolder` + `Glacier.Bundle`)**:
   - Combines weighted signals into $u_i = \exp(A_i(t))$ and solves an `ArrayPool`-backed 0/1 DP knapsack (or density-greedy vs. best-single approximation above `200,000` DP cells).
   - Uses `ITokenEstimator` (`HeuristicTokenEstimator` by default) and `TokenBudget` to account for formatting overhead and enforce `FormattedContext <= maxTokens` (including adaptive `FoldDeepSubtrees` folding).

## Dependencies

- `Glacier.Graph` — `CsrKernels.PersonalizedPageRank`
- `Glacier.Vector` — `HadamardProjection`, `DistanceKernels`
- `Glacier.Bundle` — `ITokenEstimator`, `HeuristicTokenEstimator`, `TokenBudget`
- `Glacier.Rag` — `IEmbeddingModel`, `OpenAiEmbeddingModel`, `FastHashEmbeddingModel`
- `Glacier.Inference` — `NpuSidecarDiscovery`, `NpuSidecarEndpoint`, `NpuSidecarRuntime`

## Quick Start

```csharp
using Glacier.MemEx;
using Glacier.MemEx.Models;

using var engine = new MemExEngine(new MemExOptions
{
    MaxPromptTokens = 512,
    CoarseCandidateLimit = 64,
    FoldDeepSubtrees = true,
    EnableNpuEmbeddings = false // Falls back cleanly to offline FastHashEmbeddingModel(384)
});

await engine.StoreTraceAsync(new EpisodicTrace
{
    Id = "bridge",
    Content = "ServiceA publishes telemetry events to the Redis stream.",
    Breadcrumbs = ["Architecture", "Messaging"],
    Entities = ["ServiceA", "Redis"],
    BaseImportance = 2.0f
});

await engine.StoreTraceAsync(new EpisodicTrace
{
    Id = "hop2",
    Content = "Redis consumer lag triggered downstream request timeouts.",
    Breadcrumbs = ["Incidents", "Latency"],
    Entities = ["Redis", "Timeout"],
    BaseImportance = 2.5f
});

MemoryQueryResult result = await engine.QueryAsync(
    queryText: "investigating latency spikes",
    queryEntities: ["ServiceA"],
    maxTokens: 256);

Console.WriteLine(result.FormattedContext);
```

### Sleep Consolidation & Pruning

```csharp
IReadOnlyList<EpisodicTrace> hot = engine.ExportConsolidationCandidates(maxCount: 16, minAccessCount: 1);
int removedById = engine.PruneTraces(["old-trace-id"]);
int removedStale = engine.PruneDecayedTraces(minRecencyScore: 0.05);
```

## Building & Testing

Standalone NuGet CI mode (uses published NuGet dependencies, no sibling checkout required):

```bash
dotnet build Glacier.MemEx/src/Glacier.MemEx/Glacier.MemEx.csproj -c Release -p:UseLocalProjectReferences=false
dotnet test Glacier.MemEx/tests/Glacier.MemEx.Tests/Glacier.MemEx.Tests.csproj -c Release -p:UseLocalProjectReferences=false
```

Local monorepo `ProjectReference` mode:

```bash
dotnet test Glacier.MemEx/tests/Glacier.MemEx.Tests/Glacier.MemEx.Tests.csproj -c Release
```

## License

MIT
