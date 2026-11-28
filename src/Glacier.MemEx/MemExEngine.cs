namespace Glacier.MemEx;

using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Bundle.Budgeting;
using Glacier.Inference.Hardware;
using Glacier.MemEx.Budgeting;
using Glacier.MemEx.Decay;
using Glacier.MemEx.Models;
using Glacier.MemEx.Recall;
using Glacier.Rag.Embeddings;

public sealed class MemExEngine : IDisposable
{
    private const string DefaultEmbeddingModelId = "embeddinggemma";
    private const int QueryStackAllocLimit = 256;

    private readonly ConcurrentDictionary<string, EpisodicTrace> _traces = new(StringComparer.Ordinal);
    private readonly MemExOptions _options;
    private readonly ITokenEstimator _tokenEstimator;
    private readonly SemaphoreSlim _initGate = new(1, 1);
    private readonly CancellationTokenSource _disposeCts = new();
    private readonly object _fallbackLock = new();
    private readonly object _graphBuildLock = new();
    private readonly List<IDisposable> _retiredModels = [];

    private IEmbeddingModel? _resolvedModel;
    private FastHashEmbeddingModel? _fallbackHashModel;
    private EntityCsrGraph? _cachedEntityGraph;
    private long _topologyVersion;
    private int _resolvedDimensions;
    private bool _ownsResolvedModel;
    private string? _activeEndpoint;
    private string? _activeModelId;
    private int _disposed;

    public MemExEngine(MemExOptions? options = null)
    {
        _options = options ?? new MemExOptions();
        _tokenEstimator = _options.TokenEstimator ?? new HeuristicTokenEstimator();

        if (_options.FallbackHashDimensions <= 0 || _options.FallbackHashDimensions % 8 != 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                "FallbackHashDimensions must be a positive multiple of 8.");
        }

        if (_options.EmbeddingDimensions < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                "EmbeddingDimensions cannot be negative.");
        }

        if (_options.ProbeTimeout is { } probeTimeout &&
            probeTimeout <= TimeSpan.Zero &&
            probeTimeout != Timeout.InfiniteTimeSpan)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                "ProbeTimeout must be positive.");
        }

        if (_options.EmbeddingTimeout is { } embeddingTimeout &&
            embeddingTimeout <= TimeSpan.Zero &&
            embeddingTimeout != Timeout.InfiniteTimeSpan)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                "EmbeddingTimeout must be positive.");
        }

        if (_options.EmbeddingModel is { Dimensions: <= 0 })
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                "EmbeddingModel.Dimensions must be positive.");
        }
    }

    public int Count => _traces.Count;

    public int? ActiveEmbeddingDimensions
    {
        get
        {
            int dims = Volatile.Read(ref _resolvedDimensions);
            return dims > 0 ? dims : null;
        }
    }

    public bool IsUsingNpuEmbeddings => Volatile.Read(ref _resolvedModel) is OpenAiEmbeddingModel;

    public string? ActiveEmbeddingEndpoint => Volatile.Read(ref _activeEndpoint);

    public string? ActiveEmbeddingModelId => Volatile.Read(ref _activeModelId);

    public async Task<bool> InitializeEmbeddingsAsync(CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        cancellationToken.ThrowIfCancellationRequested();

        IEmbeddingModel model = await ResolveEmbeddingModelAsync(cancellationToken).ConfigureAwait(false);
        return model is OpenAiEmbeddingModel;
    }

    public void StoreTrace(EpisodicTrace trace)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(trace);

        EnsureTokenCount(trace);

        float[]? existing = trace.Embedding;
        if ((existing == null || existing.Length == 0) &&
            _options.AutoEmbedMissingVectors &&
            !string.IsNullOrWhiteSpace(trace.Content))
        {
            IEmbeddingModel? resolved = Volatile.Read(ref _resolvedModel);
            if (resolved != null && resolved is not OpenAiEmbeddingModel)
            {
                trace.SetEmbedding(GenerateSyncEmbedding(resolved, trace.Content));
            }
            else
            {
                float[] embedded = BridgeSync(() => ComputeAutoEmbeddingAsync(trace.Content, CancellationToken.None));
                trace.SetEmbedding(embedded);
            }
        }

        QuantizeTraceIfNeeded(trace);
        StoreTraceWithTopologyUpdate(trace);
    }

    public async Task StoreTraceAsync(EpisodicTrace trace, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(trace);
        cancellationToken.ThrowIfCancellationRequested();

        EnsureTokenCount(trace);

        float[]? existing = trace.Embedding;
        if ((existing == null || existing.Length == 0) &&
            _options.AutoEmbedMissingVectors &&
            !string.IsNullOrWhiteSpace(trace.Content))
        {
            float[] embedded = await ComputeAutoEmbeddingAsync(trace.Content, cancellationToken).ConfigureAwait(false);
            trace.SetEmbedding(embedded);
        }

        QuantizeTraceIfNeeded(trace);
        StoreTraceWithTopologyUpdate(trace);
    }

    public MemoryQueryResult Query(
        string queryText,
        ReadOnlySpan<float> queryEmbedding,
        IReadOnlyList<string>? queryEntities = null,
        long? currentTimestampUs = null,
        int? maxTokens = null)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        if (queryEmbedding.Length > 0 && !TwoTierVectorFilter.AreAllFinite(queryEmbedding))
        {
            throw new ArgumentException("Query embedding contains non-finite (NaN or Infinity) values.", nameof(queryEmbedding));
        }

        if (_traces.IsEmpty)
        {
            return new MemoryQueryResult();
        }

        var traceList = _traces.Values.ToList();
        if (traceList.Count == 0)
        {
            return new MemoryQueryResult();
        }

        if (queryEmbedding.IsEmpty &&
            _options.AutoEmbedMissingVectors &&
            !string.IsNullOrWhiteSpace(queryText) &&
            HasAnyEmbeddedTrace(traceList))
        {
            IEmbeddingModel? resolved = Volatile.Read(ref _resolvedModel);
            float[] autoVec = (resolved != null && resolved is not OpenAiEmbeddingModel)
                ? GenerateSyncEmbedding(resolved, queryText)
                : BridgeSync(() => ComputeAutoEmbeddingAsync(queryText, CancellationToken.None));

            return QueryCore(autoVec, queryEntities, currentTimestampUs, maxTokens, traceList);
        }

        return QueryCore(queryEmbedding, queryEntities, currentTimestampUs, maxTokens, traceList);
    }

    public async Task<MemoryQueryResult> QueryAsync(
        string queryText,
        ReadOnlyMemory<float> queryEmbedding = default,
        IReadOnlyList<string>? queryEntities = null,
        long? currentTimestampUs = null,
        int? maxTokens = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        cancellationToken.ThrowIfCancellationRequested();

        if (!queryEmbedding.IsEmpty && !TwoTierVectorFilter.AreAllFinite(queryEmbedding.Span))
        {
            throw new ArgumentException("Query embedding contains non-finite (NaN or Infinity) values.", nameof(queryEmbedding));
        }

        if (_traces.IsEmpty)
        {
            return new MemoryQueryResult();
        }

        var traceList = _traces.Values.ToList();
        if (traceList.Count == 0)
        {
            return new MemoryQueryResult();
        }

        if (queryEmbedding.IsEmpty &&
            _options.AutoEmbedMissingVectors &&
            !string.IsNullOrWhiteSpace(queryText) &&
            HasAnyEmbeddedTrace(traceList))
        {
            float[] autoVec = await ComputeAutoEmbeddingAsync(queryText, cancellationToken).ConfigureAwait(false);
            return QueryCore(autoVec, queryEntities, currentTimestampUs, maxTokens, traceList);
        }

        return QueryCore(queryEmbedding.Span, queryEntities, currentTimestampUs, maxTokens, traceList);
    }

    private MemoryQueryResult QueryCore(
        ReadOnlySpan<float> queryEmbedding,
        IReadOnlyList<string>? queryEntities,
        long? currentTimestampUs,
        int? maxTokens,
        List<EpisodicTrace> traceList)
    {
        long nowUs = currentTimestampUs ?? DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1000L;
        int budget = maxTokens ?? _options.MaxPromptTokens;
        int n = traceList.Count;

        float[]? rentedCosines = n > QueryStackAllocLimit ? ArrayPool<float>.Shared.Rent(n) : null;
        double[]? rentedPprTrace = n > QueryStackAllocLimit ? ArrayPool<double>.Shared.Rent(n) : null;
        double[]? rentedUtilities = n > QueryStackAllocLimit ? ArrayPool<double>.Shared.Rent(n) : null;
        int[]? rentedCosts = n > QueryStackAllocLimit ? ArrayPool<int>.Shared.Rent(n) : null;
        int[]? rentedSelected = n > QueryStackAllocLimit ? ArrayPool<int>.Shared.Rent(n) : null;

        Span<float> vecCosines = n <= QueryStackAllocLimit ? stackalloc float[n] : rentedCosines.AsSpan(0, n);
        Span<double> pprTraceScores = n <= QueryStackAllocLimit ? stackalloc double[n] : rentedPprTrace.AsSpan(0, n);
        Span<double> utilities = n <= QueryStackAllocLimit ? stackalloc double[n] : rentedUtilities.AsSpan(0, n);
        Span<int> tokenCosts = n <= QueryStackAllocLimit ? stackalloc int[n] : rentedCosts.AsSpan(0, n);
        Span<int> selectedIndices = n <= QueryStackAllocLimit ? stackalloc int[n] : rentedSelected.AsSpan(0, n);

        try
        {
            // 1. Two-tier Vector Recall (FWHT 1-bit coarse filter + FP32 cosine re-ranking, non-zeroing outside coarseLimit)
            if (queryEmbedding.Length > 0)
            {
                TwoTierVectorFilter.ScoreAllTraces(
                    queryEmbedding,
                    traceList,
                    _options.CoarseCandidateLimit,
                    vecCosines);
            }
            else
            {
                vecCosines.Fill(float.NaN);
            }

            // 2. Cached CSR HippoRAG Spreading Activation (zero per-query Dictionary/HashSet rebuild)
            pprTraceScores.Clear();
            if (queryEntities != null && queryEntities.Count > 0)
            {
                EntityCsrGraph graph = GetOrBuildEntityGraph();
                if (graph.NodeCount > 0)
                {
                    int pprBufLen = graph.NodeCount + 1;
                    float[]? rentedPpr = pprBufLen > QueryStackAllocLimit ? ArrayPool<float>.Shared.Rent(pprBufLen) : null;
                    int[]? rentedStamp = pprBufLen > QueryStackAllocLimit ? ArrayPool<int>.Shared.Rent(pprBufLen) : null;

                    Span<float> pprScores = pprBufLen <= QueryStackAllocLimit
                        ? stackalloc float[pprBufLen]
                        : rentedPpr.AsSpan(0, pprBufLen);
                    Span<int> seenStamp = pprBufLen <= QueryStackAllocLimit
                        ? stackalloc int[pprBufLen]
                        : rentedStamp.AsSpan(0, pprBufLen);

                    try
                    {
                        if (HippoRagSpreadingActivation.ComputePersonalizedPageRank(graph, queryEntities, pprScores))
                        {
                            seenStamp.Clear();
                            for (int i = 0; i < n; i++)
                            {
                                List<string>? ents = traceList[i].Entities;
                                if (ents == null || ents.Count == 0) continue;

                                if (ents.Count == 1)
                                {
                                    string only = ents[0];
                                    if (!string.IsNullOrWhiteSpace(only) && graph.EntityToNodeId.TryGetValue(only, out int nodeId))
                                    {
                                        pprTraceScores[i] = pprScores[nodeId];
                                    }
                                    continue;
                                }

                                int stamp = i + 1;
                                double sum = 0.0;
                                for (int e = 0; e < ents.Count; e++)
                                {
                                    string entity = ents[e];
                                    if (!string.IsNullOrWhiteSpace(entity) &&
                                        graph.EntityToNodeId.TryGetValue(entity, out int nodeId) &&
                                        seenStamp[nodeId] != stamp)
                                    {
                                        seenStamp[nodeId] = stamp;
                                        sum += pprScores[nodeId];
                                    }
                                }
                                pprTraceScores[i] = sum;
                            }
                        }
                    }
                    finally
                    {
                        if (rentedStamp != null) ArrayPool<int>.Shared.Return(rentedStamp);
                        if (rentedPpr != null) ArrayPool<float>.Shared.Return(rentedPpr);
                    }
                }
            }

            // 3. Score all candidates using composite activation and compute formatting-aware knapsack costs
            bool allExplicit = true;
            for (int i = 0; i < n; i++)
            {
                EpisodicTrace trace = traceList[i];

                // A) Power-law temporal decay recency score
                double recency = PowerLawDecayCalculator.ComputeRecencyScore(
                    trace,
                    nowUs,
                    _options.DecayParameterD,
                    _options.TimeScaleTauSeconds);

                // B) Vector similarity score (0.0 for unembedded/mismatched traces; Hamming estimate for non-top-coarse valid traces)
                float vecSim = vecCosines[i];
                double normVec = float.IsNaN(vecSim) ? 0.0 : (vecSim + 1.0) * 0.5;

                // C) HippoRAG graph score
                double graphScore = pprTraceScores[i];

                // D) Base importance
                double importance = Math.Clamp(trace.BaseImportance, 0.1f, 5.0f);

                // Composite Activation: A_i(t)
                double activation =
                    (_options.WeightDecay * recency) +
                    (_options.WeightVector * normVec) +
                    (_options.WeightGraph * graphScore) +
                    (_options.WeightImportance * (importance / 5.0));

                // Utility u_i = exp(A_i(t))
                utilities[i] = Math.Exp(activation);

                if (trace.HasExplicitTokenCount)
                {
                    tokenCosts[i] = Math.Max(1, trace.TokenCount);
                }
                else
                {
                    allExplicit = false;
                    int overhead = trace.FormattedOverheadTokens > 0
                        ? trace.FormattedOverheadTokens
                        : BreadcrumbTreeFolder.EstimateTraceOverheadTokens(trace, _tokenEstimator);
                    tokenCosts[i] = Math.Max(1, trace.TokenCount + overhead);
                }
            }

            // 4. 0/1 Knapsack Prompt Budget Optimization (selects budget-optimal subset, then sorts in-place by Utility descending)
            int headerTokens = allExplicit ? 0 : BreadcrumbTreeFolder.EstimateHeaderTokens(_tokenEstimator);
            int packingBudget = allExplicit ? budget : Math.Max(0, budget - headerTokens);

            KnapsackPromptPacker.PackTraceIndices(
                traceList,
                utilities,
                tokenCosts,
                packingBudget,
                selectedIndices,
                out int selectedCount,
                out _);

            if (selectedCount > 1)
            {
                SortIndicesByUtilityDescending(selectedIndices.Slice(0, selectedCount), utilities);
            }

            var selectedTraces = new List<EpisodicTrace>(selectedCount);
            int totalTokens = 0;
            double totalUtility = 0.0;
            for (int k = 0; k < selectedCount; k++)
            {
                int idx = selectedIndices[k];
                EpisodicTrace t = traceList[idx];
                selectedTraces.Add(t);
                t.RecordAccess(nowUs);
                totalTokens += t.TokenCount;
                totalUtility += utilities[idx];
            }

            string formattedContext = BreadcrumbTreeFolder.FormatTraceContext(
                selectedTraces,
                budget,
                _tokenEstimator,
                _options.FoldDeepSubtrees);

            return new MemoryQueryResult
            {
                SelectedTraces = selectedTraces,
                TotalTokens = totalTokens,
                TotalUtility = totalUtility,
                FormattedContext = formattedContext
            };
        }
        finally
        {
            if (rentedSelected != null) ArrayPool<int>.Shared.Return(rentedSelected);
            if (rentedCosts != null) ArrayPool<int>.Shared.Return(rentedCosts);
            if (rentedUtilities != null) ArrayPool<double>.Shared.Return(rentedUtilities);
            if (rentedPprTrace != null) ArrayPool<double>.Shared.Return(rentedPprTrace);
            if (rentedCosines != null) ArrayPool<float>.Shared.Return(rentedCosines);
        }
    }

    private void EnsureTokenCount(EpisodicTrace trace)
    {
        if (trace.TokenCount <= 0 && !string.IsNullOrWhiteSpace(trace.Content))
        {
            trace.SetTokenCount(Math.Max(1, _tokenEstimator.EstimateTokens(trace.Content.Trim())));
        }

        trace.FormattedOverheadTokens = BreadcrumbTreeFolder.EstimateTraceOverheadTokens(trace, _tokenEstimator);
    }

    private void StoreTraceWithTopologyUpdate(EpisodicTrace trace)
    {
        bool topologyChanged = HasEntities(trace);
        _traces.AddOrUpdate(
            trace.Id,
            trace,
            (_, existing) =>
            {
                if (HasEntities(existing))
                {
                    topologyChanged = true;
                }
                return trace;
            });

        if (topologyChanged)
        {
            Interlocked.Increment(ref _topologyVersion);
        }
    }

    private static bool HasEntities(EpisodicTrace? trace) =>
        trace?.Entities is { Count: > 0 };

    private EntityCsrGraph GetOrBuildEntityGraph()
    {
        long currentVersion = Volatile.Read(ref _topologyVersion);
        EntityCsrGraph? cached = Volatile.Read(ref _cachedEntityGraph);
        if (cached != null && cached.Version == currentVersion)
        {
            return cached;
        }

        lock (_graphBuildLock)
        {
            for (int attempt = 0; attempt < 2; attempt++)
            {
                long vBefore = Volatile.Read(ref _topologyVersion);
                cached = _cachedEntityGraph;
                if (cached != null && cached.Version == vBefore)
                {
                    return cached;
                }

                List<EpisodicTrace> snapshot = _traces.Values.ToList();
                EntityCsrGraph built = EntityCsrGraph.Build(snapshot, vBefore);
                long vAfter = Volatile.Read(ref _topologyVersion);
                if (vAfter == vBefore)
                {
                    Volatile.Write(ref _cachedEntityGraph, built);
                    return built;
                }

                if (attempt == 1)
                {
                    return built;
                }
            }
        }

        return EntityCsrGraph.Empty;
    }

    private static void SortIndicesByUtilityDescending(Span<int> indices, ReadOnlySpan<double> utilities)
    {
        for (int i = 1; i < indices.Length; i++)
        {
            int keyIdx = indices[i];
            double keyUtil = utilities[keyIdx];
            int j = i - 1;
            while (j >= 0 && utilities[indices[j]] < keyUtil)
            {
                indices[j + 1] = indices[j];
                j--;
            }
            indices[j + 1] = keyIdx;
        }
    }

    private static void QuantizeTraceIfNeeded(EpisodicTrace trace)
    {
        float[]? embedding = trace.Embedding;
        if (embedding != null && embedding.Length > 0 && trace.BinaryEmbedding == null && TwoTierVectorFilter.AreAllFinite(embedding))
        {
            trace.BinaryEmbedding = TwoTierVectorFilter.QuantizeTo1Bit(embedding);
        }
    }

    private static bool HasAnyEmbeddedTrace(List<EpisodicTrace> traces)
    {
        for (int i = 0; i < traces.Count; i++)
        {
            if (traces[i].Embedding is { Length: > 0 })
                return true;
        }
        return false;
    }

    private async Task<float[]> ComputeAutoEmbeddingAsync(string text, CancellationToken cancellationToken)
    {
        IEmbeddingModel model = await ResolveEmbeddingModelAsync(cancellationToken).ConfigureAwait(false);
        if (model is OpenAiEmbeddingModel openAi)
        {
            using CancellationTokenSource? timeoutCts = CreateLinkedTimeoutCts(_options.EmbeddingTimeout, cancellationToken);
            CancellationToken effectiveToken = timeoutCts?.Token ?? cancellationToken;
            try
            {
                float[] vec = await openAi.GenerateEmbeddingAsync(text, effectiveToken).ConfigureAwait(false);
                if (vec.Length != openAi.Dimensions || !TwoTierVectorFilter.AreAllFinite(vec))
                {
                    throw new InvalidOperationException(
                        $"Embedding vector returned by '{openAi.BaseUrl}' had invalid length or non-finite values.");
                }
                return vec;
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                throw;
            }
            catch (Exception ex) when (IsRecoverableEmbeddingException(ex))
            {
                return await FallbackAfterNpuFailureAsync(openAi, text, cancellationToken).ConfigureAwait(false);
            }
        }

        return GenerateSyncEmbedding(model, text);
    }

    private async Task<float[]> FallbackAfterNpuFailureAsync(
        OpenAiEmbeddingModel failedModel,
        string text,
        CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        int lockedDims = Volatile.Read(ref _resolvedDimensions);

        if (_traces.IsEmpty)
        {
            try
            {
                await _initGate.WaitAsync(cancellationToken).ConfigureAwait(false);
            }
            catch (ObjectDisposedException)
            {
                throw new ObjectDisposedException(nameof(MemExEngine));
            }
            try
            {
                ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
                if (_traces.IsEmpty && ReferenceEquals(Volatile.Read(ref _resolvedModel), failedModel))
                {
                    int demoteDims = (lockedDims > 0 && lockedDims % 8 == 0)
                        ? lockedDims
                        : _options.FallbackHashDimensions;
                    var demoted = new FastHashEmbeddingModel(demoteDims);
                    if (_ownsResolvedModel)
                    {
                        lock (_fallbackLock)
                        {
                            _retiredModels.Add(failedModel);
                        }
                    }

                    _ownsResolvedModel = true;
                    Volatile.Write(ref _resolvedDimensions, demoted.Dimensions);
                    Volatile.Write(ref _activeEndpoint, null);
                    Volatile.Write(ref _activeModelId, "fast-hash");
                    Volatile.Write(ref _resolvedModel, demoted);
                    return GenerateSyncEmbedding(demoted, text);
                }
            }
            finally
            {
                try
                {
                    _initGate.Release();
                }
                catch (ObjectDisposedException)
                {
                }
            }
        }

        int targetDims = lockedDims > 0 ? lockedDims : _options.FallbackHashDimensions;
        return ComputeMatchedFallbackVector(targetDims, text);
    }

    private float[] ComputeMatchedFallbackVector(int targetDims, string text)
    {
        if (targetDims % 8 == 0)
        {
            FastHashEmbeddingModel exactModel = GetOrCreateFallbackHashModel(targetDims);
            return GenerateSyncEmbedding(exactModel, text);
        }

        int baseDims = Math.Max(((targetDims + 7) / 8) * 8, _options.FallbackHashDimensions);
        FastHashEmbeddingModel baseModel = GetOrCreateFallbackHashModel(baseDims);
        float[] raw = GenerateSyncEmbedding(baseModel, text);
        var projected = new float[targetDims];
        for (int i = 0; i < raw.Length; i++)
        {
            projected[i % targetDims] += raw[i];
        }

        double sumSq = 0.0;
        for (int i = 0; i < projected.Length; i++)
        {
            double v = projected[i];
            sumSq += v * v;
        }

        if (sumSq > 1e-12)
        {
            float invNorm = (float)(1.0 / Math.Sqrt(sumSq));
            for (int i = 0; i < projected.Length; i++)
            {
                projected[i] *= invNorm;
            }
        }

        return projected;
    }

    private FastHashEmbeddingModel GetOrCreateFallbackHashModel(int dimensions)
    {
        lock (_fallbackLock)
        {
            if (_fallbackHashModel != null && _fallbackHashModel.Dimensions == dimensions)
            {
                return _fallbackHashModel;
            }

            _fallbackHashModel?.Dispose();
            _fallbackHashModel = new FastHashEmbeddingModel(dimensions);
            return _fallbackHashModel;
        }
    }

    private async Task<IEmbeddingModel> ResolveEmbeddingModelAsync(CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        IEmbeddingModel? existing = Volatile.Read(ref _resolvedModel);
        if (existing != null)
            return existing;

        try
        {
            await _initGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (ObjectDisposedException)
        {
            throw new ObjectDisposedException(nameof(MemExEngine));
        }

        try
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            existing = Volatile.Read(ref _resolvedModel);
            if (existing != null)
                return existing;

            if (_options.EmbeddingModel != null)
            {
                _ownsResolvedModel = false;
                Volatile.Write(ref _resolvedDimensions, _options.EmbeddingModel.Dimensions);
                Volatile.Write(ref _activeEndpoint, (_options.EmbeddingModel as OpenAiEmbeddingModel)?.BaseUrl);
                Volatile.Write(ref _activeModelId, (_options.EmbeddingModel as OpenAiEmbeddingModel)?.ModelId ?? _options.EmbeddingModel.GetType().Name);
                Volatile.Write(ref _resolvedModel, _options.EmbeddingModel);
                return _options.EmbeddingModel;
            }

            string? rawEndpoint = _options.EmbeddingEndpoint?.Trim();
            bool isAutoEndpoint = string.IsNullOrWhiteSpace(rawEndpoint) ||
                string.Equals(rawEndpoint, "auto-npu", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(rawEndpoint, "npu", StringComparison.OrdinalIgnoreCase);

            CancellationTokenSource linkedCts;
            try
            {
                linkedCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, _disposeCts.Token);
            }
            catch (ObjectDisposedException)
            {
                throw new ObjectDisposedException(nameof(MemExEngine));
            }

            OpenAiEmbeddingModel? resolvedOpenAi = null;
            using (linkedCts)
            {
                try
                {
                    if (!isAutoEndpoint)
                    {
                        resolvedOpenAi = await TryCreateExplicitOpenAiModelAsync(rawEndpoint!, linkedCts.Token).ConfigureAwait(false);
                    }
                    else if (_options.EnableNpuEmbeddings)
                    {
                        resolvedOpenAi = await TryDiscoverOpenAiModelAsync(linkedCts.Token).ConfigureAwait(false);
                    }
                }
                catch (OperationCanceledException) when (Volatile.Read(ref _disposed) != 0 && !cancellationToken.IsCancellationRequested)
                {
                    throw new ObjectDisposedException(nameof(MemExEngine));
                }
            }

            if (Volatile.Read(ref _disposed) != 0)
            {
                resolvedOpenAi?.Dispose();
                throw new ObjectDisposedException(nameof(MemExEngine));
            }

            if (resolvedOpenAi != null)
            {
                _ownsResolvedModel = true;
                Volatile.Write(ref _resolvedDimensions, resolvedOpenAi.Dimensions);
                Volatile.Write(ref _activeEndpoint, resolvedOpenAi.BaseUrl);
                Volatile.Write(ref _activeModelId, resolvedOpenAi.ModelId);
                Volatile.Write(ref _resolvedModel, resolvedOpenAi);
                return resolvedOpenAi;
            }

            var fallback = new FastHashEmbeddingModel(_options.FallbackHashDimensions);
            _ownsResolvedModel = true;
            Volatile.Write(ref _resolvedDimensions, fallback.Dimensions);
            Volatile.Write(ref _activeEndpoint, null);
            Volatile.Write(ref _activeModelId, "fast-hash");
            Volatile.Write(ref _resolvedModel, fallback);
            return fallback;
        }
        finally
        {
            try
            {
                _initGate.Release();
            }
            catch (ObjectDisposedException)
            {
            }
        }
    }

    private async Task<OpenAiEmbeddingModel?> TryCreateExplicitOpenAiModelAsync(
        string rawEndpoint,
        CancellationToken cancellationToken)
    {
        if (!IsValidHttpEndpoint(rawEndpoint, _options.HttpClient))
        {
            return null;
        }

        try
        {
            string? modelId = string.IsNullOrWhiteSpace(_options.EmbeddingModelId)
                ? null
                : _options.EmbeddingModelId.Trim();

            if (modelId == null)
            {
                NpuSidecarEndpoint? sidecar = await NpuSidecarDiscovery.ProbeAsync(
                    httpClient: _options.HttpClient,
                    candidates: [(rawEndpoint, NpuSidecarRuntime.CustomOpenAi)],
                    perEndpointTimeout: _options.ProbeTimeout,
                    cancellationToken: cancellationToken).ConfigureAwait(false);

                modelId = SelectEmbeddingModelId(sidecar?.Models) ?? DefaultEmbeddingModelId;
            }

            int? knownDims = _options.EmbeddingDimensions > 0
                ? _options.EmbeddingDimensions
                : (_options.ProbeTimeout.HasValue
                    ? await OpenAiEmbeddingModel.ProbeDimensionsAsync(
                        rawEndpoint,
                        modelId,
                        _options.EmbeddingApiKey,
                        _options.HttpClient,
                        _options.ProbeTimeout,
                        cancellationToken).ConfigureAwait(false)
                    : null);

            if (knownDims is <= 0)
                return null;

            return await OpenAiEmbeddingModel.CreateAutoAsync(
                rawEndpoint,
                modelId,
                _options.EmbeddingApiKey,
                _options.HttpClient,
                _options.EmbeddingTimeout,
                knownDimensions: knownDims,
                cancellationToken: cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex) when (IsRecoverableEmbeddingException(ex))
        {
            return null;
        }
    }

    private async Task<OpenAiEmbeddingModel?> TryDiscoverOpenAiModelAsync(CancellationToken cancellationToken)
    {
        try
        {
            NpuSidecarEndpoint? sidecar = await NpuSidecarDiscovery.ProbeCachedAsync(
                cacheTtl: _options.ProbeCacheTtl,
                httpClient: _options.HttpClient,
                perEndpointTimeout: _options.ProbeTimeout,
                cancellationToken: cancellationToken).ConfigureAwait(false);

            if (sidecar == null)
                return null;

            string modelId = !string.IsNullOrWhiteSpace(_options.EmbeddingModelId)
                ? _options.EmbeddingModelId.Trim()
                : SelectEmbeddingModelId(sidecar.Models) ?? DefaultEmbeddingModelId;

            int? knownDims = _options.EmbeddingDimensions > 0
                ? _options.EmbeddingDimensions
                : (_options.ProbeTimeout.HasValue
                    ? await OpenAiEmbeddingModel.ProbeDimensionsAsync(
                        sidecar.BaseUrl,
                        modelId,
                        _options.EmbeddingApiKey,
                        _options.HttpClient,
                        _options.ProbeTimeout,
                        cancellationToken).ConfigureAwait(false)
                    : null);

            if (knownDims is <= 0)
                return null;

            return await OpenAiEmbeddingModel.CreateAutoAsync(
                sidecar.BaseUrl,
                modelId,
                _options.EmbeddingApiKey,
                _options.HttpClient,
                _options.EmbeddingTimeout,
                knownDimensions: knownDims,
                cancellationToken: cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex) when (IsRecoverableEmbeddingException(ex))
        {
            return null;
        }
    }

    private static bool IsValidHttpEndpoint(string endpoint, HttpClient? httpClient)
    {
        if (Uri.TryCreate(endpoint, UriKind.Absolute, out Uri? uri))
        {
            return string.Equals(uri.Scheme, Uri.UriSchemeHttp, StringComparison.OrdinalIgnoreCase) ||
                   string.Equals(uri.Scheme, Uri.UriSchemeHttps, StringComparison.OrdinalIgnoreCase);
        }

        return httpClient?.BaseAddress != null && Uri.TryCreate(endpoint, UriKind.Relative, out _);
    }

    private static string? SelectEmbeddingModelId(IReadOnlyList<string>? models)
    {
        if (models == null || models.Count == 0)
            return null;

        for (int i = 0; i < models.Count; i++)
        {
            string? m = models[i];
            if (!string.IsNullOrWhiteSpace(m) &&
                m.Contains("embed", StringComparison.OrdinalIgnoreCase))
            {
                return m.Trim();
            }
        }

        for (int i = 0; i < models.Count; i++)
        {
            string? m = models[i];
            if (!string.IsNullOrWhiteSpace(m))
                return m.Trim();
        }

        return null;
    }

    private static bool IsRecoverableEmbeddingException(Exception ex) =>
        ex is HttpRequestException
            or TaskCanceledException
            or OperationCanceledException
            or JsonException
            or InvalidOperationException
            or UriFormatException
            or NotSupportedException
            or ArgumentException;

    private static CancellationTokenSource? CreateLinkedTimeoutCts(TimeSpan? timeout, CancellationToken cancellationToken)
    {
        if (!timeout.HasValue || timeout.Value == Timeout.InfiniteTimeSpan)
            return null;

        var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        cts.CancelAfter(timeout.Value);
        return cts;
    }

    private static float[] GenerateSyncEmbedding(IEmbeddingModel model, string text)
    {
        var vector = new float[model.Dimensions];
        model.GenerateEmbedding(text.AsSpan(), vector);
        return vector;
    }

    private static T BridgeSync<T>(Func<Task<T>> asyncAction)
    {
        if (SynchronizationContext.Current is not null || TaskScheduler.Current != TaskScheduler.Default)
        {
            return Task.Run(asyncAction).GetAwaiter().GetResult();
        }

        return asyncAction().GetAwaiter().GetResult();
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        try { _disposeCts.Cancel(); } catch (ObjectDisposedException) { }

        try
        {
            if (_initGate.Wait(0))
            {
                _initGate.Release();
                _initGate.Dispose();
            }
        }
        catch (ObjectDisposedException)
        {
        }

        if (_ownsResolvedModel)
        {
            Interlocked.Exchange(ref _resolvedModel, null)?.Dispose();
        }

        lock (_fallbackLock)
        {
            for (int i = 0; i < _retiredModels.Count; i++)
            {
                _retiredModels[i].Dispose();
            }
            _retiredModels.Clear();

            _fallbackHashModel?.Dispose();
            _fallbackHashModel = null;
        }

        _disposeCts.Dispose();
    }

    /// <summary>
    /// Exports top-scoring episodic traces for offline Sleep Consolidation (dreaming / GKD distillation),
    /// prioritizing traces with high ACT-R recency, base importance, and repeated recall.
    /// </summary>
    public IReadOnlyList<EpisodicTrace> ExportConsolidationCandidates(
        int maxCount = 64,
        int minAccessCount = 0,
        long? currentTimestampUs = null)
    {
        if (maxCount <= 0) throw new ArgumentOutOfRangeException(nameof(maxCount));
        if (minAccessCount < 0) throw new ArgumentOutOfRangeException(nameof(minAccessCount));

        long nowUs = currentTimestampUs ?? DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1000L;
        (EpisodicTrace Item, double Score)[] heapBuf = ArrayPool<(EpisodicTrace Item, double Score)>.Shared.Rent(maxCount);

        try
        {
            var topHeap = new BoundedMinItemHeap<EpisodicTrace>(heapBuf, maxCount);

            foreach (var trace in _traces.Values)
            {
                if (string.IsNullOrWhiteSpace(trace.Content) || trace.AccessCount < minAccessCount)
                    continue;

                double recency = PowerLawDecayCalculator.ComputeRecencyScore(
                    trace,
                    nowUs,
                    _options.DecayParameterD,
                    _options.TimeScaleTauSeconds);

                double importance = Math.Clamp(trace.BaseImportance, 0.1f, 5.0f) / 5.0;
                double score = (_options.WeightDecay * recency) + (_options.WeightImportance * importance);
                topHeap.Offer(trace, score);
            }

            int resultCount = topHeap.Count;
            if (resultCount == 0)
                return [];

            var result = new EpisodicTrace[resultCount];
            for (int i = resultCount - 1; i >= 0; i--)
            {
                topHeap.TryDequeue(out EpisodicTrace item, out _);
                result[i] = item;
            }
            return result;
        }
        finally
        {
            ArrayPool<(EpisodicTrace Item, double Score)>.Shared.Return(heapBuf, clearArray: true);
        }
    }

    /// <summary>
    /// Synaptic Pruning: removes consolidated trace IDs from volatile fast memory.
    /// </summary>
    public int PruneTraces(IEnumerable<string> traceIds)
    {
        ArgumentNullException.ThrowIfNull(traceIds);
        int removed = 0;
        bool topologyChanged = false;
        foreach (string id in traceIds)
        {
            if (!string.IsNullOrEmpty(id) && _traces.TryRemove(id, out EpisodicTrace? removedTrace))
            {
                removed++;
                if (HasEntities(removedTrace))
                {
                    topologyChanged = true;
                }
            }
        }

        if (topologyChanged)
        {
            Interlocked.Increment(ref _topologyVersion);
        }

        return removed;
    }

    /// <summary>
    /// Synaptic Pruning: removes stale traces whose ACT-R recency score has decayed below <paramref name="minRecencyScore"/>.
    /// </summary>
    public int PruneDecayedTraces(double minRecencyScore = 0.05, long? currentTimestampUs = null)
    {
        long nowUs = currentTimestampUs ?? DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1000L;
        int removed = 0;
        bool topologyChanged = false;

        foreach (var kvp in _traces)
        {
            double recency = PowerLawDecayCalculator.ComputeRecencyScore(
                kvp.Value,
                nowUs,
                _options.DecayParameterD,
                _options.TimeScaleTauSeconds);

            if (recency < minRecencyScore && _traces.TryRemove(kvp.Key, out EpisodicTrace? removedTrace))
            {
                removed++;
                if (HasEntities(removedTrace))
                {
                    topologyChanged = true;
                }
            }
        }

        if (topologyChanged)
        {
            Interlocked.Increment(ref _topologyVersion);
        }

        return removed;
    }
}
