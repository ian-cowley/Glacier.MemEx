namespace Glacier.MemEx;

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Inference.Hardware;
using Glacier.MemEx.Budgeting;
using Glacier.MemEx.Decay;
using Glacier.MemEx.Models;
using Glacier.MemEx.Recall;
using Glacier.Rag.Embeddings;

public sealed class MemExEngine : IDisposable
{
    private const string DefaultEmbeddingModelId = "embeddinggemma";

    private readonly ConcurrentDictionary<string, EpisodicTrace> _traces = new(StringComparer.Ordinal);
    private readonly MemExOptions _options;
    private readonly SemaphoreSlim _initGate = new(1, 1);
    private readonly CancellationTokenSource _disposeCts = new();
    private readonly object _fallbackLock = new();
    private readonly List<IDisposable> _retiredModels = [];

    private IEmbeddingModel? _resolvedModel;
    private FastHashEmbeddingModel? _fallbackHashModel;
    private int _resolvedDimensions;
    private bool _ownsResolvedModel;
    private string? _activeEndpoint;
    private string? _activeModelId;
    private int _disposed;

    public MemExEngine(MemExOptions? options = null)
    {
        _options = options ?? new MemExOptions();

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
        _traces[trace.Id] = trace;
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
        _traces[trace.Id] = trace;
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

        // 1. Two-tier Vector Recall (1-bit coarse filter + FP32 cosine re-ranking)
        var vectorMatches = new Dictionary<string, float>(StringComparer.Ordinal);
        if (queryEmbedding.Length > 0)
        {
            var reranked = TwoTierVectorFilter.FilterAndRerank(
                queryEmbedding,
                traceList,
                coarseLimit: _options.CoarseCandidateLimit,
                finalLimit: traceList.Count);

            foreach (var (t, cos) in reranked)
            {
                vectorMatches[t.Id] = cos;
            }
        }

        // 2. HippoRAG Spreading Activation
        var pprScores = (queryEntities != null && queryEntities.Count > 0)
            ? HippoRagSpreadingActivation.ComputePersonalizedPageRank(queryEntities, traceList)
            : [];

        // 3. Score all candidates using composite activation
        var scoredCandidates = new List<(EpisodicTrace Trace, double Utility)>(traceList.Count);

        foreach (var trace in traceList)
        {
            // A) Power-law temporal decay recency score
            double recency = PowerLawDecayCalculator.ComputeRecencyScore(
                trace,
                nowUs,
                _options.DecayParameterD,
                _options.TimeScaleTauSeconds);

            // B) Vector similarity score
            vectorMatches.TryGetValue(trace.Id, out float vecSim);
            double normVec = (vecSim + 1.0) * 0.5; // Map [-1, 1] to [0, 1]

            // C) HippoRAG graph score
            double graphScore = HippoRagSpreadingActivation.ScoreTrace(trace, pprScores);

            // D) Base importance
            double importance = Math.Clamp(trace.BaseImportance, 0.1f, 5.0f);

            // Composite Activation: A_i(t)
            double activation =
                (_options.WeightDecay * recency) +
                (_options.WeightVector * normVec) +
                (_options.WeightGraph * graphScore) +
                (_options.WeightImportance * (importance / 5.0));

            // Utility u_i = exp(A_i(t))
            double utility = Math.Exp(activation);
            scoredCandidates.Add((trace, utility));
        }

        // 4. 0/1 Knapsack Prompt Budget Optimization (selects budget-optimal subset, then orders by Utility descending)
        var selectedTraces = KnapsackPromptPacker.PackTraces(scoredCandidates, budget);
        if (selectedTraces.Count > 1)
        {
            var utilityById = new Dictionary<string, double>(scoredCandidates.Count, StringComparer.Ordinal);
            foreach (var (trace, utility) in scoredCandidates)
            {
                utilityById[trace.Id] = utility;
            }

            selectedTraces = selectedTraces
                .OrderByDescending(t => utilityById.GetValueOrDefault(t.Id))
                .ToList();
        }

        // Record retrieval access events
        foreach (var t in selectedTraces)
        {
            t.RecordAccess(nowUs);
        }

        int totalTokens = selectedTraces.Sum(t => t.TokenCount);
        double totalUtility = scoredCandidates
            .Where(c => selectedTraces.Contains(c.Trace))
            .Sum(c => c.Utility);

        string formattedContext = BreadcrumbTreeFolder.FormatTraceContext(selectedTraces);

        return new MemoryQueryResult
        {
            SelectedTraces = selectedTraces,
            TotalTokens = totalTokens,
            TotalUtility = totalUtility,
            FormattedContext = formattedContext
        };
    }

    private static void EnsureTokenCount(EpisodicTrace trace)
    {
        if (trace.TokenCount <= 0 && !string.IsNullOrWhiteSpace(trace.Content))
        {
            // Estimate tokens: ~4 chars per token (zero reflection for NativeAOT)
            trace.SetTokenCount(Math.Max(1, trace.Content.Length / 4));
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

        if (_initGate.Wait(0))
        {
            _initGate.Release();
            _initGate.Dispose();
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
        var candidates = new List<(EpisodicTrace Trace, double Score)>();

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
            candidates.Add((trace, score));
        }

        return candidates
            .OrderByDescending(c => c.Score)
            .Take(maxCount)
            .Select(c => c.Trace)
            .ToArray();
    }

    /// <summary>
    /// Synaptic Pruning: removes consolidated trace IDs from volatile fast memory.
    /// </summary>
    public int PruneTraces(IEnumerable<string> traceIds)
    {
        ArgumentNullException.ThrowIfNull(traceIds);
        int removed = 0;
        foreach (string id in traceIds)
        {
            if (!string.IsNullOrEmpty(id) && _traces.TryRemove(id, out _))
            {
                removed++;
            }
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

        foreach (var kvp in _traces)
        {
            double recency = PowerLawDecayCalculator.ComputeRecencyScore(
                kvp.Value,
                nowUs,
                _options.DecayParameterD,
                _options.TimeScaleTauSeconds);

            if (recency < minRecencyScore && _traces.TryRemove(kvp.Key, out _))
            {
                removed++;
            }
        }

        return removed;
    }
}
