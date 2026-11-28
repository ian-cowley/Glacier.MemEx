namespace Glacier.MemEx.Recall;

using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using Glacier.MemEx.Models;
using Glacier.Vector.Compute;

public static class TwoTierVectorFilter
{
    private const int StackAllocFloatLimit = 1024;
    private const int StackAllocUlongLimit = 64;
    private const ulong DeterministicSignSeed = 0x9E3779B97F4A7C15UL;

    private static readonly ConcurrentDictionary<int, float[]> s_rademacherSignsCache = new();

    internal static bool AreAllFinite(ReadOnlySpan<float> values) => AreAllFinite(values, out _);

    private static bool AreAllFinite(ReadOnlySpan<float> values, out float maxAbs)
    {
        maxAbs = 0f;
        for (int i = 0; i < values.Length; i++)
        {
            float v = values[i];
            if (!float.IsFinite(v))
                return false;

            float abs = MathF.Abs(v);
            if (abs > maxAbs)
                maxAbs = abs;
        }
        return true;
    }

    private static float[] GetOrCreateSigns(int paddedDim)
    {
        return s_rademacherSignsCache.GetOrAdd(paddedDim, static dim =>
        {
            var signs = new float[dim];
            HadamardProjection.GenerateSigns(signs, DeterministicSignSeed ^ (ulong)dim);
            return signs;
        });
    }

    public static ulong[] QuantizeTo1Bit(ReadOnlySpan<float> embedding)
    {
        if (!AreAllFinite(embedding, out float maxAbs))
            throw new ArgumentException("Embedding contains non-finite (NaN or Infinity) values.", nameof(embedding));

        if (embedding.IsEmpty)
            return [];

        int paddedDim = HadamardProjection.GetPaddedDimension(embedding.Length);
        int words = (paddedDim + 63) >> 6;
        ulong[] bits = new ulong[words];
        QuantizeTo1BitCore(embedding, maxAbs, paddedDim, bits);
        return bits;
    }

    private static void QuantizeTo1BitCore(
        ReadOnlySpan<float> embedding,
        float maxAbs,
        int paddedDim,
        Span<ulong> destination)
    {
        float[] signs = GetOrCreateSigns(paddedDim);

        float[]? rentedFwht = paddedDim > StackAllocFloatLimit ? ArrayPool<float>.Shared.Rent(paddedDim) : null;
        Span<float> fwhtScratch = paddedDim <= StackAllocFloatLimit
            ? stackalloc float[paddedDim]
            : rentedFwht.AsSpan(0, paddedDim);

        float[]? rentedScaled = null;
        bool needsRescale = maxAbs > 1e15f || (maxAbs > 0f && maxAbs < 1e-15f);

        try
        {
            if (needsRescale)
            {
                int len = embedding.Length;
                Span<float> scaled = len <= StackAllocFloatLimit
                    ? stackalloc float[len]
                    : (rentedScaled = ArrayPool<float>.Shared.Rent(len)).AsSpan(0, len);

                double dInvScale = 1.0 / maxAbs;
                for (int i = 0; i < len; i++)
                {
                    scaled[i] = (float)(embedding[i] * dInvScale);
                }

                HadamardProjection.RotateAndBinarize(scaled, signs, destination, fwhtScratch);
            }
            else
            {
                HadamardProjection.RotateAndBinarize(embedding, signs, destination, fwhtScratch);
            }
        }
        finally
        {
            if (rentedScaled != null) ArrayPool<float>.Shared.Return(rentedScaled);
            if (rentedFwht != null) ArrayPool<float>.Shared.Return(rentedFwht);
        }
    }

    /// <summary>
    /// Filters traces using FWHT-rotated 1-bit Hamming distance coarse filter (Tier 1),
    /// followed by exact FP32 cosine similarity re-ranking (Tier 2).
    /// </summary>
    public static List<(EpisodicTrace Trace, float CosineSimilarity)> FilterAndRerank(
        ReadOnlySpan<float> queryEmbedding,
        IReadOnlyList<EpisodicTrace> traces,
        int coarseLimit = 64,
        int finalLimit = 16)
    {
        ArgumentNullException.ThrowIfNull(traces);

        if (!AreAllFinite(queryEmbedding, out float qMaxAbs))
            throw new ArgumentException("Query embedding contains non-finite (NaN or Infinity) values.", nameof(queryEmbedding));

        if (traces.Count == 0 || queryEmbedding.Length == 0 || coarseLimit <= 0 || finalLimit <= 0)
            return [];

        int[] validIndices = ArrayPool<int>.Shared.Rent(traces.Count);
        int validCount = 0;

        try
        {
            // Filter out null, mismatched-dimension, or non-finite candidate embeddings BEFORE Tier 1
            // so invalid candidates never occupy coarseLimit slots with HammingDistance == 0.
            for (int i = 0; i < traces.Count; i++)
            {
                EpisodicTrace? trace = traces[i];
                float[]? emb = trace?.Embedding;
                if (emb == null || emb.Length != queryEmbedding.Length || !AreAllFinite(emb))
                    continue;

                validIndices[validCount++] = i;
            }

            if (validCount == 0)
                return [];

            int topK = Math.Min(finalLimit, validCount);

            if (qMaxAbs <= 0.0f)
            {
                var zeroResults = new List<(EpisodicTrace Trace, float CosineSimilarity)>(topK);
                for (int k = 0; k < topK; k++)
                {
                    zeroResults.Add((traces[validIndices[k]], 0.0f));
                }
                return zeroResults;
            }

            float queryNormSq = DistanceKernels.DotProduct(queryEmbedding, queryEmbedding);
            (int Index, float Score)[] finalBuf = ArrayPool<(int Index, float Score)>.Shared.Rent(topK);

            try
            {
                var finalHeap = new BoundedMinFloatHeap(finalBuf, topK);

                // For dimensions <= 64, exact FP32 cosine is faster and avoids small-dimension quantization distortion.
                // Same when validCount <= coarseLimit.
                if (queryEmbedding.Length <= 64 || validCount <= coarseLimit)
                {
                    for (int k = 0; k < validCount; k++)
                    {
                        int traceIdx = validIndices[k];
                        float cos = ComputeCosineSimilarity(queryEmbedding, traces[traceIdx].Embedding!, queryNormSq);
                        finalHeap.Offer(traceIdx, cos);
                    }
                }
                else
                {
                    int paddedDim = HadamardProjection.GetPaddedDimension(queryEmbedding.Length);
                    int words = (paddedDim + 63) >> 6;

                    ulong[]? rentedQBits = words > StackAllocUlongLimit ? ArrayPool<ulong>.Shared.Rent(words) : null;
                    Span<ulong> queryBits = words <= StackAllocUlongLimit
                        ? stackalloc ulong[words]
                        : rentedQBits.AsSpan(0, words);

                    (int Index, int Dist)[] coarseBuf = ArrayPool<(int Index, int Dist)>.Shared.Rent(coarseLimit);

                    try
                    {
                        QuantizeTo1BitCore(queryEmbedding, qMaxAbs, paddedDim, queryBits);
                        var coarseHeap = new BoundedMaxIntHeap(coarseBuf, coarseLimit);

                        for (int k = 0; k < validCount; k++)
                        {
                            int traceIdx = validIndices[k];
                            EpisodicTrace trace = traces[traceIdx];
                            ulong[]? traceBits = trace.BinaryEmbedding;
                            if (traceBits == null || traceBits.Length != words)
                            {
                                traceBits = QuantizeTo1Bit(trace.Embedding);
                                trace.BinaryEmbedding = traceBits;
                            }

                            int dist = DistanceKernels.HammingDistance(queryBits, traceBits.AsSpan(0, words));
                            coarseHeap.Offer(traceIdx, dist);
                        }

                        var coarseItems = coarseHeap.Buffer;
                        int coarseCount = coarseHeap.Count;
                        for (int k = 0; k < coarseCount; k++)
                        {
                            int traceIdx = coarseItems[k].Index;
                            float cos = ComputeCosineSimilarity(queryEmbedding, traces[traceIdx].Embedding!, queryNormSq);
                            finalHeap.Offer(traceIdx, cos);
                        }
                    }
                    finally
                    {
                        ArrayPool<(int Index, int Dist)>.Shared.Return(coarseBuf);
                        if (rentedQBits != null) ArrayPool<ulong>.Shared.Return(rentedQBits);
                    }
                }

                int resultCount = finalHeap.Count;
                var results = new List<(EpisodicTrace Trace, float CosineSimilarity)>(resultCount);
                for (int i = 0; i < resultCount; i++)
                {
                    results.Add(default);
                }

                for (int i = resultCount - 1; i >= 0; i--)
                {
                    finalHeap.TryDequeue(out int traceIdx, out float cos);
                    results[i] = (traces[traceIdx], cos);
                }

                return results;
            }
            finally
            {
                ArrayPool<(int Index, float Score)>.Shared.Return(finalBuf);
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(validIndices);
        }
    }

    /// <summary>
    /// Scores all traces without zeroing non-top-<paramref name="coarseLimit"/> valid candidates:
    /// top-<paramref name="coarseLimit"/> candidates receive exact FP32 cosine similarity, remaining
    /// valid-dimension embedded traces receive <see cref="HadamardProjection.EstimateCosineFromHamming(int, int)"/>,
    /// and unembedded/mismatched/non-finite traces receive <see cref="float.NaN"/>.
    /// </summary>
    internal static void ScoreAllTraces(
        ReadOnlySpan<float> queryEmbedding,
        IReadOnlyList<EpisodicTrace> traces,
        int coarseLimit,
        Span<float> cosineScoresOut)
    {
        if (cosineScoresOut.Length < traces.Count)
            throw new ArgumentException($"cosineScoresOut length ({cosineScoresOut.Length}) must be >= {traces.Count}.", nameof(cosineScoresOut));

        cosineScoresOut.Slice(0, traces.Count).Fill(float.NaN);

        if (traces.Count == 0 || queryEmbedding.IsEmpty)
            return;

        if (!AreAllFinite(queryEmbedding, out float qMaxAbs))
            throw new ArgumentException("Query embedding contains non-finite (NaN or Infinity) values.", nameof(queryEmbedding));

        int[] validIndices = ArrayPool<int>.Shared.Rent(traces.Count);
        int validCount = 0;

        try
        {
            for (int i = 0; i < traces.Count; i++)
            {
                float[]? emb = traces[i]?.Embedding;
                if (emb == null || emb.Length != queryEmbedding.Length || !AreAllFinite(emb))
                    continue;

                validIndices[validCount++] = i;
            }

            if (validCount == 0)
                return;

            if (qMaxAbs <= 0.0f)
            {
                for (int k = 0; k < validCount; k++)
                {
                    cosineScoresOut[validIndices[k]] = 0.0f;
                }
                return;
            }

            float queryNormSq = DistanceKernels.DotProduct(queryEmbedding, queryEmbedding);

            if (queryEmbedding.Length <= 64 || (coarseLimit > 0 && validCount <= coarseLimit))
            {
                for (int k = 0; k < validCount; k++)
                {
                    int traceIdx = validIndices[k];
                    cosineScoresOut[traceIdx] = ComputeCosineSimilarity(queryEmbedding, traces[traceIdx].Embedding!, queryNormSq);
                }
                return;
            }

            int paddedDim = HadamardProjection.GetPaddedDimension(queryEmbedding.Length);
            int words = (paddedDim + 63) >> 6;

            ulong[]? rentedQBits = words > StackAllocUlongLimit ? ArrayPool<ulong>.Shared.Rent(words) : null;
            Span<ulong> queryBits = words <= StackAllocUlongLimit
                ? stackalloc ulong[words]
                : rentedQBits.AsSpan(0, words);

            int heapCap = Math.Max(0, coarseLimit);
            (int Index, int Dist)[]? coarseBuf = heapCap > 0
                ? ArrayPool<(int Index, int Dist)>.Shared.Rent(heapCap)
                : null;

            try
            {
                QuantizeTo1BitCore(queryEmbedding, qMaxAbs, paddedDim, queryBits);
                var coarseHeap = coarseBuf != null ? new BoundedMaxIntHeap(coarseBuf, heapCap) : default;

                for (int k = 0; k < validCount; k++)
                {
                    int traceIdx = validIndices[k];
                    EpisodicTrace trace = traces[traceIdx];
                    ulong[]? traceBits = trace.BinaryEmbedding;
                    if (traceBits == null || traceBits.Length != words)
                    {
                        traceBits = QuantizeTo1Bit(trace.Embedding);
                        trace.BinaryEmbedding = traceBits;
                    }

                    int dist = DistanceKernels.HammingDistance(queryBits, traceBits.AsSpan(0, words));
                    cosineScoresOut[traceIdx] = HadamardProjection.EstimateCosineFromHamming(dist, paddedDim);

                    if (heapCap > 0)
                    {
                        coarseHeap.Offer(traceIdx, dist);
                    }
                }

                if (heapCap > 0)
                {
                    var coarseItems = coarseHeap.Buffer;
                    int coarseCount = coarseHeap.Count;
                    for (int k = 0; k < coarseCount; k++)
                    {
                        int traceIdx = coarseItems[k].Index;
                        cosineScoresOut[traceIdx] = ComputeCosineSimilarity(queryEmbedding, traces[traceIdx].Embedding!, queryNormSq);
                    }
                }
            }
            finally
            {
                if (coarseBuf != null) ArrayPool<(int Index, int Dist)>.Shared.Return(coarseBuf);
                if (rentedQBits != null) ArrayPool<ulong>.Shared.Return(rentedQBits);
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(validIndices);
        }
    }

    private static float ComputeCosineSimilarity(ReadOnlySpan<float> query, ReadOnlySpan<float> cand, float queryNormSq)
    {
        float candNormSq = DistanceKernels.DotProduct(cand, cand);
        float dot = DistanceKernels.DotProduct(query, cand);

        if (float.IsFinite(queryNormSq) && float.IsFinite(candNormSq) && float.IsFinite(dot))
        {
            if (queryNormSq > 1e-24f && candNormSq > 1e-24f)
            {
                float denom = MathF.Sqrt(queryNormSq) * MathF.Sqrt(candNormSq);
                if (float.IsFinite(denom) && denom > 0.0f)
                {
                    float cos = dot / denom;
                    if (float.IsFinite(cos))
                        return Math.Clamp(cos, -1.0f, 1.0f);
                }
            }
        }

        // Overflow/underflow-safe scale-normalized double fallback for extreme finite magnitudes
        double qMax = 0.0;
        double cMax = 0.0;
        for (int i = 0; i < query.Length; i++)
        {
            double qa = Math.Abs((double)query[i]);
            double ca = Math.Abs((double)cand[i]);
            if (qa > qMax) qMax = qa;
            if (ca > cMax) cMax = ca;
        }

        if (qMax <= 0.0 || cMax <= 0.0 || !double.IsFinite(qMax) || !double.IsFinite(cMax))
            return 0.0f;

        double invQ = 1.0 / qMax;
        double invC = 1.0 / cMax;
        double dDot = 0.0;
        double dQNormSq = 0.0;
        double dCNormSq = 0.0;
        for (int i = 0; i < query.Length; i++)
        {
            double qv = query[i] * invQ;
            double cv = cand[i] * invC;
            dDot += qv * cv;
            dQNormSq += qv * qv;
            dCNormSq += cv * cv;
        }

        double dDenom = Math.Sqrt(dQNormSq) * Math.Sqrt(dCNormSq);
        if (!double.IsFinite(dDenom) || dDenom <= 0.0)
            return 0.0f;

        double dCos = dDot / dDenom;
        if (!double.IsFinite(dCos))
            return 0.0f;

        return Math.Clamp((float)dCos, -1.0f, 1.0f);
    }
}
