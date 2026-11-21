namespace Glacier.MemEx.Recall;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Numerics.Tensors;
using Glacier.MemEx.Models;
using Glacier.Quant.Kernels;

public static class TwoTierVectorFilter
{
    internal static bool AreAllFinite(ReadOnlySpan<float> values)
    {
        for (int i = 0; i < values.Length; i++)
        {
            if (!float.IsFinite(values[i]))
                return false;
        }
        return true;
    }

    public static ulong[] QuantizeTo1Bit(ReadOnlySpan<float> embedding)
    {
        if (!AreAllFinite(embedding))
            throw new ArgumentException("Embedding contains non-finite (NaN or Infinity) values.", nameof(embedding));

        int blocks = (embedding.Length + BlockBinary1Bit.BlockSize - 1) / BlockBinary1Bit.BlockSize;
        int paddedLen = blocks * BlockBinary1Bit.BlockSize;

        float[] padded = new float[paddedLen];
        embedding.CopyTo(padded);

        var blockArray = new BlockBinary1Bit[blocks];
        SimdQuantizer.QuantizeRow1Bit(padded, blockArray);

        ulong[] bits = new ulong[blocks];
        for (int i = 0; i < blocks; i++)
        {
            bits[i] = blockArray[i].Bits;
        }
        return bits;
    }

    /// <summary>
    /// Filters traces using 1-bit Hamming distance coarse filter (Tier 1),
    /// followed by exact FP32 cosine similarity re-ranking (Tier 2).
    /// </summary>
    public static List<(EpisodicTrace Trace, float CosineSimilarity)> FilterAndRerank(
        ReadOnlySpan<float> queryEmbedding,
        IReadOnlyList<EpisodicTrace> traces,
        int coarseLimit = 64,
        int finalLimit = 16)
    {
        ArgumentNullException.ThrowIfNull(traces);

        if (!AreAllFinite(queryEmbedding))
            throw new ArgumentException("Query embedding contains non-finite (NaN or Infinity) values.", nameof(queryEmbedding));

        if (traces.Count == 0 || queryEmbedding.Length == 0 || coarseLimit <= 0 || finalLimit <= 0)
            return [];

        // Filter out null, mismatched-dimension, or non-finite candidate embeddings BEFORE Tier 1
        // so invalid candidates never occupy coarseLimit slots with HammingDistance == 0.
        var validTraces = new List<EpisodicTrace>(traces.Count);
        foreach (var trace in traces)
        {
            if (trace?.Embedding == null ||
                trace.Embedding.Length != queryEmbedding.Length ||
                !AreAllFinite(trace.Embedding))
            {
                continue;
            }
            validTraces.Add(trace);
        }

        if (validTraces.Count == 0)
            return [];

        IReadOnlyList<EpisodicTrace> topCandidates;
        // For dimensions <= BlockBinary1Bit.BlockSize (64), 1-bit block zero-padding + mean-bias
        // quantization distorts magnitudes and fits in a single ulong, whereas exact FP32 cosine
        // is both faster and eliminates small-dimension starvation. Same when validTraces <= coarseLimit.
        if (queryEmbedding.Length <= BlockBinary1Bit.BlockSize || validTraces.Count <= coarseLimit)
        {
            topCandidates = validTraces;
        }
        else
        {
            // 1. Tier 1: Coarse 1-bit binary Hamming distance filter
            ulong[] queryBits = QuantizeTo1Bit(queryEmbedding);

            var coarseCandidates = new List<(EpisodicTrace Trace, int HammingDistance)>(validTraces.Count);
            foreach (var trace in validTraces)
            {
                if (trace.BinaryEmbedding == null || trace.BinaryEmbedding.Length != queryBits.Length)
                {
                    trace.BinaryEmbedding = QuantizeTo1Bit(trace.Embedding);
                }

                int dist = 0;
                for (int i = 0; i < queryBits.Length; i++)
                {
                    dist += BitOperations.PopCount(queryBits[i] ^ trace.BinaryEmbedding[i]);
                }

                coarseCandidates.Add((trace, dist));
            }

            topCandidates = coarseCandidates
                .OrderBy(c => c.HammingDistance)
                .Take(coarseLimit)
                .Select(c => c.Trace)
                .ToList();
        }

        // 2. Tier 2: Exact FP32 Cosine Similarity Re-Ranking
        var ranked = new List<(EpisodicTrace Trace, float CosineSimilarity)>(topCandidates.Count);
        float queryNorm = TensorPrimitives.Norm(queryEmbedding);

        foreach (var cand in topCandidates)
        {
            float cosine = ComputeCosineSimilarity(queryEmbedding, cand.Embedding!, queryNorm);
            ranked.Add((cand, cosine));
        }

        return ranked
            .OrderByDescending(r => r.CosineSimilarity)
            .Take(finalLimit)
            .ToList();
    }

    private static float ComputeCosineSimilarity(ReadOnlySpan<float> query, ReadOnlySpan<float> cand, float queryNorm)
    {
        float candNorm = TensorPrimitives.Norm(cand);
        float dot = TensorPrimitives.Dot(query, cand);

        if (float.IsFinite(queryNorm) && float.IsFinite(candNorm) && float.IsFinite(dot))
        {
            if (queryNorm <= 1e-12f || candNorm <= 1e-12f)
                return 0.0f;

            float denom = queryNorm * candNorm;
            if (float.IsFinite(denom) && denom > 0.0f)
            {
                float cos = dot / denom;
                if (float.IsFinite(cos))
                    return Math.Clamp(cos, -1.0f, 1.0f);
            }
        }

        // Overflow-safe double fallback for extreme finite magnitudes
        double dDot = 0.0;
        double dQNormSq = 0.0;
        double dCNormSq = 0.0;
        for (int i = 0; i < query.Length; i++)
        {
            double qv = query[i];
            double cv = cand[i];
            dDot += qv * cv;
            dQNormSq += qv * qv;
            dCNormSq += cv * cv;
        }

        double dDenom = Math.Sqrt(dQNormSq) * Math.Sqrt(dCNormSq);
        if (!double.IsFinite(dDenom) || dDenom <= 1e-24)
            return 0.0f;

        double dCos = dDot / dDenom;
        if (!double.IsFinite(dCos))
            return 0.0f;

        return Math.Clamp((float)dCos, -1.0f, 1.0f);
    }
}
