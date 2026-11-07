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
    public static ulong[] QuantizeTo1Bit(ReadOnlySpan<float> embedding)
    {
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
        if (traces.Count == 0 || queryEmbedding.Length == 0) return [];

        // 1. Tier 1: Coarse 1-bit binary Hamming distance filter
        ulong[] queryBits = QuantizeTo1Bit(queryEmbedding);

        var coarseCandidates = new List<(EpisodicTrace Trace, int HammingDistance)>(traces.Count);
        foreach (var trace in traces)
        {
            if (trace.Embedding == null) continue;

            trace.BinaryEmbedding ??= QuantizeTo1Bit(trace.Embedding);

            int dist = 0;
            int len = Math.Min(queryBits.Length, trace.BinaryEmbedding.Length);
            for (int i = 0; i < len; i++)
            {
                dist += BitOperations.PopCount(queryBits[i] ^ trace.BinaryEmbedding[i]);
            }

            coarseCandidates.Add((trace, dist));
        }

        // Sort by Hamming distance ascending (smaller distance = higher similarity)
        var topCandidates = coarseCandidates
            .OrderBy(c => c.HammingDistance)
            .Take(coarseLimit)
            .Select(c => c.Trace)
            .ToList();

        // 2. Tier 2: Exact FP32 Cosine Similarity Re-Ranking
        var ranked = new List<(EpisodicTrace Trace, float CosineSimilarity)>(topCandidates.Count);
        float queryNorm = TensorPrimitives.Norm(queryEmbedding);
        if (queryNorm <= 1e-12f) queryNorm = 1.0f;

        foreach (var cand in topCandidates)
        {
            if (cand.Embedding == null || cand.Embedding.Length != queryEmbedding.Length)
                continue;

            float dot = TensorPrimitives.Dot(queryEmbedding, cand.Embedding);
            float candNorm = TensorPrimitives.Norm(cand.Embedding);
            if (candNorm <= 1e-12f) candNorm = 1.0f;

            float cosine = dot / (queryNorm * candNorm);
            ranked.Add((cand, Math.Clamp(cosine, -1.0f, 1.0f)));
        }

        return ranked
            .OrderByDescending(r => r.CosineSimilarity)
            .Take(finalLimit)
            .ToList();
    }
}
