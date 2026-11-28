namespace Glacier.MemEx.Recall;

using System;
using System.Collections.Generic;
using Glacier.MemEx.Models;

/// <summary>
/// Immutable 1-based Compressed Sparse Row (CSR) representation of the entity co-occurrence graph,
/// ready for contiguous-span <c>Glacier.Graph.Kernels.CsrKernels.PersonalizedPageRank</c>.
/// </summary>
internal sealed class EntityCsrGraph
{
    public static readonly EntityCsrGraph Empty = new(
        nodeCount: 0,
        rowOffsets: [0, 0],
        columnIndices: [],
        nodeNames: [string.Empty],
        entityToNodeId: new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase),
        version: 0);

    public int NodeCount { get; }
    public int[] RowOffsets { get; }
    public int[] ColumnIndices { get; }
    public string[] NodeNames { get; }
    public Dictionary<string, int> EntityToNodeId { get; }
    public long Version { get; }

    private EntityCsrGraph(
        int nodeCount,
        int[] rowOffsets,
        int[] columnIndices,
        string[] nodeNames,
        Dictionary<string, int> entityToNodeId,
        long version)
    {
        NodeCount = nodeCount;
        RowOffsets = rowOffsets;
        ColumnIndices = columnIndices;
        NodeNames = nodeNames;
        EntityToNodeId = entityToNodeId;
        Version = version;
    }

    public static EntityCsrGraph Build(IReadOnlyList<EpisodicTrace> traces, long version = 0)
    {
        if (traces == null || traces.Count == 0)
        {
            return version == 0
                ? Empty
                : new EntityCsrGraph(0, [0, 0], [], [string.Empty], new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase), version);
        }

        var entityToNodeId = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var names = new List<string> { string.Empty };

        // Pass 1: Assign 1-based node IDs to every distinct non-whitespace entity
        for (int t = 0; t < traces.Count; t++)
        {
            var ents = traces[t]?.Entities;
            if (ents == null || ents.Count == 0) continue;

            for (int i = 0; i < ents.Count; i++)
            {
                string entity = ents[i];
                if (string.IsNullOrWhiteSpace(entity)) continue;

                if (!entityToNodeId.ContainsKey(entity))
                {
                    int id = names.Count;
                    entityToNodeId[entity] = id;
                    names.Add(entity);
                }
            }
        }

        int nodeCount = names.Count - 1;
        if (nodeCount == 0)
        {
            return version == 0
                ? Empty
                : new EntityCsrGraph(0, [0, 0], [], [string.Empty], entityToNodeId, version);
        }

        // Pass 2: Build deduplicated undirected co-occurrence adjacency sets (self-loops excluded)
        var adj = new HashSet<int>?[nodeCount + 1];
        var traceNodeIds = new List<int>(8);

        for (int t = 0; t < traces.Count; t++)
        {
            var ents = traces[t]?.Entities;
            if (ents == null || ents.Count < 2) continue;

            traceNodeIds.Clear();
            for (int i = 0; i < ents.Count; i++)
            {
                string entity = ents[i];
                if (string.IsNullOrWhiteSpace(entity)) continue;

                if (entityToNodeId.TryGetValue(entity, out int nodeId) && !traceNodeIds.Contains(nodeId))
                {
                    traceNodeIds.Add(nodeId);
                }
            }

            if (traceNodeIds.Count < 2) continue;

            for (int i = 0; i < traceNodeIds.Count; i++)
            {
                int u = traceNodeIds[i];
                HashSet<int> setU = adj[u] ??= [];
                for (int j = 0; j < traceNodeIds.Count; j++)
                {
                    if (i != j)
                    {
                        setU.Add(traceNodeIds[j]);
                    }
                }
            }
        }

        // Pass 3: Pack into 1-based CSR arrays with sorted, deduplicated column indices
        var rowOffsets = new int[nodeCount + 2];
        rowOffsets[0] = 0;
        rowOffsets[1] = 0;
        for (int u = 1; u <= nodeCount; u++)
        {
            int degree = adj[u]?.Count ?? 0;
            rowOffsets[u + 1] = checked(rowOffsets[u] + degree);
        }

        int totalEdges = rowOffsets[nodeCount + 1];
        int[] columnIndices = totalEdges == 0 ? [] : new int[totalEdges];

        for (int u = 1; u <= nodeCount; u++)
        {
            HashSet<int>? setU = adj[u];
            if (setU == null || setU.Count == 0) continue;

            int start = rowOffsets[u];
            int idx = start;
            foreach (int v in setU)
            {
                columnIndices[idx++] = v;
            }

            Array.Sort(columnIndices, start, setU.Count);
        }

        return new EntityCsrGraph(
            nodeCount,
            rowOffsets,
            columnIndices,
            names.ToArray(),
            entityToNodeId,
            version);
    }
}
