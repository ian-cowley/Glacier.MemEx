namespace Glacier.MemEx.Models;

using System;
using System.Collections.Generic;

public sealed class EpisodicTrace
{
    private readonly object _syncRoot = new();
    private readonly List<long> _accessTimestampsUs = [];

    public string Id { get; init; } = Guid.NewGuid().ToString("N");
    public string Content { get; init; } = "";
    public long TimestampUs { get; init; } = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() * 1000L;
    public IReadOnlyList<long> AccessTimestampsUs
    {
        get
        {
            lock (_syncRoot)
            {
                return _accessTimestampsUs.Count == 0 ? [] : _accessTimestampsUs.ToArray();
            }
        }
    }
    public int AccessCount
    {
        get
        {
            lock (_syncRoot)
            {
                return _accessTimestampsUs.Count;
            }
        }
    }
    public object SyncRoot => _syncRoot;
    public List<string> Breadcrumbs { get; init; } = [];
    public float[]? Embedding { get; init; }
    public ulong[]? BinaryEmbedding { get; set; }
    public List<string> Entities { get; init; } = [];
    public int TokenCount { get; init; }
    public float BaseImportance { get; init; } = 1.0f;
    public Dictionary<string, string> Metadata { get; init; } = [];

    public void RecordAccess(long timestampUs)
    {
        lock (_syncRoot)
        {
            _accessTimestampsUs.Add(timestampUs);
        }
    }

    public void RecordAccess(IEnumerable<long> timestampsUs)
    {
        ArgumentNullException.ThrowIfNull(timestampsUs);
        lock (_syncRoot)
        {
            _accessTimestampsUs.AddRange(timestampsUs);
        }
    }

    public double ComputeAccessDecaySum(long currentTimestampUs, double d, double tauSeconds)
    {
        lock (_syncRoot)
        {
            int count = _accessTimestampsUs.Count;
            if (count == 0) return 0.0;

            double sum = 0.0;
            for (int i = 0; i < count; i++)
            {
                double deltaSec = Math.Max(0.0, (currentTimestampUs - _accessTimestampsUs[i]) / 1_000_000.0);
                sum += Math.Pow(deltaSec + tauSeconds, -d);
            }
            return sum;
        }
    }

    public long[] GetAccessTimestampsSnapshot()
    {
        lock (_syncRoot)
        {
            return _accessTimestampsUs.Count == 0 ? [] : _accessTimestampsUs.ToArray();
        }
    }
}

public sealed class MemExOptions
{
    public double DecayParameterD { get; init; } = 0.5;
    public double TimeScaleTauSeconds { get; init; } = 1.0;
    public float WeightDecay { get; init; } = 0.35f;
    public float WeightVector { get; init; } = 0.35f;
    public float WeightGraph { get; init; } = 0.20f;
    public float WeightImportance { get; init; } = 0.10f;
    public int CoarseCandidateLimit { get; init; } = 64;
    public int MaxPromptTokens { get; init; } = 2048;
}

public sealed class MemoryQueryResult
{
    public IReadOnlyList<EpisodicTrace> SelectedTraces { get; init; } = [];
    public int TotalTokens { get; init; }
    public double TotalUtility { get; init; }
    public string FormattedContext { get; init; } = "";
}
