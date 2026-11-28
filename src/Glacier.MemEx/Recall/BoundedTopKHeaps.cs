namespace Glacier.MemEx.Recall;

using System;
using System.Runtime.CompilerServices;

/// <summary>
/// Bounded binary max-heap for keeping the <c>K</c> smallest integer distances (root = largest distance in top-K).
/// </summary>
internal struct BoundedMaxIntHeap
{
    private readonly (int Index, int Dist)[] _buffer;
    private readonly int _capacity;
    private int _count;

    public BoundedMaxIntHeap((int Index, int Dist)[] buffer, int capacity)
    {
        _buffer = buffer;
        _capacity = Math.Min(capacity, buffer.Length);
        _count = 0;
    }

    public readonly int Count => _count;
    public readonly (int Index, int Dist)[] Buffer => _buffer;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Offer(int index, int dist)
    {
        if (_capacity <= 0) return;

        if (_count < _capacity)
        {
            int i = _count++;
            while (i > 0)
            {
                int p = (i - 1) >> 1;
                if (dist <= _buffer[p].Dist) break;
                _buffer[i] = _buffer[p];
                i = p;
            }
            _buffer[i] = (index, dist);
        }
        else if (dist < _buffer[0].Dist)
        {
            // Replace root and sift down
            int i = 0;
            int half = _count >> 1;
            while (i < half)
            {
                int child = (i << 1) + 1;
                int right = child + 1;
                if (right < _count && _buffer[right].Dist > _buffer[child].Dist)
                {
                    child = right;
                }
                if (dist >= _buffer[child].Dist) break;
                _buffer[i] = _buffer[child];
                i = child;
            }
            _buffer[i] = (index, dist);
        }
    }
}

/// <summary>
/// Bounded binary min-heap for keeping the <c>K</c> largest float scores (root = smallest score in top-K).
/// </summary>
internal struct BoundedMinFloatHeap
{
    private readonly (int Index, float Score)[] _buffer;
    private readonly int _capacity;
    private int _count;

    public BoundedMinFloatHeap((int Index, float Score)[] buffer, int capacity)
    {
        _buffer = buffer;
        _capacity = Math.Min(capacity, buffer.Length);
        _count = 0;
    }

    public readonly int Count => _count;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Offer(int index, float score)
    {
        if (_capacity <= 0 || float.IsNaN(score)) return;

        if (_count < _capacity)
        {
            int i = _count++;
            while (i > 0)
            {
                int p = (i - 1) >> 1;
                if (score >= _buffer[p].Score) break;
                _buffer[i] = _buffer[p];
                i = p;
            }
            _buffer[i] = (index, score);
        }
        else if (score > _buffer[0].Score)
        {
            int i = 0;
            int half = _count >> 1;
            while (i < half)
            {
                int child = (i << 1) + 1;
                int right = child + 1;
                if (right < _count && _buffer[right].Score < _buffer[child].Score)
                {
                    child = right;
                }
                if (score <= _buffer[child].Score) break;
                _buffer[i] = _buffer[child];
                i = child;
            }
            _buffer[i] = (index, score);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryDequeue(out int index, out float score)
    {
        if (_count == 0)
        {
            index = -1;
            score = float.MinValue;
            return false;
        }

        index = _buffer[0].Index;
        score = _buffer[0].Score;
        _count--;

        if (_count > 0)
        {
            var target = _buffer[_count];
            int i = 0;
            int half = _count >> 1;
            while (i < half)
            {
                int child = (i << 1) + 1;
                int right = child + 1;
                if (right < _count && _buffer[right].Score < _buffer[child].Score)
                {
                    child = right;
                }
                if (target.Score <= _buffer[child].Score) break;
                _buffer[i] = _buffer[child];
                i = child;
            }
            _buffer[i] = target;
        }

        return true;
    }
}

/// <summary>
/// Bounded binary min-heap for keeping the <c>K</c> largest double-scored objects (root = smallest score in top-K).
/// </summary>
internal struct BoundedMinItemHeap<T>
{
    private readonly (T Item, double Score)[] _buffer;
    private readonly int _capacity;
    private int _count;

    public BoundedMinItemHeap((T Item, double Score)[] buffer, int capacity)
    {
        _buffer = buffer;
        _capacity = Math.Min(capacity, buffer.Length);
        _count = 0;
    }

    public readonly int Count => _count;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Offer(T item, double score)
    {
        if (_capacity <= 0 || double.IsNaN(score)) return;

        if (_count < _capacity)
        {
            int i = _count++;
            while (i > 0)
            {
                int p = (i - 1) >> 1;
                if (score >= _buffer[p].Score) break;
                _buffer[i] = _buffer[p];
                i = p;
            }
            _buffer[i] = (item, score);
        }
        else if (score > _buffer[0].Score)
        {
            int i = 0;
            int half = _count >> 1;
            while (i < half)
            {
                int child = (i << 1) + 1;
                int right = child + 1;
                if (right < _count && _buffer[right].Score < _buffer[child].Score)
                {
                    child = right;
                }
                if (score <= _buffer[child].Score) break;
                _buffer[i] = _buffer[child];
                i = child;
            }
            _buffer[i] = (item, score);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryDequeue(out T item, out double score)
    {
        if (_count == 0)
        {
            item = default!;
            score = double.MinValue;
            return false;
        }

        item = _buffer[0].Item;
        score = _buffer[0].Score;
        _count--;

        if (_count > 0)
        {
            var target = _buffer[_count];
            int i = 0;
            int half = _count >> 1;
            while (i < half)
            {
                int child = (i << 1) + 1;
                int right = child + 1;
                if (right < _count && _buffer[right].Score < _buffer[child].Score)
                {
                    child = right;
                }
                if (target.Score <= _buffer[child].Score) break;
                _buffer[i] = _buffer[child];
                i = child;
            }
            _buffer[i] = target;
        }

        return true;
    }
}
