namespace Glacier.MemEx.Budgeting;

using System;
using System.Collections.Generic;
using System.Text;
using Glacier.Bundle.Budgeting;
using Glacier.MemEx.Models;

public static class BreadcrumbTreeFolder
{
    private const string ContextHeader = "## Context: Episodic Memory Retrieval";
    private const int FoldedContentChars = 240;

    public static string FormatTraceContext(IReadOnlyList<EpisodicTrace> traces, bool foldDeepSubtrees = false)
    {
        if (traces == null || traces.Count == 0) return string.Empty;
        return FormatTraceContextRange(traces, traces.Count, foldDeepSubtrees);
    }

    public static string FormatTraceContext(
        IReadOnlyList<EpisodicTrace> traces,
        int maxTokens,
        ITokenEstimator? estimator = null,
        bool foldDeepSubtrees = false)
    {
        if (traces == null || traces.Count == 0 || maxTokens <= 0)
            return string.Empty;

        estimator ??= new HeuristicTokenEstimator();

        // 1. Try full requested rendering
        string formatted = FormatTraceContextRange(traces, traces.Count, foldDeepSubtrees);
        if (estimator.EstimateTokens(formatted) <= maxTokens)
            return formatted;

        // 2. Adaptive subtree folding when unfolded formatting overflows budget
        if (!foldDeepSubtrees)
        {
            formatted = FormatTraceContextRange(traces, traces.Count, foldDeepSubtrees: true);
            if (estimator.EstimateTokens(formatted) <= maxTokens)
                return formatted;
        }

        // 3. Drop lower-priority trailing traces while keeping as many top-utility traces as fit
        for (int count = traces.Count - 1; count >= 1; count--)
        {
            formatted = FormatTraceContextRange(traces, count, foldDeepSubtrees: true);
            if (estimator.EstimateTokens(formatted) <= maxTokens)
                return formatted;
        }

        // 4. Compact single top trace (collapse oversized breadcrumbs/entities if needed) and enforce TokenBudget cap
        string singleCompact = FormatCompactSingleTrace(traces[0], maxTokens, estimator);
        if (estimator.EstimateTokens(singleCompact) <= maxTokens)
            return singleCompact;

        var budget = new TokenBudget(maxTokens, estimator);
        string truncated = budget.TruncateToTokens(singleCompact, maxTokens);
        if (truncated.Length > 0 && truncated.Length < singleCompact.Length && char.IsHighSurrogate(truncated[^1]))
        {
            truncated = truncated.Substring(0, truncated.Length - 1);
        }

        while (truncated.Length > 0 && estimator.EstimateTokens(truncated) > maxTokens)
        {
            truncated = SafeSubstring(truncated, truncated.Length - 1);
        }

        return truncated;
    }

    private static string SafeSubstring(string value, int maxLength)
    {
        if (string.IsNullOrEmpty(value) || maxLength <= 0)
            return string.Empty;

        int len = Math.Min(maxLength, value.Length);
        if (len < value.Length && char.IsHighSurrogate(value[len - 1]))
        {
            len--;
        }

        return len > 0 ? value.Substring(0, len) : string.Empty;
    }

    internal static int EstimateHeaderTokens(ITokenEstimator estimator) =>
        estimator.EstimateTokens(ContextHeader + Environment.NewLine);

    internal static int EstimateTraceOverheadTokens(EpisodicTrace trace, ITokenEstimator estimator)
    {
        string breadcrumbStr = trace.Breadcrumbs is { Count: > 0 }
            ? string.Join(" > ", trace.Breadcrumbs)
            : "Episodic Trace";

        var sb = new StringBuilder(128);
        sb.AppendLine();
        sb.Append("### [10] ").AppendLine(breadcrumbStr);

        if (trace.Entities is { Count: > 0 })
        {
            sb.Append("*Entities*: ").AppendLine(string.Join(", ", trace.Entities));
        }

        sb.AppendLine();
        return estimator.EstimateTokens(sb.ToString());
    }

    private static string FormatTraceContextRange(IReadOnlyList<EpisodicTrace> traces, int count, bool foldDeepSubtrees)
    {
        var sb = new StringBuilder();
        sb.AppendLine(ContextHeader);

        for (int i = 0; i < count; i++)
        {
            EpisodicTrace trace = traces[i];
            sb.AppendLine();

            string breadcrumbStr = trace.Breadcrumbs is { Count: > 0 }
                ? string.Join(" > ", trace.Breadcrumbs)
                : "Episodic Trace";

            sb.AppendLine($"### [{i + 1}] {breadcrumbStr}");

            if (trace.Entities is { Count: > 0 })
            {
                sb.AppendLine($"*Entities*: {string.Join(", ", trace.Entities)}");
            }

            string content = (trace.Content ?? string.Empty).Trim();
            if (foldDeepSubtrees && content.Length > FoldedContentChars)
            {
                sb.AppendLine(SafeSubstring(content, FoldedContentChars) + " ... [folded subtree]");
            }
            else
            {
                sb.AppendLine(content);
            }
        }

        return sb.ToString();
    }

    private static string FormatCompactSingleTrace(EpisodicTrace trace, int maxTokens, ITokenEstimator estimator)
    {
        string breadcrumbStr = trace.Breadcrumbs is { Count: > 0 }
            ? (trace.Breadcrumbs.Count > 2
                ? $"{trace.Breadcrumbs[0]} > ... > {trace.Breadcrumbs[^1]}"
                : string.Join(" > ", trace.Breadcrumbs))
            : "Episodic Trace";

        if (breadcrumbStr.Length > 64)
        {
            breadcrumbStr = SafeSubstring(breadcrumbStr, 61) + "...";
        }

        var sb = new StringBuilder();
        sb.AppendLine(ContextHeader);
        sb.AppendLine();
        sb.AppendLine($"### [1] {breadcrumbStr}");

        if (trace.Entities is { Count: > 0 })
        {
            string entitiesStr = trace.Entities.Count > 4
                ? $"{string.Join(", ", trace.Entities[0], trace.Entities[1], trace.Entities[2])}, ... (+{trace.Entities.Count - 3})"
                : string.Join(", ", trace.Entities);
            if (entitiesStr.Length > 64)
            {
                entitiesStr = SafeSubstring(entitiesStr, 61) + "...";
            }
            sb.AppendLine($"*Entities*: {entitiesStr}");
        }

        string prefix = sb.ToString();
        int prefixTokens = estimator.EstimateTokens(prefix);
        int remainingTokens = maxTokens - prefixTokens;
        if (remainingTokens <= 0)
            return prefix;

        string content = (trace.Content ?? string.Empty).Trim();
        if (content.Length > FoldedContentChars)
        {
            content = SafeSubstring(content, FoldedContentChars) + " ... [folded subtree]";
        }

        if (estimator.EstimateTokens(prefix + content + Environment.NewLine) <= maxTokens)
        {
            sb.AppendLine(content);
            return sb.ToString();
        }

        var contentBudget = new TokenBudget(remainingTokens, estimator);
        string clippedContent = contentBudget.TruncateToTokens(content, Math.Max(1, remainingTokens - 1));
        if (clippedContent.Length > 0 && clippedContent.Length < content.Length && char.IsHighSurrogate(clippedContent[^1]))
        {
            clippedContent = clippedContent.Substring(0, clippedContent.Length - 1);
        }
        sb.AppendLine(clippedContent);
        return sb.ToString();
    }
}
