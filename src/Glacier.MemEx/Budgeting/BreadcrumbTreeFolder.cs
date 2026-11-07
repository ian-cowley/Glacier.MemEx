namespace Glacier.MemEx.Budgeting;

using System;
using System.Collections.Generic;
using System.Text;
using Glacier.MemEx.Models;

public static class BreadcrumbTreeFolder
{
    public static string FormatTraceContext(IReadOnlyList<EpisodicTrace> traces, bool foldDeepSubtrees = false)
    {
        if (traces.Count == 0) return "";

        var sb = new StringBuilder();
        sb.AppendLine("## Context: Episodic Memory Retrieval");

        for (int i = 0; i < traces.Count; i++)
        {
            var trace = traces[i];
            sb.AppendLine();

            // Structural breadcrumbs header
            string breadcrumbStr = trace.Breadcrumbs.Count > 0
                ? string.Join(" > ", trace.Breadcrumbs)
                : "Episodic Trace";

            sb.AppendLine($"### [{i + 1}] {breadcrumbStr}");

            if (trace.Entities.Count > 0)
            {
                sb.AppendLine($"*Entities*: {string.Join(", ", trace.Entities)}");
            }

            string content = trace.Content.Trim();
            if (foldDeepSubtrees && content.Length > 240)
            {
                // Fold deep content to summary
                sb.AppendLine(content.Substring(0, 240) + " ... [folded subtree]");
            }
            else
            {
                sb.AppendLine(content);
            }
        }

        return sb.ToString();
    }
}
