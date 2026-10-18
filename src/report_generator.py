"""
report_generator.py
--------------------
Generates downloadable text and HTML analysis reports.
"""

from __future__ import annotations

import datetime
import os
import pathlib

import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTS_DIR = _ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

DISCLAIMER = (
    "DISCLAIMER: Similarity scores do not automatically constitute evidence of "
    "plagiarism. They indicate textual overlap and should be reviewed by a human. "
    "AI-content detection is probabilistic and may produce false positives or "
    "false negatives. Results should NOT be used as definitive proof."
)


def _now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Plain-text report
# ---------------------------------------------------------------------------

def generate_text_report(
    doc_names: list[str],
    pairwise_df: pd.DataFrame,
    sentence_matches: list[dict],
    ai_results: list[dict],
) -> str:
    """
    Build a complete plain-text report and return it as a string.

    Parameters
    ----------
    doc_names : list[str]
        Names of all analysed documents.
    pairwise_df : pd.DataFrame
        Output from compute_document_similarity – pairwise similarity table.
    sentence_matches : list[dict]
        Output from find_all_sentence_matches.
    ai_results : list[dict]
        List of dicts with keys: name, prediction, confidence, ai_prob, human_prob, label.

    Returns
    -------
    str
        The full report as a plain-text string.
    """
    lines: list[str] = []
    sep = "=" * 70

    lines += [
        sep,
        "   AI CONTENT AND PLAGIARISM DETECTION REPORT",
        sep,
        f"Generated : {_now_str()}",
        "",
        DISCLAIMER,
        "",
    ]

    # 1. Summary
    lines += [
        sep,
        "1. ANALYSIS SUMMARY",
        sep,
        f"  • Documents analysed : {len(doc_names)}",
        f"  • Comparisons made   : {len(pairwise_df)}",
    ]
    if not pairwise_df.empty:
        highest = pairwise_df["Similarity (%)"].max()
        average = pairwise_df["Similarity (%)"].mean()
        lines += [
            f"  • Highest similarity : {highest:.2f}%",
            f"  • Average similarity : {average:.2f}%",
        ]
    lines.append("")

    # 2. Documents
    lines += [sep, "2. DOCUMENTS ANALYSED", sep]
    for i, name in enumerate(doc_names, 1):
        lines.append(f"  {i}. {name}")
    lines.append("")

    # 3. Pairwise similarity
    lines += [sep, "3. PAIRWISE SIMILARITY RESULTS", sep]
    if pairwise_df.empty:
        lines.append("  No comparisons available (need at least 2 documents).")
    else:
        col_w = 30
        header = (
            f"  {'Document A':<{col_w}} {'Document B':<{col_w}} "
            f"{'Similarity':>12}  Status"
        )
        lines.append(header)
        lines.append("  " + "-" * (col_w * 2 + 30))
        for _, row in pairwise_df.iterrows():
            lines.append(
                f"  {row['Document A']:<{col_w}} {row['Document B']:<{col_w}} "
                f"  {row['Similarity (%)']:>8.2f}%  {row['Status']}"
            )
    lines.append("")

    # 4. Sentence-level matches
    lines += [sep, "4. SENTENCE-LEVEL MATCHES", sep]
    if not sentence_matches:
        lines.append("  No highly similar sentence pairs found.")
    else:
        for idx, match in enumerate(sentence_matches[:30], 1):  # cap at 30
            lines += [
                f"  [{idx}] Similarity: {match['similarity']:.1f}%",
                f"  {match['doc_a']}:",
                f"    \"{match['sentence_a']}\"",
                f"  {match['doc_b']}:",
                f"    \"{match['sentence_b']}\"",
                "",
            ]
    lines.append("")

    # 5. AI detection
    lines += [sep, "5. AI CONTENT DETECTION RESULTS", sep]
    if not ai_results:
        lines.append("  No AI detection results available.")
    else:
        for res in ai_results:
            if res.get("model_ready"):
                lines += [
                    f"  Document  : {res['name']}",
                    f"  Prediction: {res['label']}",
                    f"  Confidence: {res['confidence']:.1f}%",
                    f"  AI prob   : {res['ai_prob']:.1f}%",
                    f"  Human prob: {res['human_prob']:.1f}%",
                    "",
                ]
            else:
                lines += [
                    f"  Document  : {res['name']}",
                    "  AI detection model not available.",
                    "",
                ]
    lines.append("")

    # Footer
    lines += [sep, DISCLAIMER, sep]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_html_report(
    doc_names: list[str],
    pairwise_df: pd.DataFrame,
    sentence_matches: list[dict],
    ai_results: list[dict],
) -> str:
    """
    Build a styled HTML report and return it as a string.

    Parameters
    ----------
    Same as generate_text_report.

    Returns
    -------
    str
        Full HTML document as a string.
    """
    timestamp = _now_str()

    def _status_color(status: str) -> str:
        return {
            "Low": "#27ae60",
            "Moderate": "#f39c12",
            "High": "#e67e22",
            "Very High": "#c0392b",
        }.get(status, "#7f8c8d")

    def _pred_color(label: str) -> str:
        if "AI" in label:
            return "#c0392b"
        if "Human" in label:
            return "#27ae60"
        return "#7f8c8d"

    # Pairwise table rows
    table_rows = ""
    for _, row in pairwise_df.iterrows():
        color = _status_color(row["Status"])
        table_rows += (
            f"<tr>"
            f"<td>{row['Document A']}</td>"
            f"<td>{row['Document B']}</td>"
            f"<td><strong>{row['Similarity (%)']:.2f}%</strong></td>"
            f"<td style='color:{color};font-weight:bold'>{row['Status']}</td>"
            f"</tr>"
        )

    # Sentence match rows
    sent_rows = ""
    for i, m in enumerate(sentence_matches[:30], 1):
        sent_rows += (
            f"<tr>"
            f"<td>{i}</td>"
            f"<td>{m['doc_a']}</td>"
            f"<td><em>{m['sentence_a']}</em></td>"
            f"<td>{m['doc_b']}</td>"
            f"<td><em>{m['sentence_b']}</em></td>"
            f"<td><strong>{m['similarity']:.1f}%</strong></td>"
            f"</tr>"
        )

    # AI results cards
    ai_cards = ""
    for res in ai_results:
        color = _pred_color(res.get("label", ""))
        if res.get("model_ready"):
            ai_cards += (
                f"<div class='card'>"
                f"<h3>{res['name']}</h3>"
                f"<p>Prediction: <span style='color:{color};font-weight:bold'>{res['label']}</span></p>"
                f"<p>Confidence: <strong>{res['confidence']:.1f}%</strong></p>"
                f"<p>AI probability: {res['ai_prob']:.1f}% &nbsp;|&nbsp; "
                f"Human probability: {res['human_prob']:.1f}%</p>"
                f"</div>"
            )
        else:
            ai_cards += (
                f"<div class='card'>"
                f"<h3>{res['name']}</h3>"
                f"<p>AI detection model not available. Please train the model first.</p>"
                f"</div>"
            )

    highest_sim = f"{pairwise_df['Similarity (%)'].max():.2f}%" if not pairwise_df.empty else "N/A"
    avg_sim = f"{pairwise_df['Similarity (%)'].mean():.2f}%" if not pairwise_df.empty else "N/A"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Content &amp; Plagiarism Detection Report</title>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #333; background: #f9f9f9; }}
    h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
    h2 {{ color: #2980b9; margin-top: 30px; }}
    .meta {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
    th {{ background: #3498db; color: white; padding: 10px; text-align: left; }}
    td {{ padding: 8px 10px; border-bottom: 1px solid #ddd; }}
    tr:hover {{ background: #f0f4ff; }}
    .card {{ background: white; border-left: 5px solid #3498db; padding: 15px 20px;
             margin: 10px 0; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .disclaimer {{ background: #fff3cd; border: 1px solid #ffc107; padding: 15px; border-radius: 4px;
                  font-size: 0.9em; color: #856404; margin: 20px 0; }}
    .summary-grid {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 10px 0; }}
    .summary-card {{ background: white; border-radius: 8px; padding: 15px 25px;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.12); text-align: center; }}
    .summary-card .value {{ font-size: 2em; font-weight: bold; color: #2980b9; }}
    .summary-card .label {{ font-size: 0.85em; color: #666; }}
  </style>
</head>
<body>
  <h1>🔍 AI Content &amp; Plagiarism Detection Report</h1>
  <p class="meta">Generated: {timestamp}</p>

  <div class="disclaimer">{DISCLAIMER}</div>

  <h2>1. Analysis Summary</h2>
  <div class="summary-grid">
    <div class="summary-card"><div class="value">{len(doc_names)}</div><div class="label">Documents</div></div>
    <div class="summary-card"><div class="value">{len(pairwise_df)}</div><div class="label">Comparisons</div></div>
    <div class="summary-card"><div class="value">{highest_sim}</div><div class="label">Highest Similarity</div></div>
    <div class="summary-card"><div class="value">{avg_sim}</div><div class="label">Average Similarity</div></div>
  </div>

  <h2>2. Documents Analysed</h2>
  <ol>{"".join(f"<li>{n}</li>" for n in doc_names)}</ol>

  <h2>3. Pairwise Similarity Results</h2>
  {"<p>No comparisons available.</p>" if pairwise_df.empty else
   f"<table><tr><th>Document A</th><th>Document B</th><th>Similarity</th><th>Status</th></tr>{table_rows}</table>"}

  <h2>4. Sentence-Level Matches</h2>
  {"<p>No highly similar sentence pairs found.</p>" if not sentence_matches else
   f"<table><tr><th>#</th><th>Doc A</th><th>Sentence A</th><th>Doc B</th><th>Sentence B</th><th>Similarity</th></tr>{sent_rows}</table>"}

  <h2>5. AI Content Detection Results</h2>
  {ai_cards if ai_cards else "<p>No AI detection results available.</p>"}

  <hr>
  <p class="disclaimer">{DISCLAIMER}</p>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_report(content: str, filename: str) -> pathlib.Path:
    """Save report content to the reports/ directory and return the path."""
    path = REPORTS_DIR / filename
    path.write_text(content, encoding="utf-8")
    return path

