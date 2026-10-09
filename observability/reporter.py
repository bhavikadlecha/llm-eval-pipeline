"""
Observability & Reporting Module.

Logs every test result to a JSONL file and generates a final
rich-formatted console report + a self-contained HTML report.
"""
import json
import os
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
JSONL_LOG_PATH = REPORTS_DIR / f"eval_run_{RUN_TIMESTAMP}.jsonl"
HTML_REPORT_PATH = REPORTS_DIR / f"eval_report_{RUN_TIMESTAMP}.html"


def log_result(
    test_id: str,
    category: str,
    input_prompt: str,
    rag_response: str,
    passed_test: bool,
    confidence_score: float,
    reasoning_log: str,
) -> None:
    """Append a single test result to the JSONL log file."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "test_id": test_id,
        "category": category,
        "input_prompt": input_prompt,
        "rag_response": rag_response,
        "passed_test": passed_test,
        "confidence_score": round(confidence_score, 4),
        "reasoning_log": reasoning_log,
    }
    with open(JSONL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def generate_final_report() -> None:
    """
    Read all results from the JSONL log and:
    1. Print a rich-formatted summary table to the console.
    2. Write a self-contained HTML report.
    """
    if not JSONL_LOG_PATH.exists():
        console.print("[red]No log file found. No tests were run.[/red]")
        return

    results = []
    with open(JSONL_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if not results:
        console.print("[yellow]Log file is empty.[/yellow]")
        return

    total = len(results)
    passed = sum(1 for r in results if r["passed_test"])
    failed = total - passed
    pass_rate = (passed / total * 100) if total > 0 else 0

    # ── Rich Console Table ───────────────────────────────────────────────────
    console.rule("[bold blue]LLM Evaluation Pipeline - Final Report[/bold blue]")

    table = Table(box=box.ROUNDED, show_lines=True, header_style="bold magenta")
    table.add_column("Test ID", style="cyan", no_wrap=True)
    table.add_column("Category", style="yellow")
    table.add_column("Passed", justify="center")
    table.add_column("Confidence", justify="center")
    table.add_column("Reasoning (excerpt)")

    for r in results:
        status = "PASS" if r["passed_test"] else "FAIL"
        style = "green" if r["passed_test"] else "red"
        excerpt = r["reasoning_log"][:80] + "..." if len(r["reasoning_log"]) > 80 else r["reasoning_log"]
        table.add_row(
            r["test_id"],
            r["category"],
            f"[{style}]{status}[/{style}]",
            f"{r['confidence_score']:.2f}",
            excerpt,
        )

    console.print(table)
    console.print()
    console.print(f"[bold]Total Tests :[/bold] {total}")
    console.print(f"[bold green]Passed      :[/bold green] {passed}")
    console.print(f"[bold red]Failed      :[/bold red] {failed}")
    console.print(f"[bold]Pass Rate   :[/bold] {pass_rate:.1f}%")
    console.print(f"\n[dim]JSONL log   : {JSONL_LOG_PATH}[/dim]")
    console.print(f"[dim]HTML report : {HTML_REPORT_PATH}[/dim]")
    console.rule()

    # ── HTML Report ──────────────────────────────────────────────────────────
    rows_html = ""
    for r in results:
        status_label = "PASS" if r["passed_test"] else "FAIL"
        row_class = "pass" if r["passed_test"] else "fail"
        rows_html += f"""
        <tr class="{row_class}">
            <td>{r['test_id']}</td>
            <td>{r['category']}</td>
            <td class="status">{status_label}</td>
            <td>{r['confidence_score']:.2f}</td>
            <td class="reason">{r['reasoning_log']}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LLM Eval Report - {RUN_TIMESTAMP}</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e0e0e0; padding: 2rem; }}
  h1 {{ color: #7c9ef7; }}
  .summary {{ display: flex; gap: 2rem; margin-bottom: 1.5rem; }}
  .card {{ background: #1e2030; border-radius: 8px; padding: 1rem 2rem; text-align: center; }}
  .card .num {{ font-size: 2rem; font-weight: bold; }}
  .card.total .num {{ color: #7c9ef7; }}
  .card.passed .num {{ color: #56d364; }}
  .card.failed .num {{ color: #f85149; }}
  .card.rate .num {{ color: #f0b429; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ background: #1e2030; padding: 0.75rem; text-align: left; color: #7c9ef7; }}
  td {{ padding: 0.65rem; border-bottom: 1px solid #2a2d3e; vertical-align: top; }}
  tr.pass td.status {{ color: #56d364; font-weight: bold; }}
  tr.fail td.status {{ color: #f85149; font-weight: bold; }}
  td.reason {{ font-size: 0.85rem; color: #a8b2d8; }}
  tr:hover {{ background: #1e2030; }}
</style>
</head>
<body>
<h1>LLM Evaluation Pipeline - Build Report</h1>
<p style="color:#666">Run: {RUN_TIMESTAMP}</p>
<div class="summary">
  <div class="card total"><div class="num">{total}</div>Total Tests</div>
  <div class="card passed"><div class="num">{passed}</div>Passed</div>
  <div class="card failed"><div class="num">{failed}</div>Failed</div>
  <div class="card rate"><div class="num">{pass_rate:.1f}%</div>Pass Rate</div>
</div>
<table>
  <thead>
    <tr><th>Test ID</th><th>Category</th><th>Status</th><th>Confidence</th><th>Reasoning Log</th></tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
</body>
</html>"""

    with open(HTML_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
