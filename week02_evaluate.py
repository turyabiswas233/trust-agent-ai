"""Week 02 — Evaluate extractions: validate, score, and generate charts.

Reads the raw extraction JSONL produced by week02_extract.py, validates
against the schema, compares to the answer key, computes metrics, and
produces matplotlib charts plus a markdown report.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import jsonschema
except ImportError:
    jsonschema = None  # type: ignore[assignment]

SCHEMA_PATH = Path("schemas/notice_schema_v1.json")
ANSWER_KEY_PATH = Path("data/answer_key.json")
EXTRACTIONS_PATH = Path("output/week02/extractions_raw.jsonl")
OUTPUT_DIR = Path("output/week02")
CHARTS_DIR = OUTPUT_DIR / "charts"

SCORED_FIELDS = [
    "notice_id", "domain_name", "registrar", "notice_type",
    "notice_date", "action_required", "deadline", "amount_due_usd",
]


def load_schema() -> dict[str, Any]:
    with SCHEMA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def load_answer_key() -> dict[str, dict[str, Any]]:
    """Load answer key indexed by notice file name (notice_001.txt -> entry)."""
    with ANSWER_KEY_PATH.open(encoding="utf-8") as f:
        entries = json.load(f)
    key = {}
    for i, entry in enumerate(entries):
        fname = f"notice_{i+1:03d}.txt"
        key[fname] = entry
    return key


def load_extractions() -> list[dict[str, Any]]:
    records = []
    with EXTRACTIONS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def try_parse_json(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """Attempt to parse JSON from raw LLM output, handling markdown fences."""
    text = raw.strip()
    # Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    # Try parsing
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, None
        return None, "Parsed JSON is not an object"
    except json.JSONDecodeError as e:
        return None, str(e)


def validate_against_schema(obj: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Validate a parsed object against the JSON schema. Returns list of errors."""
    if jsonschema is None:
        return []
    errors = []
    validator = jsonschema.Draft7Validator(schema)
    for error in validator.iter_errors(obj):
        errors.append(error.message)
    return errors


def score_fields(
    extracted: dict[str, Any], expected: dict[str, Any]
) -> dict[str, bool]:
    """Compare extracted fields to expected, return per-field match."""
    results = {}
    for field in SCORED_FIELDS:
        ext_val = extracted.get(field)
        exp_val = expected.get(field)
        if field == "amount_due_usd":
            # Numeric comparison with tolerance
            if ext_val is None and exp_val is None:
                results[field] = True
            elif ext_val is not None and exp_val is not None:
                try:
                    results[field] = abs(float(ext_val) - float(exp_val)) < 0.02
                except (ValueError, TypeError):
                    results[field] = False
            else:
                results[field] = False
        elif field == "action_required":
            results[field] = bool(ext_val) == bool(exp_val)
        else:
            results[field] = str(ext_val or "").strip().lower() == str(exp_val or "").strip().lower()
    return results


def evaluate_all(
    records: list[dict[str, Any]],
    schema: dict[str, Any],
    answer_key: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate all extraction records. Returns structured metrics."""
    methods = {}
    for rec in records:
        method = rec["method"]
        run = rec["run"]
        notice_file = rec["notice_file"]

        if method not in methods:
            methods[method] = {}
        if run not in methods[method]:
            methods[method][run] = {
                "total": 0,
                "valid_json": 0,
                "schema_valid": 0,
                "field_scores": {f: [] for f in SCORED_FIELDS},
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost_usd": 0.0,
                "latencies": [],
            }

        run_data = methods[method][run]
        run_data["total"] += 1
        run_data["total_input_tokens"] += rec.get("input_tokens", 0)
        run_data["total_output_tokens"] += rec.get("output_tokens", 0)
        run_data["total_cost_usd"] += rec.get("cost_usd", 0.0)
        run_data["latencies"].append(rec.get("latency_s", 0.0))

        parsed, parse_error = try_parse_json(rec.get("raw_output", ""))
        if parsed is None:
            run_data["field_scores"] = {
                f: run_data["field_scores"][f] + [False] for f in SCORED_FIELDS
            }
            continue

        run_data["valid_json"] += 1

        schema_errors = validate_against_schema(parsed, schema)
        if not schema_errors:
            run_data["schema_valid"] += 1

        expected = answer_key.get(notice_file)
        if expected:
            field_matches = score_fields(parsed, expected)
            for f in SCORED_FIELDS:
                run_data["field_scores"][f].append(field_matches.get(f, False))

    # Aggregate per method
    summary = {}
    for method, runs in methods.items():
        run_metrics = []
        for run_num, rd in sorted(runs.items()):
            invalid_rate = 1.0 - (rd["schema_valid"] / rd["total"]) if rd["total"] else 0
            field_accuracy = {}
            for f in SCORED_FIELDS:
                scores = rd["field_scores"][f]
                field_accuracy[f] = sum(scores) / len(scores) if scores else 0.0
            overall_accuracy = (
                sum(field_accuracy.values()) / len(field_accuracy) if field_accuracy else 0
            )
            run_metrics.append({
                "run": run_num,
                "total": rd["total"],
                "valid_json": rd["valid_json"],
                "schema_valid": rd["schema_valid"],
                "invalid_rate": invalid_rate,
                "field_accuracy": field_accuracy,
                "overall_accuracy": overall_accuracy,
                "total_input_tokens": rd["total_input_tokens"],
                "total_output_tokens": rd["total_output_tokens"],
                "total_cost_usd": rd["total_cost_usd"],
                "mean_latency_s": statistics.mean(rd["latencies"]) if rd["latencies"] else 0,
                "latencies": rd["latencies"],
            })
        # Spread across runs
        invalid_rates = [rm["invalid_rate"] for rm in run_metrics]
        overall_accs = [rm["overall_accuracy"] for rm in run_metrics]
        summary[method] = {
            "runs": run_metrics,
            "mean_invalid_rate": statistics.mean(invalid_rates),
            "std_invalid_rate": statistics.stdev(invalid_rates) if len(invalid_rates) > 1 else 0,
            "mean_overall_accuracy": statistics.mean(overall_accs),
            "std_overall_accuracy": statistics.stdev(overall_accs) if len(overall_accs) > 1 else 0,
            "total_cost_usd": sum(rm["total_cost_usd"] for rm in run_metrics),
            "total_tokens": sum(
                rm["total_input_tokens"] + rm["total_output_tokens"] for rm in run_metrics
            ),
        }
    return summary


# ─── Chart Generation ────────────────────────────────────────────────────

def _setup_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

METHOD_LABELS = {"prompt_only": "Method A\n(Prompt-Only)", "grammar_constrained": "Method B\n(Grammar-Constrained)"}
METHOD_COLORS = {"prompt_only": "#e74c3c", "grammar_constrained": "#2ecc71"}


def chart_invalid_rate(summary: dict[str, Any], out_dir: Path):
    """Bar chart: invalid rate per run, grouped by method."""
    _setup_style()
    fig, ax = plt.subplots()

    method_names = list(summary.keys())
    num_runs = len(summary[method_names[0]]["runs"])
    x = np.arange(num_runs)
    width = 0.35

    for i, method in enumerate(method_names):
        rates = [r["invalid_rate"] * 100 for r in summary[method]["runs"]]
        offset = (i - 0.5) * width + width / 2
        bars = ax.bar(
            x + offset, rates, width,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, None),
            edgecolor="white", linewidth=0.5
        )
        for bar, val in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Run")
    ax.set_ylabel("Invalid Rate (%)")
    ax.set_title("Invalid Rate per Run: Prompt-Only vs Grammar-Constrained")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Run {i+1}" for i in range(num_runs)])
    ax.legend()
    ax.set_ylim(0, max(100, ax.get_ylim()[1] * 1.15))
    fig.tight_layout()
    fig.savefig(out_dir / "invalid_rate_comparison.png", dpi=150)
    plt.close(fig)


def chart_field_accuracy(summary: dict[str, Any], out_dir: Path):
    """Grouped bar chart: mean field accuracy per field, by method."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    method_names = list(summary.keys())
    x = np.arange(len(SCORED_FIELDS))
    width = 0.35

    for i, method in enumerate(method_names):
        # Average across runs
        mean_accs = []
        for field in SCORED_FIELDS:
            vals = [r["field_accuracy"][field] * 100 for r in summary[method]["runs"]]
            mean_accs.append(statistics.mean(vals))
        offset = (i - 0.5) * width + width / 2
        ax.bar(
            x + offset, mean_accs, width,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, None),
            edgecolor="white", linewidth=0.5
        )

    ax.set_xlabel("Field")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Mean Field Accuracy Across 3 Runs")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in SCORED_FIELDS], fontsize=9)
    ax.legend()
    ax.set_ylim(0, 110)
    fig.tight_layout()
    fig.savefig(out_dir / "field_accuracy_comparison.png", dpi=150)
    plt.close(fig)


def chart_cost_tokens(summary: dict[str, Any], out_dir: Path):
    """Grouped bar chart: total tokens and cost by method."""
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    method_names = list(summary.keys())
    labels = [METHOD_LABELS.get(m, m) for m in method_names]
    colors = [METHOD_COLORS.get(m, "#999") for m in method_names]

    # Token chart
    input_tokens = []
    output_tokens = []
    for method in method_names:
        it = sum(r["total_input_tokens"] for r in summary[method]["runs"])
        ot = sum(r["total_output_tokens"] for r in summary[method]["runs"])
        input_tokens.append(it)
        output_tokens.append(ot)

    x = np.arange(len(method_names))
    ax1.bar(x, input_tokens, 0.4, label="Input Tokens", color="#3498db", edgecolor="white")
    ax1.bar(x, output_tokens, 0.4, bottom=input_tokens, label="Output Tokens", color="#f39c12", edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Tokens")
    ax1.set_title("Total Tokens by Method")
    ax1.legend()

    # Cost chart
    costs = [summary[m]["total_cost_usd"] for m in method_names]
    bars = ax2.bar(x, costs, 0.4, color=colors, edgecolor="white")
    for bar, val in zip(bars, costs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"${val:.4f}", ha="center", va="bottom", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Cost (USD)")
    ax2.set_title("Total Notional Cost by Method")

    fig.tight_layout()
    fig.savefig(out_dir / "cost_token_comparison.png", dpi=150)
    plt.close(fig)


def chart_latency_boxplot(summary: dict[str, Any], out_dir: Path):
    """Box plot: per-call latency distribution by method."""
    _setup_style()
    fig, ax = plt.subplots()

    method_names = list(summary.keys())
    all_latencies = []
    labels = []
    colors = []
    for method in method_names:
        lats = []
        for r in summary[method]["runs"]:
            lats.extend(r["latencies"])
        all_latencies.append(lats)
        labels.append(METHOD_LABELS.get(method, method))
        colors.append(METHOD_COLORS.get(method, "#999"))

    bp = ax.boxplot(all_latencies, labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Per-Call Latency Distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "latency_boxplot.png", dpi=150)
    plt.close(fig)


def chart_run_spread(summary: dict[str, Any], out_dir: Path):
    """Line chart: overall field accuracy per run for each method."""
    _setup_style()
    fig, ax = plt.subplots()

    method_names = list(summary.keys())
    for method in method_names:
        runs = sorted(summary[method]["runs"], key=lambda r: r["run"])
        x = [r["run"] for r in runs]
        y = [r["overall_accuracy"] * 100 for r in runs]
        ax.plot(
            x, y, marker="o", linewidth=2, markersize=8,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, None),
        )
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9)

    ax.set_xlabel("Run")
    ax.set_ylabel("Overall Field Accuracy (%)")
    ax.set_title("Field Accuracy Spread Across 3 Runs")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Run 1", "Run 2", "Run 3"])
    ax.legend()
    ax.set_ylim(0, 110)
    fig.tight_layout()
    fig.savefig(out_dir / "run_spread.png", dpi=150)
    plt.close(fig)


# ─── Report Generation ───────────────────────────────────────────────────

def generate_report(summary: dict[str, Any], out_dir: Path):
    """Generate WEEK02_REPORT.md with embedded chart references."""
    lines = [
        "# Week 02 — Structured Extraction Benchmark Report",
        "",
        "## Methods Compared",
        "| | Method A (Prompt-Only) | Method B (Grammar-Constrained) |",
        "|---|---|---|",
    ]

    methods = list(summary.keys())
    ma = summary.get("prompt_only", {})
    mb = summary.get("grammar_constrained", {})

    if ma and mb:
        lines.append(
            f"| **Mean Invalid Rate** | {ma['mean_invalid_rate']*100:.1f}% "
            f"(±{ma['std_invalid_rate']*100:.1f}%) | "
            f"{mb['mean_invalid_rate']*100:.1f}% "
            f"(±{mb['std_invalid_rate']*100:.1f}%) |"
        )
        lines.append(
            f"| **Mean Field Accuracy** | {ma['mean_overall_accuracy']*100:.1f}% "
            f"(±{ma['std_overall_accuracy']*100:.1f}%) | "
            f"{mb['mean_overall_accuracy']*100:.1f}% "
            f"(±{mb['std_overall_accuracy']*100:.1f}%) |"
        )
        lines.append(
            f"| **Total Tokens** | {ma['total_tokens']:,} | {mb['total_tokens']:,} |"
        )
        lines.append(
            f"| **Total Cost (USD)** | ${ma['total_cost_usd']:.6f} | ${mb['total_cost_usd']:.6f} |"
        )

    lines += [
        "",
        "## Invalid Rate per Run",
        "![Invalid Rate](charts/invalid_rate_comparison.png)",
        "",
        "## Field Accuracy by Field",
        "![Field Accuracy](charts/field_accuracy_comparison.png)",
        "",
        "## Cost & Token Comparison",
        "![Cost Tokens](charts/cost_token_comparison.png)",
        "",
        "## Latency Distribution",
        "![Latency](charts/latency_boxplot.png)",
        "",
        "## Run Spread (Consistency)",
        "![Run Spread](charts/run_spread.png)",
        "",
    ]

    # Per-run detail tables
    for method in methods:
        label = "Method A (Prompt-Only)" if method == "prompt_only" else "Method B (Grammar-Constrained)"
        lines.append(f"## {label} — Per-Run Details")
        lines.append("")
        lines.append("| Run | Invalid Rate | Overall Accuracy | Tokens | Cost | Mean Latency |")
        lines.append("|---|---|---|---|---|---|")
        for rm in summary[method]["runs"]:
            lines.append(
                f"| {rm['run']} | {rm['invalid_rate']*100:.1f}% | "
                f"{rm['overall_accuracy']*100:.1f}% | "
                f"{rm['total_input_tokens'] + rm['total_output_tokens']:,} | "
                f"${rm['total_cost_usd']:.6f} | {rm['mean_latency_s']:.2f}s |"
            )
        lines.append("")

    report_path = out_dir / "WEEK02_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {report_path}")

    # Generate comparison_report.csv
    import csv
    csv_path = out_dir / "comparison_report.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "method", "run", "total_notices", "valid_json", "schema_valid",
            "invalid_rate_pct", "overall_accuracy_pct", "input_tokens",
            "output_tokens", "total_tokens", "total_cost_usd", "mean_latency_s"
        ])
        for method in methods:
            for rm in summary[method]["runs"]:
                writer.writerow([
                    method,
                    rm["run"],
                    rm["total"],
                    rm["valid_json"],
                    rm["schema_valid"],
                    f"{rm['invalid_rate']*100:.2f}",
                    f"{rm['overall_accuracy']*100:.2f}",
                    rm["total_input_tokens"],
                    rm["total_output_tokens"],
                    rm["total_input_tokens"] + rm["total_output_tokens"],
                    f"{rm['total_cost_usd']:.6f}",
                    f"{rm['mean_latency_s']:.2f}",
                ])
    print(f"CSV report written to {csv_path}")


def run_evaluation(
    extractions_path: Path = EXTRACTIONS_PATH,
    output_dir: Path = OUTPUT_DIR
) -> dict[str, Any]:
    print("Loading data...")
    schema = load_schema()
    answer_key = load_answer_key()
    
    records = []
    with extractions_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} extraction records from {extractions_path}")

    print("Evaluating...")
    summary = evaluate_all(records, schema, answer_key)

    # Save metrics JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics_summary.json"
    summary_serializable = {}
    for method, data in summary.items():
        md = {**data}
        md["runs"] = []
        for r in data["runs"]:
            rd = {k: v for k, v in r.items() if k != "latencies"}
            md["runs"].append(rd)
        summary_serializable[method] = md
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary_serializable, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to {metrics_path}")

    # Generate charts
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    print("Generating charts...")
    chart_invalid_rate(summary, charts_dir)
    chart_field_accuracy(summary, charts_dir)
    chart_cost_tokens(summary, charts_dir)
    chart_latency_boxplot(summary, charts_dir)
    chart_run_spread(summary, charts_dir)
    print(f"Charts saved to {charts_dir}")

    # Generate markdown report
    generate_report(summary, output_dir)

    # Print summary to console
    for method, data in summary.items():
        label = "Prompt-Only" if method == "prompt_only" else "Grammar-Constrained"
        print(f"\n--- {label} ---")
        print(f"  Mean invalid rate:    {data['mean_invalid_rate']*100:.1f}%")
        print(f"  Mean field accuracy:  {data['mean_overall_accuracy']*100:.1f}%")
        print(f"  Total tokens:         {data['total_tokens']:,}")
        print(f"  Total cost:           ${data['total_cost_usd']:.6f}")

    return summary


def main():
    run_evaluation()


if __name__ == "__main__":
    main()
