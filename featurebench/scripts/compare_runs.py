"""
Compare two FeatureBench runs (e.g. vanilla vs --spec-mode) on the same tasks.

Joins per-task metrics from:
  - claude_code_stream_output.jsonl  (tokens, cost, duration, num_turns, tool mix)
  - patch.diff                       (source LOC, spec LOC, files touched)
  - report.json (optional)           (resolved / pass_rate from `fb eval`)

Usage:
    python -m featurebench.scripts.compare_runs <vanilla_run_dir> <spec_run_dir>
    python -m featurebench.scripts.compare_runs vanilla/ spec/ \\
        --eval-vanilla vanilla/report.json --eval-spec spec/report.json \\
        --output-prefix compare_2026-05-19

If --eval-* is omitted the script auto-detects <run_dir>/report.json.

Outputs:
  <prefix>.csv  — one row per task, both runs side by side
  <prefix>.md   — aggregates + pairwise (A pass / B pass / both / neither) breakdown

Built for the claude_code agent. Other agents store stream data differently
and will need their own parser.
"""

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STREAM_FILE = "claude_code_stream_output.jsonl"
PATCH_FILE = "patch.diff"
SPEC_DIR_PREFIX = ".specify/"

_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$")


# ---------- stream-json parsing ----------

@dataclass
class StreamMetrics:
    duration_ms: Optional[int] = None
    duration_api_ms: Optional[int] = None
    num_turns: Optional[int] = None
    total_cost_usd: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
    is_error: Optional[bool] = None
    # Tool-use counts across assistant turns: { "Bash": 3, "Edit": 5, ... }
    tool_counts: Dict[str, int] = field(default_factory=dict)

    def total_input(self) -> int:
        return sum(
            v or 0
            for v in (self.input_tokens, self.cache_creation_input_tokens, self.cache_read_input_tokens)
        )

    def cache_hit_rate(self) -> Optional[float]:
        denom = self.total_input()
        if denom <= 0:
            return None
        return (self.cache_read_input_tokens or 0) / denom


def _coerce_int(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _coerce_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def parse_stream_jsonl(path: Path) -> StreamMetrics:
    """Walk Claude Code's stream-json output to collect per-task metrics."""
    metrics = StreamMetrics()
    if not path.exists():
        return metrics

    tool_counts: Counter = Counter()

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            t = obj.get("type")
            if t == "assistant":
                msg = obj.get("message") or {}
                for block in (msg.get("content") or []):
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        name = block.get("name") or "<unknown>"
                        tool_counts[name] += 1
            elif t == "result":
                # Final summary line.
                metrics.duration_ms = _coerce_int(obj.get("duration_ms"))
                metrics.duration_api_ms = _coerce_int(obj.get("duration_api_ms"))
                metrics.num_turns = _coerce_int(obj.get("num_turns"))
                metrics.total_cost_usd = _coerce_float(obj.get("total_cost_usd"))
                metrics.is_error = obj.get("is_error")
                usage = obj.get("usage") or {}
                metrics.input_tokens = _coerce_int(usage.get("input_tokens"))
                metrics.output_tokens = _coerce_int(usage.get("output_tokens"))
                metrics.cache_creation_input_tokens = _coerce_int(
                    usage.get("cache_creation_input_tokens")
                )
                metrics.cache_read_input_tokens = _coerce_int(
                    usage.get("cache_read_input_tokens")
                )

    metrics.tool_counts = dict(tool_counts)
    return metrics


# ---------- patch.diff parsing (lightweight, no full reconstruction) ----------

@dataclass
class PatchStats:
    files_touched: int = 0
    source_added: int = 0       # '+' lines under non-.specify/ paths
    source_removed: int = 0     # '-' lines under non-.specify/ paths
    spec_added: int = 0         # '+' lines under .specify/
    has_spec_md: bool = False
    has_plan_md: bool = False
    has_tasks_md: bool = False


def parse_patch_stats(path: Path) -> PatchStats:
    stats = PatchStats()
    if not path.exists():
        return stats

    current_path: Optional[str] = None
    is_spec_file = False
    touched: set = set()

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _DIFF_GIT_RE.match(raw)
        if m:
            current_path = m.group(2)
            is_spec_file = current_path.startswith(SPEC_DIR_PREFIX)
            touched.add(current_path)
            if current_path == ".specify/spec.md":
                stats.has_spec_md = True
            elif current_path == ".specify/plan.md":
                stats.has_plan_md = True
            elif current_path == ".specify/tasks.md":
                stats.has_tasks_md = True
            continue
        if current_path is None:
            continue
        if raw.startswith("+++") or raw.startswith("---") or raw.startswith("@@"):
            continue
        if raw.startswith("+"):
            if is_spec_file:
                stats.spec_added += 1
            else:
                stats.source_added += 1
        elif raw.startswith("-"):
            if not is_spec_file:
                stats.source_removed += 1

    stats.files_touched = len(touched)
    return stats


# ---------- per-task row ----------

@dataclass
class TaskRow:
    task_id: str
    attempt: str
    stream: StreamMetrics
    patch: PatchStats
    resolved: Optional[bool] = None
    pass_rate: Optional[float] = None
    patch_applied: Optional[bool] = None


def load_eval_report(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """Load fb eval's report.json into { instance_id: {resolved, pass_rate, ...} }."""
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    # report.json shape (per run_evaluation.py): { "instances": {iid: {...}} } OR
    # a flat list under "results". Be defensive.
    if isinstance(data, dict):
        if "instances" in data and isinstance(data["instances"], dict):
            return {k: v for k, v in data["instances"].items()}
        if "results" in data and isinstance(data["results"], list):
            return {item.get("instance_id"): item for item in data["results"] if item.get("instance_id")}
        # Fallback: dict keyed by instance_id directly.
        if all(isinstance(v, dict) for v in data.values()):
            return data
    return {}


def collect_run(run_dir: Path, eval_path: Optional[Path]) -> Dict[Tuple[str, str], TaskRow]:
    """{(task_id, attempt): TaskRow} for a run directory."""
    base = run_dir / "run_outputs"
    if not base.is_dir():
        return {}

    eval_report = load_eval_report(eval_path)
    rows: Dict[Tuple[str, str], TaskRow] = {}

    for attempt_dir in sorted(p for p in base.glob("*/attempt-*") if p.is_dir()):
        task_id = attempt_dir.parent.name
        attempt = attempt_dir.name
        stream = parse_stream_jsonl(attempt_dir / STREAM_FILE)
        patch = parse_patch_stats(attempt_dir / PATCH_FILE)
        # Prefer the per-instance eval report at
        # <run>/eval_outputs/<task_id>/<attempt>/report.json (keyed by instance_id),
        # falling back to the run-level eval report.
        per_instance_report = run_dir / "eval_outputs" / task_id / attempt / "report.json"
        ev: Dict[str, Any] = {}
        if per_instance_report.exists():
            try:
                pdata = json.loads(per_instance_report.read_text(encoding="utf-8"))
                if isinstance(pdata, dict):
                    ev = pdata.get(task_id, {}) or next(iter(pdata.values()), {})
            except Exception:
                ev = {}
        if not ev:
            ev = eval_report.get(task_id, {})
        rows[(task_id, attempt)] = TaskRow(
            task_id=task_id,
            attempt=attempt,
            stream=stream,
            patch=patch,
            resolved=ev.get("resolved"),
            pass_rate=ev.get("pass_rate"),
            patch_applied=ev.get("patch_applied") or ev.get("patch_successfully_applied"),
        )
    return rows


# ---------- aggregation + output ----------

CSV_COLUMNS = [
    "task_id", "attempt",
    # A = first run, B = second run
    "A_resolved", "B_resolved",
    "A_pass_rate", "B_pass_rate",
    "A_duration_ms", "B_duration_ms",
    "A_num_turns", "B_num_turns",
    "A_input_tokens", "B_input_tokens",
    "A_output_tokens", "B_output_tokens",
    "A_cache_read_tokens", "B_cache_read_tokens",
    "A_total_cost_usd", "B_total_cost_usd",
    "A_files_touched", "B_files_touched",
    "A_source_added", "B_source_added",
    "A_source_removed", "B_source_removed",
    "A_spec_added", "B_spec_added",
    "B_has_spec_md", "B_has_plan_md", "B_has_tasks_md",
    "A_tool_mix", "B_tool_mix",
]


def _row_for_csv(task_id: str, attempt: str, a: Optional[TaskRow], b: Optional[TaskRow]) -> Dict[str, Any]:
    def g(r: Optional[TaskRow], path: str) -> Any:
        if r is None:
            return None
        obj: Any = r
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj

    def tool_mix(r: Optional[TaskRow]) -> str:
        if r is None or not r.stream.tool_counts:
            return ""
        return ";".join(f"{k}={v}" for k, v in sorted(r.stream.tool_counts.items()))

    return {
        "task_id": task_id,
        "attempt": attempt,
        "A_resolved": g(a, "resolved"),
        "B_resolved": g(b, "resolved"),
        "A_pass_rate": g(a, "pass_rate"),
        "B_pass_rate": g(b, "pass_rate"),
        "A_duration_ms": g(a, "stream.duration_ms"),
        "B_duration_ms": g(b, "stream.duration_ms"),
        "A_num_turns": g(a, "stream.num_turns"),
        "B_num_turns": g(b, "stream.num_turns"),
        "A_input_tokens": g(a, "stream.input_tokens"),
        "B_input_tokens": g(b, "stream.input_tokens"),
        "A_output_tokens": g(a, "stream.output_tokens"),
        "B_output_tokens": g(b, "stream.output_tokens"),
        "A_cache_read_tokens": g(a, "stream.cache_read_input_tokens"),
        "B_cache_read_tokens": g(b, "stream.cache_read_input_tokens"),
        "A_total_cost_usd": g(a, "stream.total_cost_usd"),
        "B_total_cost_usd": g(b, "stream.total_cost_usd"),
        "A_files_touched": g(a, "patch.files_touched"),
        "B_files_touched": g(b, "patch.files_touched"),
        "A_source_added": g(a, "patch.source_added"),
        "B_source_added": g(b, "patch.source_added"),
        "A_source_removed": g(a, "patch.source_removed"),
        "B_source_removed": g(b, "patch.source_removed"),
        "A_spec_added": g(a, "patch.spec_added"),
        "B_spec_added": g(b, "patch.spec_added"),
        "B_has_spec_md": g(b, "patch.has_spec_md"),
        "B_has_plan_md": g(b, "patch.has_plan_md"),
        "B_has_tasks_md": g(b, "patch.has_tasks_md"),
        "A_tool_mix": tool_mix(a),
        "B_tool_mix": tool_mix(b),
    }


def _mean(xs: List[Any]) -> Optional[float]:
    nums = [float(x) for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return statistics.fmean(nums) if nums else None


def _sum(xs: List[Any]) -> float:
    return sum(float(x) for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool))


def aggregate_markdown(
    rows: List[Dict[str, Any]],
    a_label: str,
    b_label: str,
) -> str:
    n = len(rows)

    def col(name: str) -> List[Any]:
        return [r[name] for r in rows]

    def pass_count(side: str) -> int:
        return sum(1 for r in rows if r[f"{side}_resolved"] is True)

    pass_a = pass_count("A")
    pass_b = pass_count("B")

    # Pairwise outcomes
    both_pass = sum(1 for r in rows if r["A_resolved"] is True and r["B_resolved"] is True)
    only_a = sum(1 for r in rows if r["A_resolved"] is True and r["B_resolved"] is not True)
    only_b = sum(1 for r in rows if r["B_resolved"] is True and r["A_resolved"] is not True)
    neither = sum(1 for r in rows if r["A_resolved"] is not True and r["B_resolved"] is not True)

    def fmt(v: Optional[float], digits: int = 1) -> str:
        if v is None:
            return "—"
        return f"{v:.{digits}f}"

    def agg_for(side: str) -> Dict[str, Optional[float]]:
        return {
            "duration_ms_mean": _mean(col(f"{side}_duration_ms")),
            "turns_mean": _mean(col(f"{side}_num_turns")),
            "input_tokens_sum": _sum(col(f"{side}_input_tokens")),
            "output_tokens_sum": _sum(col(f"{side}_output_tokens")),
            "cache_read_sum": _sum(col(f"{side}_cache_read_tokens")),
            "cost_sum": _sum(col(f"{side}_total_cost_usd")),
            "source_added_mean": _mean(col(f"{side}_source_added")),
            "files_touched_mean": _mean(col(f"{side}_files_touched")),
        }

    a = agg_for("A")
    b = agg_for("B")

    # Tokens per passing task (efficiency).
    def tpp(input_sum: float, output_sum: float, passes: int) -> Optional[float]:
        if passes <= 0:
            return None
        total = (input_sum or 0) + (output_sum or 0)
        if total <= 0:
            return None  # No stream telemetry (e.g. codex agent).
        return total / passes

    tpp_a = tpp(a["input_tokens_sum"], a["output_tokens_sum"], pass_a)
    tpp_b = tpp(b["input_tokens_sum"], b["output_tokens_sum"], pass_b)

    lines: List[str] = []
    lines.append(f"# Run comparison: `{a_label}` (A) vs `{b_label}` (B)")
    lines.append("")
    lines.append(f"- Tasks compared: **{n}**")
    lines.append("")
    lines.append("## Pass rates")
    lines.append("")
    lines.append(f"| | Passed | Pass rate |")
    lines.append(f"|-|-|-|")
    lines.append(f"| A ({a_label}) | {pass_a}/{n} | {fmt((pass_a / n) * 100, 1) if n else '—'}% |")
    lines.append(f"| B ({b_label}) | {pass_b}/{n} | {fmt((pass_b / n) * 100, 1) if n else '—'}% |")
    lines.append("")
    # Mean per-task pass_rate (partial credit). Useful when `resolved` is binary
    # and most tasks fail strict resolution but differ on partial pass_rate.
    pr_a = _mean(col("A_pass_rate"))
    pr_b = _mean(col("B_pass_rate"))
    if pr_a is not None or pr_b is not None:
        lines.append("Mean per-task test pass_rate (partial credit):")
        lines.append("")
        lines.append(f"- A ({a_label}): **{fmt((pr_a or 0) * 100, 1)}%**")
        lines.append(f"- B ({b_label}): **{fmt((pr_b or 0) * 100, 1)}%**")
        lines.append("")
    lines.append("### Pairwise breakdown")
    lines.append("")
    lines.append(f"- Both pass: **{both_pass}**")
    lines.append(f"- Only A passes: **{only_a}**")
    lines.append(f"- Only B passes: **{only_b}**")
    lines.append(f"- Neither passes: **{neither}**")
    lines.append("")
    lines.append("## Cost / tokens / time (sums and means across all tasks)")
    lines.append("")
    lines.append("| Metric | A | B | Δ (B − A) |")
    lines.append("|-|-|-|-|")

    def row(label: str, va: Optional[float], vb: Optional[float], digits: int = 1) -> str:
        delta = (vb - va) if (va is not None and vb is not None) else None
        return f"| {label} | {fmt(va, digits)} | {fmt(vb, digits)} | {fmt(delta, digits)} |"

    lines.append(row("Total cost (USD, sum)", a["cost_sum"], b["cost_sum"], 4))
    lines.append(row("Input tokens (sum)", a["input_tokens_sum"], b["input_tokens_sum"], 0))
    lines.append(row("Output tokens (sum)", a["output_tokens_sum"], b["output_tokens_sum"], 0))
    lines.append(row("Cache-read tokens (sum)", a["cache_read_sum"], b["cache_read_sum"], 0))
    lines.append(row("Duration ms (mean)", a["duration_ms_mean"], b["duration_ms_mean"], 0))
    lines.append(row("Num turns (mean)", a["turns_mean"], b["turns_mean"], 1))
    lines.append(row("Source LOC added (mean)", a["source_added_mean"], b["source_added_mean"], 1))
    lines.append(row("Files touched (mean)", a["files_touched_mean"], b["files_touched_mean"], 1))
    lines.append(row("Tokens-per-passing-task", tpp_a, tpp_b, 0))
    lines.append("")
    lines.append("## Spec-mode adherence (B only)")
    has_spec = sum(1 for r in rows if r["B_has_spec_md"] is True)
    has_plan = sum(1 for r in rows if r["B_has_plan_md"] is True)
    has_tasks = sum(1 for r in rows if r["B_has_tasks_md"] is True)
    all_three = sum(
        1 for r in rows
        if r["B_has_spec_md"] is True and r["B_has_plan_md"] is True and r["B_has_tasks_md"] is True
    )
    lines.append("")
    lines.append(f"- spec.md present: **{has_spec}/{n}**")
    lines.append(f"- plan.md present: **{has_plan}/{n}**")
    lines.append(f"- tasks.md present: **{has_tasks}/{n}**")
    lines.append(f"- All three present: **{all_three}/{n}**")
    lines.append("")
    lines.append("## Tasks where outcomes diverge")
    lines.append("")
    lines.append("| task_id | A | B |")
    lines.append("|-|-|-|")
    for r in rows:
        if r["A_resolved"] != r["B_resolved"]:
            a_mark = "✅" if r["A_resolved"] is True else "❌"
            b_mark = "✅" if r["B_resolved"] is True else "❌"
            lines.append(f"| `{r['task_id']}` | {a_mark} | {b_mark} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_a", type=Path, help="First run dir (treated as 'A' in the report, e.g. vanilla).")
    parser.add_argument("run_b", type=Path, help="Second run dir (treated as 'B' in the report, e.g. spec).")
    parser.add_argument("--eval-vanilla", "--eval-a", dest="eval_a", type=Path, default=None,
                        help="report.json from `fb eval` for run A (default: <run_a>/report.json if present).")
    parser.add_argument("--eval-spec", "--eval-b", dest="eval_b", type=Path, default=None,
                        help="report.json from `fb eval` for run B (default: <run_b>/report.json if present).")
    parser.add_argument("--label-a", default=None, help="Label for run A in the report (default: dir name).")
    parser.add_argument("--label-b", default=None, help="Label for run B in the report (default: dir name).")
    parser.add_argument("--output-prefix", default="compare",
                        help="Prefix for output files (writes <prefix>.csv and <prefix>.md).")
    parser.add_argument("--attempt", default=None,
                        help="Restrict comparison to a single attempt (e.g. 'attempt-1'). Default: all attempts.")
    args = parser.parse_args()

    if not args.run_a.is_dir() or not args.run_b.is_dir():
        print(f"ERROR: one or both run dirs not found: {args.run_a} / {args.run_b}", file=sys.stderr)
        return 2

    eval_a = args.eval_a or (args.run_a / "report.json" if (args.run_a / "report.json").exists() else None)
    eval_b = args.eval_b or (args.run_b / "report.json" if (args.run_b / "report.json").exists() else None)
    if eval_a is None:
        print(f"[warn] no eval report for A ({args.run_a}); pass/fail columns will be blank.", file=sys.stderr)
    if eval_b is None:
        print(f"[warn] no eval report for B ({args.run_b}); pass/fail columns will be blank.", file=sys.stderr)

    rows_a = collect_run(args.run_a, eval_a)
    rows_b = collect_run(args.run_b, eval_b)

    keys = set(rows_a.keys()) | set(rows_b.keys())
    if args.attempt:
        keys = {(t, a) for (t, a) in keys if a == args.attempt}

    if not keys:
        print("ERROR: no overlapping or visible attempts to compare.", file=sys.stderr)
        return 1

    csv_rows: List[Dict[str, Any]] = []
    for (task_id, attempt) in sorted(keys):
        csv_rows.append(_row_for_csv(task_id, attempt, rows_a.get((task_id, attempt)), rows_b.get((task_id, attempt))))

    out_csv = Path(f"{args.output_prefix}.csv")
    out_md = Path(f"{args.output_prefix}.md")

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_rows)

    label_a = args.label_a or args.run_a.name or str(args.run_a)
    label_b = args.label_b or args.run_b.name or str(args.run_b)
    md = aggregate_markdown(csv_rows, label_a, label_b)
    out_md.write_text(md, encoding="utf-8")

    only_a = sum(1 for r in csv_rows if r["A_resolved"] is True and r["B_resolved"] is not True)
    only_b = sum(1 for r in csv_rows if r["B_resolved"] is True and r["A_resolved"] is not True)

    print(f"Compared {len(csv_rows)} (task_id, attempt) pairs.")
    print(f"  Only-A passes: {only_a}   Only-B passes: {only_b}")
    print(f"CSV:      {out_csv}")
    print(f"Markdown: {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
