"""
Extract Spec-Kit artifacts (.specify/{spec,plan,tasks}.md) from each task's
patch.diff in a FeatureBench run directory, for easy manual review.

Usage:
    python -m featurebench.scripts.extract_spec_artifacts <run_dir>
    python -m featurebench.scripts.extract_spec_artifacts runs/2026-05-19__18-00-00

Reads:    <run_dir>/run_outputs/<task_id>/attempt-<n>/patch.diff
Writes:   <run_dir>/spec_artifacts/<task_id>/attempt-<n>/{spec,plan,tasks}.md
          <run_dir>/spec_artifacts/<task_id>/attempt-<n>/review_notes.md
          <run_dir>/spec_artifacts/_adherence.csv

Adherence: a task is "fully adherent" iff all three .specify/*.md files exist
in its patch. The CSV summarizes per-task adherence.
"""

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


SPEC_FILES = ("spec.md", "plan.md", "tasks.md")
SPEC_DIR_PREFIX = ".specify/"

# Unified-diff "diff --git a/<path> b/<path>" header
_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$")
# Hunk header
_HUNK_RE = re.compile(r"^@@ ")


@dataclass
class ExtractedFile:
    path: str           # original repo-relative path, e.g. ".specify/spec.md"
    content: str        # reconstructed file content


def parse_unified_diff(diff_text: str) -> Dict[str, ExtractedFile]:
    """
    Parse a unified diff and reconstruct the *new* (b/) file content for every
    file in the diff.

    This is a best-effort reconstruction that handles the common cases we see
    from `git diff` output in FeatureBench runs:
      - New-file diffs (a/dev/null -> b/<path>): we collect every "+" line.
      - Modified-file diffs: we still collect "+" lines, but the result will
        only contain the added regions, not the full final file. That's fine
        here because .specify/*.md are always new files created by the agent.

    Returns: { path -> ExtractedFile }
    """
    files: Dict[str, ExtractedFile] = {}
    current_path: Optional[str] = None
    current_lines: List[str] = []
    in_hunk = False

    def flush() -> None:
        if current_path is not None:
            files[current_path] = ExtractedFile(
                path=current_path,
                content="".join(current_lines),
            )

    for raw_line in diff_text.splitlines(keepends=True):
        line = raw_line.rstrip("\n")
        m = _DIFF_GIT_RE.match(line)
        if m:
            # Starting a new file block; flush any previous one.
            flush()
            current_path = m.group(2)
            current_lines = []
            in_hunk = False
            continue
        if current_path is None:
            continue
        if _HUNK_RE.match(line):
            in_hunk = True
            continue
        if not in_hunk:
            continue
        # In a hunk: collect added lines.
        if line.startswith("+++") or line.startswith("---"):
            # Header lines inside the file block, not real content.
            continue
        if line.startswith("+"):
            current_lines.append(raw_line[1:])  # strip the leading '+'
        # Context (' ') and deletions ('-') are ignored for reconstruction.
    flush()
    return files


def extract_spec_artifacts(patch_path: Path) -> Dict[str, str]:
    """Return { 'spec.md'|'plan.md'|'tasks.md': content } found in the patch."""
    if not patch_path.exists():
        return {}
    diff_text = patch_path.read_text(encoding="utf-8", errors="replace")
    files = parse_unified_diff(diff_text)
    out: Dict[str, str] = {}
    for path, ef in files.items():
        if not path.startswith(SPEC_DIR_PREFIX):
            continue
        leaf = path[len(SPEC_DIR_PREFIX):]
        if leaf in SPEC_FILES:
            out[leaf] = ef.content
    return out


REVIEW_TEMPLATE = """\
# Review notes — {task_id} (attempt {attempt})

> Fill this in after reading spec.md, plan.md, tasks.md, and the source hunks in patch.diff.

## Adherence
- [ ] spec.md present
- [ ] plan.md present
- [ ] tasks.md present
- [ ] Agent followed spec → plan → tasks → implement order (check first edits in run.log)

## spec.md quality
- [ ] Captures all acceptance criteria from the original task statement
- [ ] Calls out edge cases (not just the happy path)
- [ ] No implementation details leaking in
- Notes:

## plan.md quality
- [ ] Identifies the correct files to modify
- [ ] Names the interfaces / signatures the implementation will use
- [ ] Discusses at least one alternative considered
- Notes:

## tasks.md quality
- [ ] Tasks are atomic (≈ one function / one file each)
- [ ] Ordered (later tasks depend on earlier ones)
- [ ] Each task references a spec.md acceptance criterion
- Notes:

## Plan-implementation drift
- [ ] Source-code hunks in patch.diff match what plan.md committed to
- [ ] If the plan was wrong, was plan.md updated before the code changed?
- Notes:

## Spec-test alignment
- [ ] FAIL_TO_PASS tests look like a reasonable suite given only spec.md
- Notes:

## Failure modes observed
- [ ] Token bloat with no benefit
- [ ] Premature design lock-in
- [ ] Skipped one of the three artifacts
- [ ] Spec rewritten after implementation
- Notes:

## Overall verdict
Vanilla-better / Spec-better / Tie / Both-failed — and why:
"""


def find_attempt_dirs(run_dir: Path) -> List[Path]:
    """Find every attempt-* directory under run_dir/run_outputs/."""
    base = run_dir / "run_outputs"
    if not base.is_dir():
        return []
    return sorted(p for p in base.glob("*/attempt-*") if p.is_dir())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Path to a FeatureBench run directory (contains run_outputs/).")
    parser.add_argument(
        "--no-template",
        action="store_true",
        help="Skip writing review_notes.md per task.",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        print(f"ERROR: run_dir does not exist or is not a directory: {run_dir}", file=sys.stderr)
        return 2

    attempt_dirs = find_attempt_dirs(run_dir)
    if not attempt_dirs:
        print(f"No attempts found under {run_dir}/run_outputs/", file=sys.stderr)
        return 1

    out_root = run_dir / "spec_artifacts"
    out_root.mkdir(parents=True, exist_ok=True)

    adherence_rows: List[Dict[str, object]] = []
    fully_adherent = 0
    any_artifact = 0

    for attempt_dir in attempt_dirs:
        # attempt_dir = run_outputs/<task_id>/attempt-<n>
        task_id = attempt_dir.parent.name
        attempt = attempt_dir.name  # "attempt-1"
        patch = attempt_dir / "patch.diff"

        artifacts = extract_spec_artifacts(patch)
        present = {leaf: (leaf in artifacts) for leaf in SPEC_FILES}

        dest = out_root / task_id / attempt
        if artifacts:
            dest.mkdir(parents=True, exist_ok=True)
            for leaf, content in artifacts.items():
                (dest / leaf).write_text(content, encoding="utf-8")
            if not args.no_template:
                (dest / "review_notes.md").write_text(
                    REVIEW_TEMPLATE.format(task_id=task_id, attempt=attempt),
                    encoding="utf-8",
                )

        all_three = all(present.values())
        if all_three:
            fully_adherent += 1
        if any(present.values()):
            any_artifact += 1

        adherence_rows.append({
            "task_id": task_id,
            "attempt": attempt,
            "spec_md": present["spec.md"],
            "plan_md": present["plan.md"],
            "tasks_md": present["tasks.md"],
            "fully_adherent": all_three,
            "patch_path": str(patch.relative_to(run_dir)) if patch.exists() else "",
        })

    # Write adherence CSV
    csv_path = out_root / "_adherence.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(adherence_rows[0].keys()))
        writer.writeheader()
        writer.writerows(adherence_rows)

    total = len(adherence_rows)
    print(f"Scanned {total} attempts under {run_dir}")
    print(f"  Any .specify/* artifact:  {any_artifact}/{total}")
    print(f"  All three artifacts:       {fully_adherent}/{total}")
    print(f"Extracted to: {out_root}")
    print(f"Adherence CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
