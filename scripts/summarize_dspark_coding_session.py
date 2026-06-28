#!/usr/bin/env python3
"""Aggregate DSpark coding-session benchmark JSONs.

Reads coding_session_<label>_<timestamp>_run*.json from --out-dir and prints
per-session + cross-session stats for the decode-speed and acceptance signals,
plus the context-length-binned acceptance curve averaged across sessions
(the headline "does the decay reproduce" output).
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import statistics
from pathlib import Path
from typing import Any

SESSION_METRICS = [
    ("mean_acceptance", "session.aggregate.mean_acceptance"),
    ("mean_accepted_per_draft", "session.aggregate.mean_accepted_per_draft"),
    ("mean_tau_est", "session.aggregate.mean_tau_est"),
    ("mean_decode_tok_s", "session.aggregate.mean_decode_tok_s"),
    ("effective_session_tok_s", "session.aggregate.effective_session_tok_s"),
    ("total_generation_tokens", "session.aggregate.total_generation_tokens"),
    ("session_wall_s", "session.aggregate.session_wall_s"),
    ("turns_completed", "session.aggregate.turns_completed"),
]


def nested_get(d: dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def stat(values: list[float]) -> dict[str, Any]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0}
    m = statistics.fmean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {
        "n": len(vals),
        "mean": m,
        "stdev": s,
        "cv": (abs(s / m) if m else 0.0),
        "min": min(vals),
        "max": max(vals),
    }


def merge_bucket_curves(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average acceptance/tau per context bucket across sessions."""
    acc: dict[str, dict[str, list[float]]] = {}
    for s in sessions:
        for b in (s.get("session", {}).get("acceptance_by_context_bucket") or []):
            name = b.get("context_bucket")
            if not name:
                continue
            slot = acc.setdefault(name, {"acceptance": [], "apd": []})
            if b.get("mean_acceptance") is not None:
                slot["acceptance"].append(b["mean_acceptance"])
            if b.get("mean_accepted_per_draft") is not None:
                slot["apd"].append(b["mean_accepted_per_draft"])
    order = ["0-2k", "2k-4k", "4k-8k", "8k-16k", "16k+"]
    out = []
    for name in order:
        if name not in acc:
            continue
        slot = acc[name]
        out.append(
            {
                "context_bucket": name,
                "sessions": len(slot["acceptance"]),
                "mean_acceptance": statistics.fmean(slot["acceptance"]) if slot["acceptance"] else None,
                "mean_accepted_per_draft": statistics.fmean(slot["apd"]) if slot["apd"] else None,
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="experiments/dspark-benchmarks")
    p.add_argument("label")
    p.add_argument("--timestamp", default=None, help="If omitted, use the latest matching run set.")
    p.add_argument("--json", default="")
    args = p.parse_args()

    pattern = f"coding_session_{args.label}_{args.timestamp or '*'}_run*.json"
    files = sorted(_glob.glob(str(Path(args.out_dir) / pattern)))
    if not files:
        raise SystemExit(f"No session JSONs matched {pattern} in {args.out_dir}")
    sessions = [json.loads(Path(f).read_text()) for f in files]

    print(f"# coding-session summary: label={args.label}  runs={len(sessions)}")
    print(f"# files:")
    for f in files:
        print(f"#   {f}")
    print()

    # per-session headline
    hdr = ["run", "turns", "accept", "acc/draft", "tau", "decode_tok_s", "eff_tok_s", "wall_s", "gen_tok"]
    print("{:>3} {:>5} {:>7} {:>9} {:>5} {:>10} {:>9} {:>7} {:>8}".format(*hdr))
    per_run: dict[str, list[float]] = {name: [] for name, _ in SESSION_METRICS}
    for i, s in enumerate(sessions, 1):
        agg = s.get("session", {}).get("aggregate", {}) or {}
        row = [
            i,
            agg.get("turns_completed"),
            agg.get("mean_acceptance"),
            agg.get("mean_accepted_per_draft"),
            agg.get("mean_tau_est"),
            agg.get("mean_decode_tok_s"),
            agg.get("effective_session_tok_s"),
            agg.get("session_wall_s"),
            agg.get("total_generation_tokens"),
        ]
        print(
            "{:>3} {:>5} {:>7.3f} {:>9.2f} {:>5.2f} {:>10.2f} {:>9.2f} {:>7.1f} {:>8.0f}".format(
                row[0],
                row[1] if row[1] is not None else "-",
                row[2] if row[2] is not None else float("nan"),
                row[3] if row[3] is not None else float("nan"),
                row[4] if row[4] is not None else float("nan"),
                row[5] if row[5] is not None else float("nan"),
                row[6] if row[6] is not None else float("nan"),
                row[7] if row[7] is not None else float("nan"),
                row[8] if row[8] is not None else 0,
            )
        )
        for name, dotted in SESSION_METRICS:
            v = nested_get(s, dotted)
            if v is not None:
                per_run[name].append(float(v))

    print("\n# aggregate across runs (mean / stdev / CV):")
    agg_out: dict[str, Any] = {}
    for name, _ in SESSION_METRICS:
        st = stat(per_run[name])
        if st.get("n"):
            print(f"  {name:>26}: {st['mean']:.4f}  ±{st['stdev']:.4f}  CV={st['cv']*100:.1f}%  "
                  f"[{st['min']:.4f},{st['max']:.4f}]  n={st['n']}")
            agg_out[name] = st

    curve = merge_bucket_curves(sessions)
    print("\n# acceptance by context bucket (averaged across runs) — the decay curve:")
    print("  {:>10} {:>8} {:>12} {:>18}".format("bucket", "sessions", "mean_accept", "mean_acc/draft(tau-1)"))
    for b in curve:
        ma = b["mean_acceptance"]
        mp = b["mean_accepted_per_draft"]
        print("  {:>10} {:>8} {:>12} {:>18}".format(
            b["context_bucket"], b["sessions"],
            f"{ma:.3f}" if ma is not None else "-",
            f"{mp:.2f}" if mp is not None else "-",
        ))

    out = {"label": args.label, "files": files, "aggregate": agg_out, "acceptance_by_context_bucket": curve}
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
