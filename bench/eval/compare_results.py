"""Compare saved answers on matching task IDs, including interrupted runs."""
import argparse
import json
from pathlib import Path


def read_results(folder):
    models = {}
    for path in sorted(folder.glob("*.jsonl")):
        if ".stopped-" in path.name:
            continue
        name = path.name[:-6]
        if name.endswith(".partial"):
            name = name[:-8]
        # A completed file takes precedence over its checkpoint.
        if name in models and ".partial." in path.name:
            continue
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        keyed = {(r["task"], str(r["id"])): r for r in records}
        if len(keyed) != len(records):
            raise ValueError(f"duplicate task IDs in {path}")
        models[name] = keyed
    return models


def report(folder):
    models = read_results(folder)
    counts = {task: max((sum(key[0] == task for key in rows) for rows in models.values()), default=0)
              for task in ("gsm8k", "humaneval")}
    lines = ["# RTX 5060 saved-answer comparison", "",
        "Official PrismML CUDA runtime; this evaluates model choices on an 8 GB GPU. "
        "CIOT ARM kernel speedups are a separate experiment.", "",
        f"Saved subset: up to the first {counts['gsm8k']} GSM8K and first {counts['humaneval']} HumanEval tasks, greedy, thinking pre-closed. "
        "This is a small deterministic subset, not a full benchmark score.", "",
        "File sizes are not equal: Bonsai 5.95 GB, Qwen 27B 7.27 GB, Qwen 8B 6.73 GB. "
        "The constraint is the same GPU capacity; CPU offload differs.", "",
        "Timing includes historical and resumed sessions. Historical power is unrecorded in "
        "answer files; the handoff says Bonsai ran on AC. The Qwen 27B resumed session "
        "crossed from battery to AC; Qwen 8B records power per task. "
        "Do not treat mixed-power timing ratios as a controlled speed comparison.", "",
        "| model | task | correct | mean generated tokens | seconds/task | decode tok/s | power labels |",
        "|---|---|---|---|---|---|---|"]
    for name, keyed in models.items():
        for task in ("gsm8k", "humaneval"):
            rs = [r for (kind, _), r in keyed.items() if kind == task]
            if not rs:
                continue
            n = len(rs)
            gen = sum(r.get("gen_n") or 0 for r in rs)
            ms = sum(r.get("gen_ms") or 0 for r in rs)
            speed = f"{gen * 1000 / ms:.1f}" if ms else "unknown"
            power = ", ".join(sorted({"mixed-session" if r.get("session_power_change") else
                r.get("power_state", "unrecorded") for r in rs}))
            lines.append(f"| {name} | {task} | {sum(r['correct'] for r in rs)}/{n} | "
                f"{gen/n:.1f} | {sum(r['wall_s'] for r in rs)/n:.2f} | {speed} | {power} |")
    baseline = models.get("bonsai2-27b-ptq1_0", {})
    lines += ["", "Paired accuracy on shared task IDs:", "",
        "| comparison | task | shared tasks | both correct | Bonsai only | other only | both wrong |",
        "|---|---|---|---|---|---|---|"]
    for name, other in models.items():
        if name == "bonsai2-27b-ptq1_0":
            continue
        for task in ("gsm8k", "humaneval"):
            keys = [key for key in baseline.keys() & other.keys() if key[0] == task]
            counts = [0, 0, 0, 0]
            for key in keys:
                a, b = bool(baseline[key]["correct"]), bool(other[key]["correct"])
                counts[(0 if a else 2) + (0 if b else 1)] += 1
            if keys:
                lines.append(f"| Bonsai vs {name} | {task} | {len(keys)} | {counts[0]} | "
                    f"{counts[1]} | {counts[2]} | {counts[3]} |")
    lines += ["", "Saved incorrect answers (inspect the JSONL text before interpreting failures):", ""]
    for name, keyed in models.items():
        bad = [f"{task}/{idx}" for (task, idx), rec in keyed.items() if not rec["correct"]]
        lines.append(f"- {name}: {', '.join(bad) or 'none'}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path(__file__).parent / "results-rtx")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    text = report(args.results)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)
