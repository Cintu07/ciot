"""Compare native greedy token IDs from stock and candidate CPU builds."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench" / "eval"))
from run_eval import chat

PROMPTS = [
    "Explain what a CPU memory bus does in one sentence.",
    "What is 17 + 25? Give a short explanation and the answer.",
    "Write a Python function that adds two integers. Reply with code only.",
]


def runtime_hashes(tool):
    return {name: hashlib.sha256((tool.parent / name).read_bytes()).hexdigest()
            for name in ("llama.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll")
            if (tool.parent / name).is_file()}


def capture(tool, model, threads, out, name, count):
    records = []
    for index, text in enumerate(PROMPTS):
        prompt_file = out / f"prompt-{index}.txt"
        prompt_file.write_text(chat(text), encoding="utf-8")
        command = [str(tool), str(model), str(prompt_file.resolve()), str(count), str(threads)]
        with (out / f"{name}-tokens-{index}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=log, timeout=600)
        if result.returncode:
            raise RuntimeError(f"{name} native token check failed ({result.returncode}); inspect logs")
        response = json.loads(result.stdout)
        tokens = response.get("tokens")
        if not tokens or not all(isinstance(t, int) for t in tokens):
            raise RuntimeError(f"{name}: missing nonempty token IDs")
        records.append({"prompt": chat(text), "response": response})
        (out / f"{name}-tokens.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock-tool", type=Path, required=True)
    ap.add_argument("--candidate-tool", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=32)
    args = ap.parse_args()
    if not 1 <= args.tokens <= 128 or not 1 <= args.threads <= 8:
        ap.error("tokens must be 1..128 and threads 1..8")
    for path in (args.stock_tool, args.candidate_tool, args.model):
        if not path.is_file():
            ap.error(f"missing file: {path}")
    args.out.mkdir(parents=True, exist_ok=True)
    stock = capture(args.stock_tool.resolve(), args.model.resolve(), args.threads, args.out, "stock", args.tokens)
    candidate = capture(args.candidate_tool.resolve(), args.model.resolve(), args.threads, args.out, "candidate", args.tokens)
    matches = [a["response"]["tokens"] == b["response"]["tokens"] for a, b in zip(stock, candidate)]
    report = {"all_match": all(matches), "prompts": len(matches), "matches": matches,
              "compared_generated_tokens": [len(r["response"]["tokens"]) for r in stock],
              "scope": "Exact token IDs for these three prompts only; not a universal logit-equivalence proof",
              "model": args.model.name, "model_bytes": args.model.stat().st_size,
              "tool_sha256": {"stock": hashlib.sha256(args.stock_tool.read_bytes()).hexdigest(),
                              "candidate": hashlib.sha256(args.candidate_tool.read_bytes()).hexdigest()},
              "runtime_sha256": {"stock": runtime_hashes(args.stock_tool),
                                 "candidate": runtime_hashes(args.candidate_tool)}}
    (args.out / "token-check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["all_match"]:
        raise SystemExit("Greedy token IDs differ; inspect saved outputs before submission.")


if __name__ == "__main__":
    main()
