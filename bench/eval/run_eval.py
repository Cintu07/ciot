"""Equal-memory head-to-head on this laptop: accuracy and wall-clock time per task.

Starts llama-server once per model, sends the same prompts to every model (ChatML,
thinking pre-closed, greedy), and grades:
  - GSM8K: exact final number on an 'Answer: <number>' line
  - HumanEval: the official unit tests, run in a subprocess (model-written code is
    executed locally in a temp dir with a 30 s timeout; this is not a sandbox)

usage: python bench/eval/run_eval.py --model models/X.gguf --name X [--gsm 60] [--he 40]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SERVER = os.path.join(ROOT, "vendor", "prism-llama.cpp", "build", "bin", "llama-server.exe")
STOP = ["<|im_end|>", "<|endoftext|>"]


def chat(user):
    # thinking pre-closed: the model answers directly, same for every model in the comparison
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def load(name, n):
    with open(os.path.join(HERE, "data", name), encoding="utf-8") as f:
        return [json.loads(line) for line in f][:n]


def post(port, payload, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def wait_ready(port, proc, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("server exited during load")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if json.load(r).get("status") == "ok":
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("server did not become ready")


def last_number(text):
    m = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if not m:
        m = re.findall(r"(-?[\d,]*\.?\d+)", text)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def grade_gsm(item, text):
    gold = float(item["answer"].split("####")[-1].strip().replace(",", ""))
    got = last_number(text)
    return got is not None and abs(got - gold) < 1e-6


def extract_code(text):
    # the closing fence is optional: models often end the turn right after the code
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)", text, re.S)
    if blocks:
        return blocks[0]
    return text


def grade_he(item, text):
    code = extract_code(text)
    imports = "\n".join(l for l in item["prompt"].splitlines() if l.startswith(("import ", "from ")))
    if f"def {item['entry_point']}" not in code:
        code = item["prompt"] + code  # model continued the body only
    program = f"{imports}\n\n{code}\n\n{item['test']}\n\ncheck({item['entry_point']})\n"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(program)
        try:
            r = subprocess.run([sys.executable, path], cwd=d, capture_output=True, timeout=30)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--gsm", type=int, default=60)
    ap.add_argument("--he", type=int, default=40)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--server-args", default="")
    args = ap.parse_args()

    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, f"{args.name}.server.log"), "w", encoding="utf-8")
    cmd = [SERVER, "-m", args.model, "-t", str(args.threads), "-tb", str(args.threads), "--prio", "2", "--no-mmap",
           "-c", "4096", "-np", "1", "--port", str(args.port), "--host", "127.0.0.1"] + args.server_args.split()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=dict(os.environ, CIOT_T2="1"))
    try:
        load_s = wait_ready(args.port, proc)
        print(f"[{args.name}] loaded in {load_s:.0f}s", flush=True)

        tasks = [("gsm8k", it) for it in load("gsm8k_test.jsonl", args.gsm)] + \
                [("humaneval", it) for it in load("humaneval.jsonl", args.he)]
        results = []
        for idx, (kind, item) in enumerate(tasks):
            if kind == "gsm8k":
                prompt = chat(f"{item['question']}\n\nSolve this step by step, briefly. "
                              "End with a final line of the form 'Answer: <number>'.")
                n_predict = 400
            else:
                prompt = chat("Complete this Python function. Reply with only the complete function, including "
                              "its signature and any imports it needs, in one ```python code block.\n\n"
                              f"```python\n{item['prompt']}```")
                n_predict = 512
            t0 = time.perf_counter()
            r = post(args.port, {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "top_k": 1,
                                 "cache_prompt": False, "stop": STOP})
            wall = time.perf_counter() - t0
            text = r.get("content", "")
            ok = grade_gsm(item, text) if kind == "gsm8k" else grade_he(item, text)
            tm = r.get("timings", {})
            rec = {"task": kind, "id": item.get("task_id", idx), "correct": ok, "wall_s": wall,
                   "prompt_n": tm.get("prompt_n"), "prompt_ms": tm.get("prompt_ms"),
                   "gen_n": tm.get("predicted_n"), "gen_ms": tm.get("predicted_ms"),
                   "truncated": r.get("truncated", False) or tm.get("predicted_n", 0) >= n_predict, "text": text}
            results.append(rec)
            print(f"[{args.name}] {idx + 1}/{len(tasks)} {kind} {'ok ' if ok else 'BAD'} {wall:6.1f}s "
                  f"gen {rec['gen_n']} tok", flush=True)

        with open(os.path.join(out_dir, f"{args.name}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        summary = {"name": args.name, "model": os.path.basename(args.model),
                   "model_gb": os.path.getsize(args.model) / 1e9, "load_s": load_s}
        for kind in ("gsm8k", "humaneval"):
            rs = [x for x in results if x["task"] == kind]
            if not rs:
                continue
            n_ok = sum(x["correct"] for x in rs)
            wall = sum(x["wall_s"] for x in rs)
            gen = sum(x["gen_n"] or 0 for x in rs)
            gen_ms = sum(x["gen_ms"] or 0 for x in rs)
            summary[kind] = {"n": len(rs), "correct": n_ok, "accuracy": n_ok / len(rs), "wall_s": wall,
                             "s_per_task": wall / len(rs), "s_per_correct": wall / max(n_ok, 1),
                             "gen_tokens": gen, "decode_tok_s": gen / (gen_ms / 1000) if gen_ms else None,
                             "truncated": sum(x["truncated"] for x in rs)}
        with open(os.path.join(out_dir, f"{args.name}.summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


if __name__ == "__main__":
    main()
