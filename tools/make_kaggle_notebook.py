"""Builds kaggle/ciot_equal_memory_eval.ipynb from the same eval scripts used on the laptop."""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FORK_COMMIT = "6bfcd79a2d426abcd2b50e3c2d09ae2225e70a17"


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# CIOT-V2: equal-memory head-to-head

Same prompts and grading as the laptop run (GSM8K exact answer, HumanEval unit tests), thinking pre-closed, greedy.

| model | file | size |
|---|---|---|
| Ternary Bonsai 2 27B | PQ2_0 | 7.21 GB |
| Qwen3.8-27B (same base model, ordinary 2-bit) | UD-IQ2_XXS | 7.27 GB |
| Qwen3-8B | Q6_K | 6.73 GB |

**Before running:** Settings → Accelerator **GPU T4 x2**, Internet **On**. Then **Save Version → Save & Run All (Commit)** so it
runs in the background (about 3 hours). Download `ciot_results.zip` from the Output tab when it finishes.
"""),
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
!nproc; free -g | head -2; df -h /tmp /kaggle/working | tail -2
"""),
    code(f"""
%%bash
# PrismML's llama.cpp fork, pinned to the commit the laptop numbers use, built for the T4 (sm_75)
set -e
mkdir -p /tmp/ciot/eval && cd /tmp/ciot
if [ ! -x llama.cpp/build/bin/llama-server ]; then
  git clone --quiet --filter=blob:none -b prism https://github.com/PrismML-Eng/llama.cpp
  cd llama.cpp && git checkout --quiet {FORK_COMMIT}
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF \\
        -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF > cmake.log 2>&1 || (tail -40 cmake.log; exit 1)
  cmake --build build --target llama-server -j $(nproc) > build.log 2>&1 || (tail -40 build.log; exit 1)
fi
ls -la /tmp/ciot/llama.cpp/build/bin/llama-server
"""),
    code("%%writefile /tmp/ciot/eval/fetch_data.py\n" + read("bench/eval/fetch_data.py")),
    code("""
import os
os.makedirs("/tmp/ciot/eval", exist_ok=True)
!python /tmp/ciot/eval/fetch_data.py
"""),
    code("%%writefile /tmp/ciot/eval/run_eval.py\n" + read("bench/eval/run_eval.py")),
    code("""
import os, subprocess, sys
from huggingface_hub import hf_hub_download

GSM, HE = 250, 164   # GSM8K problems (of 1319) and HumanEval problems (all 164)
OUT = "/kaggle/working/results"
SERVER = "/tmp/ciot/llama.cpp/build/bin/llama-server"
MODELS = [
    ("bonsai2-27b-pq2_0",   "prism-ml/Ternary-Bonsai-2-27B-gguf", "Ternary-Bonsai-2-27B-PQ2_0.gguf"),
    ("qwen3.8-27b-iq2_xxs", "unsloth/Qwen3.8-27B-GGUF",           "Qwen3.8-27B-UD-IQ2_XXS.gguf"),
    ("qwen3-8b-q6_k",       "Qwen/Qwen3-8B-GGUF",                 "Qwen3-8B-Q6_K.gguf"),
]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")  # every model fits on one T4

for name, repo, fname in MODELS:
    if os.path.exists(f"{OUT}/{name}.summary.json"):
        print(f"{name}: already done"); continue
    path = hf_hub_download(repo, fname, local_dir="/tmp/ciot/models")
    cmd = [sys.executable, "-u", "/tmp/ciot/eval/run_eval.py", "--model", path, "--name", name, "--gsm", str(GSM),
           "--he", str(HE), "--gpu", "--server", SERVER, "--out", OUT]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env) as p:
        for line in p.stdout:
            print(line, end="")
    os.remove(path)  # one model on disk at a time
"""),
    code("""
import glob, json, shutil
rows = [json.load(open(f)) for f in sorted(glob.glob("/kaggle/working/results/*.summary.json"))]
print("| model | GB | GSM8K | HumanEval | gen tokens per task | T4 decode tok/s |")
print("|---|---|---|---|---|---|")
for r in rows:
    g, h = r.get("gsm8k", {}), r.get("humaneval", {})
    tasks = g.get("n", 0) + h.get("n", 0)
    gen = (g.get("gen_tokens", 0) + h.get("gen_tokens", 0)) / max(tasks, 1)
    print(f"| {r['name']} | {r['model_gb']:.2f} | {g.get('correct')}/{g.get('n')} ({100*g.get('accuracy',0):.1f}%) | "
          f"{h.get('correct')}/{h.get('n')} ({100*h.get('accuracy',0):.1f}%) | {gen:.0f} | {g.get('decode_tok_s') or 0:.1f} |")
shutil.make_archive("/kaggle/working/ciot_results", "zip", "/kaggle/working/results")
print("saved /kaggle/working/ciot_results.zip")
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(ROOT, "kaggle", "ciot_equal_memory_eval.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, indent=1)
print(out)
