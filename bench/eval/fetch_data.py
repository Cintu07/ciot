"""Download the eval sets once into bench/eval/data as JSONL (stdlib only).

GSM8K test (openai/gsm8k, main) and HumanEval (openai/openai_humaneval), via the
Hugging Face datasets-server rows API.
"""
import json
import os
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def rows(dataset, config, split, total):
    out = []
    while len(out) < total:
        q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                    "offset": len(out), "length": min(100, total - len(out))})
        with urllib.request.urlopen(f"https://datasets-server.huggingface.co/rows?{q}", timeout=60) as r:
            batch = json.load(r)["rows"]
        if not batch:
            break
        out.extend(b["row"] for b in batch)
    return out


def save(name, items):
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{path}: {len(items)} items")


if __name__ == "__main__":
    save("gsm8k_test.jsonl", rows("openai/gsm8k", "main", "test", 1319))
    save("humaneval.jsonl", rows("openai/openai_humaneval", "openai_humaneval", "test", 164))
