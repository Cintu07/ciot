# RTX 5060 saved-answer comparison

Official PrismML CUDA runtime; this evaluates model choices on an 8 GB GPU. CIOT ARM kernel speedups are a separate experiment.

Saved subset: up to the first 60 GSM8K and first 40 HumanEval tasks, greedy, thinking pre-closed. This is a small deterministic subset, not a full benchmark score.

File sizes are not equal: Bonsai 5.95 GB, Qwen 27B 7.27 GB, Qwen 8B 6.73 GB. The constraint is the same GPU capacity; CPU offload differs.

Timing includes historical and resumed sessions. Historical power is unrecorded in answer files; the handoff says Bonsai ran on AC. The Qwen 27B resumed session crossed from battery to AC; Qwen 8B records power per task. Do not treat mixed-power timing ratios as a controlled speed comparison.

| model | task | correct | mean generated tokens | seconds/task | decode tok/s | power labels |
|---|---|---|---|---|---|---|
| bonsai2-27b-ptq1_0 | gsm8k | 59/60 | 197.3 | 5.64 | 37.6 | unrecorded |
| bonsai2-27b-ptq1_0 | humaneval | 38/40 | 159.5 | 4.84 | 36.8 | unrecorded |
| qwen3-8b-q6_k | gsm8k | 56/60 | 141.4 | 6.62 | 22.0 | ac |
| qwen3-8b-q6_k | humaneval | 38/40 | 154.7 | 7.72 | 20.6 | ac |
| qwen3.8-27b-iq2_xxs | gsm8k | 53/60 | 204.2 | 45.01 | 4.7 | unrecorded |
| qwen3.8-27b-iq2_xxs | humaneval | 37/40 | 164.4 | 32.44 | 5.3 | mixed-session, unrecorded |

Paired accuracy on shared task IDs:

| comparison | task | shared tasks | both correct | Bonsai only | other only | both wrong |
|---|---|---|---|---|---|---|
| Bonsai vs qwen3-8b-q6_k | gsm8k | 60 | 56 | 3 | 0 | 1 |
| Bonsai vs qwen3-8b-q6_k | humaneval | 40 | 37 | 1 | 1 | 1 |
| Bonsai vs qwen3.8-27b-iq2_xxs | gsm8k | 60 | 53 | 6 | 0 | 1 |
| Bonsai vs qwen3.8-27b-iq2_xxs | humaneval | 40 | 37 | 1 | 0 | 2 |

Saved incorrect answers (inspect the JSONL text before interpreting failures):

- bonsai2-27b-ptq1_0: gsm8k/37, humaneval/HumanEval/32, humaneval/HumanEval/38
- qwen3-8b-q6_k: gsm8k/12, gsm8k/21, gsm8k/37, gsm8k/40, humaneval/HumanEval/26, humaneval/HumanEval/32
- qwen3.8-27b-iq2_xxs: gsm8k/2, gsm8k/8, gsm8k/12, gsm8k/13, gsm8k/34, gsm8k/37, gsm8k/39, humaneval/HumanEval/9, humaneval/HumanEval/32, humaneval/HumanEval/38
