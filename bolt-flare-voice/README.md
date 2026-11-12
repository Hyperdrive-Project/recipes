# Nemotron 3.5 Bolt on Flare: real-time voice on one DGX Spark

NVIDIA Nemotron 3.5 Bolt (74B total, ~9B active; hybrid Mamba / attention / MoE, 88 layers, NVIDIA's own 2-bit expert
packing, 24 GB) served by NVIDIA Flare on **one DGX Spark**, tuned for short spoken turns. This is **Hyperdrive Local
Voice**, the brain of a Reachy Mini
desk robot: speech-to-text, Bolt and text-to-speech all run locally on the same Spark.

For voice, the number that matters is how fast the robot starts talking, not peak tok/s. That's what this recipe optimizes.

## Results

Cached voice turn, reasoning off, one DGX Spark, same model and drafter. The baseline is the stock Flare recipe
for Bolt.

| Metric | Stock recipe | Hyperdrive | How |
|---|---|---|---|
| **First token** | 0.219 s | **0.084 s** (2.6x faster) | speech package (2 users, 16 speculative rows), throughput scheduler, one-step short prompt tails |
| **First phrase** (what the robot speaks first) | 0.59 s | **~0.30 s** (~2x faster) | same |
| First turn of a session | – | **0.081 s** | the persona prompt is warmed into the prefix cache at boot |
| **Decode** | 37.9 tok/s | **60.8 tok/s** (+60%) | fixed-width verify with 4 drafts per round |
| 8th token | – | 217 ms | verify width 4 (was 230 ms at 8 drafts) |
| AIPerf 4K in / 1K out, 1 user | 71.6 tok/s | 73.6 tok/s | |

Quality is unchanged. Reasoning-on gates: math 48/50, code 27-29/30, tools 11/11. lm-eval: GSM8K 0.956, HumanEval
0.970, needle 15/15 up to 120K, the same as the stock recipe. Measured 2026-10-24 to 10-26 PDT.

**Verify width** (60 held-out voice sessions, T=0.7, [`results/verify-width/`](results/verify-width/)): 4 drafts per
round is the best for voice, at 60.8 tok/s and the 5th token at 171 ms, vs 58.8 tok/s and 179 ms at the default.
Adaptive verification is worse for voice (54.5 tok/s).

What this recipe is not: the fastest raw Bolt tok/s on a Spark. Other public builds reach higher single-stream decode on
long outputs. This recipe trades that for the lowest time to the first spoken word, which is what a voice turn feels like.

## What we changed in Flare

The patch series in [`patches/`](patches/) applies on top of upstream
[NVIDIA/flare-local-inference](https://github.com/NVIDIA/flare-local-inference):

- **`--prefill-short-tail N [--prefill-short-tail-keep-last]`**: prefill a short new prompt tail (a voice turn) in one
  step instead of chunking it. Keep-last is chosen automatically for reasoning turns so the system-prompt checkpoint survives.
- **`FLARE_LILICORR_DRAFT_DIR`**: point the export at a local LiLiCorr drafter in the released layout.
- **Copied proposals for LiLiCorr rounds** (`--history-drafts` now works for Bolt) and **`--history-seed PATH`**: preload
  decoded history with a phrase corpus, so a persona's stock phrases ("Right away.") are drafted by copy the first time
  they come up. *Built, not yet measured.*
- A rebase onto upstream 3267c054, with an autotune-tier shim for the pinned FlashInfer.

| Series | Base | Patches | Status |
|---|---|---|---|
| [`patches/flare-bolt-robot/`](patches/flare-bolt-robot/) | upstream `d5fd9824` | 6 | the build behind every number above |
| [`patches/flare-bolt-robot-next/`](patches/flare-bolt-robot-next/) | upstream `3267c054` | 9 | rebased + history drafts and seed; not yet measured |
| [`patches/model-optimizer/`](patches/model-optimizer/) | NVIDIA/Model-Optimizer `37888814` | 2 | for the drafter fine-tune experiment below |

## Run it

Pins: model `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` rev `c6cc2e8f`, drafter
`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr` rev `abd47806`, CUDA 13.0, one DGX Spark (GB10, 128 GB).

```bash
# 1. Build Flare with the patch series
git clone https://github.com/NVIDIA/flare-local-inference ~/flare-bolt && cd ~/flare-bolt
git checkout -b bolt-voice d5fd9824593c2a70f6bdc001b39f16b4c5b3b0b4
git am /path/to/recipes/bolt-flare-voice/patches/flare-bolt-robot/*.patch
uv sync --locked --extra prepare --extra test --python 3.12
cargo build --release --manifest-path flare/Cargo.toml

# 2. Export the speech package: 2 concurrent requests, 16 speculative rows
#    (edit serve/flare-config.toml paths first; the dev signing key and trust store live next to it, in $FX)
.venv/bin/ignition export nemotron-bolt-24gb --config "$FX/flare-config.toml" --gpus 0 --max-concurrency 2 \
  --speculative-token-capacity 16 --output "$FX/packages/bolt-lilicorr-c2-speech.safetensors"

# 3. Serve (throughput scheduler, 60G memory limit, one-step short tails, verify 4, persona warm-up)
FL=~/flare-bolt FX=$FX /path/to/recipes/bolt-flare-voice/serve/serve-bolt.sh
```

Send chat turns with `"chat_template_kwargs": {"enable_thinking": false}`. Set `WARM_PROMPT` to the exact system prompt
your client sends; the default is [`serve/persona.txt`](serve/persona.txt), a short, spoken-style robot persona. If Ray
workers load the weights, raise their locked-memory limit (`prlimit --memlock=16332894208`) for Flare's io_uring reader.

**Sharing the GPU with speech models:** leave `MPS=0`. With CUDA MPS the first audio from TTS gets faster, but Bolt's first
clause slows down 1.4-4.6x. A time-division gate (TTS speaks while Bolt waits) plus clause-then-continue works better.

## Benchmarks

All probes talk to the OpenAI-compatible endpoint on `127.0.0.1:8000`:

| Script | Measures |
|---|---|
| [`benchmarks/robot_turns.py`](benchmarks/robot_turns.py) | multi-turn robot sessions: first token, first phrase, decode, per-turn stats |
| [`benchmarks/firstk.py`](benchmarks/firstk.py) | time to the 1st / 5th / 8th token on voice turns |
| [`benchmarks/warmcheck.py`](benchmarks/warmcheck.py) | first turn of a fresh session (is the persona cached?) |
| [`benchmarks/turn_ttft_nostream.py`](benchmarks/turn_ttft_nostream.py) | non-streaming turn latency |
| [`benchmarks/decodegap.py`](benchmarks/decodegap.py) | decode round cost and gaps between rounds |

Quality uses the shared suite in [`../correctness/`](../correctness/) (reasoning on).

## What didn't help

- **Fine-tuning the LiLiCorr drafter on voice turns.** Bolt answered 8,000 synthetic voice sessions (40K replies), and we
  trained the released drafter on them for one epoch with ModelOpt. Training accuracy rose from 24% to 33%, but
  acceptance on held-out voice didn't move: 2.86 -> 2.83 tokens per round. Before / after numbers are in
  [`results/drafter-ft/`](results/drafter-ft/). The likely cause is that it adapted to vLLM's hidden states rather than
  to the voice domain. The released drafter stays.
- **MPS** (above) and **adaptive verification** (54.5 vs 60.8 tok/s).

## Layout

```
patches/      Flare patch series (robot build + next) and the ModelOpt patches for the drafter experiment
serve/        serve-bolt.sh, flare-config.toml, persona.txt
benchmarks/   voice-latency probes
results/      verify-width sweep (verify-width/) and drafter fine-tune before/after (drafter-ft/)
```

Flare, Nemotron and DGX Spark are NVIDIA's. This recipe is independent and not endorsed by NVIDIA.
