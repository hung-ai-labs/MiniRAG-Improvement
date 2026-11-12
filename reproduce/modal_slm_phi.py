"""Phi-3.5-mini-instruct twin of modal_slm.py: same image, GPU, auth and vLLM flags;
only MODEL and the app name differ.

Serve the paper's SLM on a rented GPU, speaking the OpenAI protocol.

Why this exists: the local Ollama path works but pins the laptop's GPU for
about three hours per run. This is the same model on a rented A10G, reachable
over HTTP, so the machine stays cool and a run takes under an hour.

Why vLLM rather than transformers: minirag/llm/hf.py generates one prompt at a
time (hf.py:150), which is what made the local estimate 10-14 hours before
Ollama. vLLM batches concurrent requests, and it speaks the OpenAI-compatible
protocol that minirag/llm/gemini.py already uses -- so pointing GEMINI_API_BASE
at this URL runs the whole pipeline with no new backend code.

    modal deploy reproduce/modal_slm_phi.py   # one-off, no GPU billed yet
    modal app stop minirag-slm-phi               # when finished

The GPU spins up on the first request and shuts down after SCALEDOWN seconds
of silence, so an idle deployment costs nothing.
"""

import modal

MODEL = "microsoft/Phi-3.5-mini-instruct"   # the paper's other SLM row: 53.29% acc
GPU = "A100-80GB"   # Phi-3.5-mini has no GQA (~390 KB KV/token): on a 24 GB A10G only ~6 requests of ~5.5k tokens fit
PORT = 8000
SCALEDOWN = 300                       # idle seconds before the GPU is released

# Context has to clear the real prompt sizes: measured on this project's index,
# the retrieved context is 3,908 tokens at the median and 8,291 at the maximum,
# before the Entities table and the prompt template are added.
MAX_MODEL_LEN = 32768   # Phi-3.5-mini supports 128k; 32k covers every prompt here

# LongRoPE fix (12/11/2026). vLLM 0.11.0 phi3_long_rope_scaled_rope.py picks short vs long
# factors per forward call with torch.any(positions > 4096): one long sequence in the batch
# switches every token in it, so a sequence's KV cache mixes two position encodings
# depending on its batch-mates (prefix caching and chunked prefill make it worse). The Phi
# index built that way lost 167/516 chunks (85/116 of chunks >= 800 tokens, whose extraction
# prompt crosses 4,096). Setting short_factor = long_factor makes the choice irrelevant;
# llama.cpp (num_ctx > 4096) uses the long factors for every position the same way.
# ROPE_FIX = False reproduces the original server (app minirag-slm-phi).
ROPE_FIX = True
app = modal.App("minirag-slm-phi-ropefix" if ROPE_FIX else "minirag-slm-phi")

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.11.0", "huggingface_hub[hf_transfer]==0.35.3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"})
)

# Weights persist between runs, so only the first cold start pays the download.
hf_cache = modal.Volume.from_name("minirag-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("minirag-vllm-cache", create_if_missing=True)

# The endpoint is public, so it needs a token of its own. Create it once with:
#   modal secret create minirag-slm-key MINIRAG_SLM_KEY=<pick-something-long>
auth = modal.Secret.from_name("minirag-slm-key")


@app.function(
    image=vllm_image,
    gpu=GPU,
    scaledown_window=SCALEDOWN,
    timeout=30 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[auth],
    max_containers=1,
)
@modal.concurrent(max_inputs=32)      # vLLM batches these; this is the whole speed win
@modal.web_server(port=PORT, startup_timeout=15 * 60)
def serve():
    import json
    import os
    import shlex
    import subprocess

    extra = []
    if ROPE_FIX:
        from huggingface_hub import hf_hub_download
        cfg = json.load(open(hf_hub_download(MODEL, "config.json")))
        rs = dict(cfg["rope_scaling"])
        assert rs.get("type", rs.get("rope_type")) in ("longrope", "su") and len(rs["long_factor"]) == len(rs["short_factor"])
        rs["short_factor"] = rs["long_factor"]          # one factor set for every position, as llama.cpp does
        extra = ["--hf-overrides", shlex.quote(json.dumps({"rope_scaling": rs}))]

    subprocess.Popen(" ".join([
        "vllm serve", MODEL,
        "--served-model-name", MODEL, "minirag-slm",
        "--host 0.0.0.0", f"--port {PORT}",
        f"--max-model-len {MAX_MODEL_LEN}",
        "--api-key", os.environ["MINIRAG_SLM_KEY"],
        "--disable-log-requests",
        *extra,
    ]), shell=True)
