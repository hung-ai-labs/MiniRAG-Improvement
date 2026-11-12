"""Bản vá LongRoPE cho Phi-3 / Phi-3.5 khi tự chạy vLLM (không qua Modal).

modal_slm_phi.py đã bật sẵn bản vá này (ROPE_FIX = True). Dùng file này khi phục vụ Phi trên
GPU của mình, cho dataset bất kỳ:

    python -m reproduce.phi_longrope_fix                       # in ra lệnh vllm serve đã vá
    python -m reproduce.phi_longrope_fix --model microsoft/Phi-3-mini-128k-instruct

Lỗi (vLLM 0.11.0, phi3_long_rope_scaled_rope.py): chọn hệ số short hay long cho CẢ batch bằng
torch.any(positions > original_max_position_embeddings). Khi nhiều request chạy song song, KV
cache của một chuỗi bị trộn hai kiểu mã hoá vị trí, và output hỏng. A/B trên 96 chunk với 32
request đồng thời: server gốc 96/96 chunk rỗng, server đã vá 1/96
(review/phi_rope_ab_20261112/summary.json).

Cách vá: short_factor = long_factor, tức một bộ hệ số cho mọi vị trí, giống llama.cpp/ollama khi
num_ctx > 4096. Chỉ đổi cấu hình phục vụ, không đổi trọng số, prompt hay pipeline.
Sau khi index xong, chạy `python -m reproduce.index_health <workdir>` để chắc index không bị thưa.
"""
import argparse
import json
import shlex


def rope_overrides(model):
    """JSON cho --hf-overrides, hoặc None nếu model không dùng LongRoPE (Qwen, Llama, ...)."""
    from huggingface_hub import hf_hub_download
    cfg = json.load(open(hf_hub_download(model, 'config.json')))
    rs = cfg.get('rope_scaling') or {}
    if rs.get('type', rs.get('rope_type')) not in ('longrope', 'su'):
        return None
    assert len(rs['long_factor']) == len(rs['short_factor'])
    return json.dumps({'rope_scaling': {**rs, 'short_factor': rs['long_factor']}})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='microsoft/Phi-3.5-mini-instruct')
    ap.add_argument('--max-model-len', type=int, default=32768)
    a = ap.parse_args()
    o = rope_overrides(a.model)
    cmd = ['vllm', 'serve', a.model, '--served-model-name', a.model, 'minirag-slm',
           '--max-model-len', str(a.max_model_len)]
    if o:
        cmd += ['--hf-overrides', o]
    else:
        print(f'# {a.model} không dùng LongRoPE: không cần vá')
    print(' '.join(shlex.quote(x) for x in cmd))


if __name__ == '__main__':
    main()
