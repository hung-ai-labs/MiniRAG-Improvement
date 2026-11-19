"""Zero-shot cross-encoder logits on a rented GPU (row 4, +CE; see review/mhr_ce_20261116/PROTOCOL.md).

Same call as reproduce/evidence_set/quick_scorer.py: CrossEncoder(max_length=512) and the raw logit
(Identity activation), fp32. Chunk texts travel once, inside the image (chunks_text.json written by
reproduce.mhr_ce.scores); each call sends a few questions and their pool chunk ids.
Ephemeral app: started and stopped by reproduce.mhr_ce.scores, nothing stays deployed.
"""
import json
import modal

CE_MODEL = 'cross-encoder/ms-marco-MiniLM-L6-v2'
REVISION = '233902d25c440f23af6f7d6e94d2946bac0bee0a'
GPU = 'A10G'

app = modal.App('minirag-ce')
hf_cache = modal.Volume.from_name('minirag-hf-cache', create_if_missing=True)


def image(chunks_path):
    return (modal.Image.debian_slim(python_version='3.13')
            .pip_install('torch==2.14.0', 'transformers==5.16.1', 'sentence-transformers==6.0.1')
            .add_local_file(chunks_path, '/data/chunks_text.json'))


def build(chunks_path):
    @app.cls(image=image(chunks_path), gpu=GPU, timeout=30 * 60, max_containers=4,
             volumes={'/root/.cache/huggingface': hf_cache}, serialized=True)
    class Scorer:
        @modal.enter()
        def load(self):
            import torch
            from sentence_transformers import CrossEncoder
            torch.backends.cuda.matmul.allow_tf32 = False  # plain fp32, as on CPU
            torch.backends.cudnn.allow_tf32 = False
            self.torch = torch
            self.model = CrossEncoder(CE_MODEL, revision=REVISION, device='cuda', max_length=512)
            self.text = json.load(open('/data/chunks_text.json', encoding='utf-8'))

        @modal.method()
        def score(self, items):
            """items: [(query_id, question, [chunk_id, ...]), ...] -> [(query_id, {chunk_id: logit})]."""
            pairs, keys = [], []
            for qid, question, ids in items:
                for c in ids:
                    pairs.append((question, self.text[c]))
                    keys.append((qid, c))
            s = self.model.predict(pairs, batch_size=64, show_progress_bar=False,
                                   activation_fn=self.torch.nn.Identity(), convert_to_numpy=True)
            out = {}
            for (qid, c), v in zip(keys, s.ravel().tolist()):
                out.setdefault(qid, {})[c] = float(v)
            return [(qid, out.get(qid, {})) for qid, _, _ in items]

    return Scorer
