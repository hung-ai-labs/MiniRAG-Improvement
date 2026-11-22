"""Render the one-run MultiHop-RAG ACC/ERR tables from verified results."""
import json
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1]
SOURCE = REPORT.parent / "review/mhr_full_20261119/results.json"
ARMS = ("C0", "ST", "F", "CE", "SET")
NAMES = ("C0", "+ST", "+F", "+CE", "+SET")


def render(scores=None):
    lines = [
        r"\begin{tabular}{l*{5}{rr}}\toprule",
        " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{name}}}" for name in NAMES) + r"\\",
        " & " + " & ".join([r"ACC $\uparrow$ & ERR $\downarrow$"] * 5) + r"\\\midrule",
    ]
    if scores is None:
        cells = [r"\multicolumn{2}{c}{--}"] * 5
    else:
        best = {"acc": max(scores[arm]["acc"] for arm in ARMS), "err": min(scores[arm]["err"] for arm in ARMS)}
        cells = []
        for arm in ARMS:
            for metric in ("acc", "err"):
                value = scores[arm][metric]
                formatted = f"{value:.2f}".replace(".", ",")
                cells.append(rf"\textbf{{{formatted}}}" if value == best[metric] else formatted)
    lines.extend([" & ".join(["Qwen2.5-3B-Instruct", *cells]) + r"\\", r"\bottomrule\end{tabular}"])
    return "\n".join(lines) + "\n"


def main():
    result = json.loads(SOURCE.read_text(encoding="utf-8"))
    full = result["n_pending"] == 0
    scores = result["arms"]
    expected = result["n_full"] if full else result["n_existing"]
    if set(scores) != set(ARMS) or any(scores[arm]["n"] != expected for arm in ARMS):
        raise ValueError("Every configuration must cover the same question set")
    output = REPORT / "generated"
    output.mkdir(exist_ok=True)
    (output / "mhr_full_main.tex").write_text(render(scores if full else None), encoding="utf-8")
    (output / "mhr_full_status.tex").write_text(
        (r"\newif\ifFullResults" + "\n" + (r"\FullResultstrue" if full else r"\FullResultsfalse") + "\n"),
        encoding="utf-8",
    )
    if not full:
        (output / "mhr_full_interim.tex").write_text(render(scores), encoding="utf-8")


if __name__ == "__main__":
    main()
