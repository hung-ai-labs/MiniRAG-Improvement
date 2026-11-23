"""Generate the manuscript's tables from the two held-out evaluation reports."""
import json
from pathlib import Path
from statistics import mean

REPORT = Path(__file__).resolve().parents[1]
REVIEW = REPORT.parent / "review"
OUTPUT = REPORT / "generated"
LIHUA = REVIEW / "evidence_set_20261111" / "qa_nondev435" / "ladder" / "results.json"
MULTIHOP = REVIEW / "mhr_set_20261117"
ARMS = [("C0", "C0"), ("ST", "BASELINE"), ("F", "F"), ("CE", "CE0F"), ("SET", "CE0F_SET")]


def number(value, digits=2):
    return f"{value:.{digits}f}".replace(".", ",")


def write_experiment_details(lihua_report, multihop_report):
    seeds = ("101", "202", "303")
    comparisons = [
        ("mhr_st_20261112", "ST_vs_C0"),
        ("mhr_f_20261114", "F_vs_ST"),
        ("mhr_ce_20261116", "CE_vs_F"),
        ("mhr_set_20261117", "SET_vs_CE"),
    ]
    paired = [r"\begin{tabular}{lrrr}\toprule",
              r"Bước so sánh & Seed 101 & Seed 202 & Seed 303\\\midrule"]
    forest = [r"\begin{tikzpicture}[x=0.85cm,y=0.5cm,font=\small]",
              r"\draw[dashed,gray] (0,0.3)--(0,12.8);"]
    for group, (directory, comparison) in enumerate(comparisons):
        reports = json.loads((REVIEW / directory / "RESULTS.json").read_text(encoding="utf-8"))["seeds"]
        label = comparison.replace("_vs_", " so với ")
        cells = []
        for index, seed in enumerate(seeds):
            result = reports[seed][comparison]
            delta = result["delta_acc_pp"]
            pvalue = result["p_exact"]
            probability = "$<0{,}001$" if pvalue < .001 else number(pvalue, 3)
            cells.append(f"{number(delta)} ({probability})")
            low, high = result["paired_bootstrap_ci95_pp"]
            height = 12 - 3 * group - index
            forest.append(rf"\draw ({low:.4f},{height}) -- ({high:.4f},{height});")
            forest.append(rf"\fill ({delta:.4f},{height}) circle (2pt);")
            forest.append(rf"\node[anchor=east] at (-3.6,{height}) {{{label if index == 0 else ''} \quad {seed}}};")
        paired.append(" & ".join([label, *cells]) + r"\\")
    paired.append(r"\bottomrule\end{tabular}")
    forest.extend([r"\draw (-3,0)--(9,0);",
                   r"\foreach \v in {-2,0,2,4,6,8} {\draw (\v,0)--(\v,-0.15) node[below]{\v};}",
                   r"\end{tikzpicture}"])
    evidence = [r"\begin{tabular}{lrrrr}\toprule",
                r"& LiHua & \multicolumn{3}{c}{MultiHop-RAG}\\",
                r"Cấu hình & Đủ (câu) & Recall (\%) & Đủ (\%) & Token TB\\\midrule"]
    for arm, alias in ARMS:
        per_seed = [multihop_report["seeds"][seed] for seed in seeds]
        micro = 100 * mean(s["evidence_recall"][arm]["micro"] for s in per_seed)
        full = 100 * mean(s["evidence_recall"][arm]["full"] for s in per_seed)
        tokens = mean(s["sources_tokens"][arm]["mean"] for s in per_seed)
        count = lihua_report["build"]["full_evidence"][alias]
        evidence.append(" & ".join([arm, str(count), number(micro, 1), number(full, 1),
                                    number(tokens, 1)]) + r"\\")
    evidence.append(r"\bottomrule\end{tabular}")
    for name, lines in [("pairs", paired), ("intervals", forest), ("evidence", evidence)]:
        (OUTPUT / f"selection_{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

def main():
    lihua = json.loads(LIHUA.read_text(encoding="utf-8"))["table"]["all"]
    multihop = json.loads((MULTIHOP / "RESULTS.json").read_text(encoding="utf-8"))["generation_mean_sd"]
    lines = [r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}\toprule",
             r"\textbf{Cấu hình} & \multicolumn{2}{c}{\textbf{LiHua-World}} & \multicolumn{2}{c}{\textbf{MultiHop-RAG}}\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r" & ACC $\uparrow$ & ERR $\downarrow$ & ACC $\uparrow$ & ERR $\downarrow$\\\midrule"]
    for key, alias in ARMS:
        cells = [key if key == "C0" else "+" + key]
        for dataset, name, sd_key, names in (
            (lihua, alias, "sd_rounds", [a for _, a in ARMS]),
            (multihop, key, "sd", [k for k, _ in ARMS]),
        ):
            for metric, optimum in (("accurate", max), ("error", min)):
                stats = dataset[name][metric]
                average = number(stats["mean"])
                if stats["mean"] == optimum(dataset[n][metric]["mean"] for n in names):
                    average = rf"\textbf{{{average}}}"
                cells.append(average + r"\,$\pm$\," + number(stats[sd_key]))
        lines.append(" & ".join(cells) + r"\\")
    lines.append(r"\bottomrule\end{tabular*}")
    OUTPUT.mkdir(exist_ok=True)
    write_experiment_details(json.loads(LIHUA.read_text(encoding="utf-8")),
                             json.loads((MULTIHOP / "RESULTS.json").read_text(encoding="utf-8")))
    (OUTPUT / "selection_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    data = {}
    for line in (MULTIHOP / "BANG_SO_SANH.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("| C0+ST+F+CE"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            data[cells[0]] = (cells[-2].replace("%", r"\%").replace(".", ","), float(cells[-1]))
    before, after = data["C0+ST+F+CE"], data["C0+ST+F+CE+set"]
    cost = (f"Trên MultiHop-RAG, SET giảm số token nguồn trung bình từ {number(before[1], 1)} "
            f"xuống {number(after[1], 1)} ({number(100 * (1 - after[1] / before[1]), 1)}\\%). "
            f"Tỷ lệ câu đủ bằng chứng chuyển từ {before[0]} xuống {after[0]}.")
    (OUTPUT / "selection_cost.tex").write_text(cost + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
