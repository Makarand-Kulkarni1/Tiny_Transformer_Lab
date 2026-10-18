import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# label, run name (files are results/<run>_<split>_eval.json), short description
MODELS = [
    ("From scratch", "v2aug3", "852k parameters, trained from zero"),
    ("LoRA Qwen2.5-0.5B", "lora", "8.8M trained, 494M total"),
    ("API gpt-oss-120b", "api", "120B, 5 examples in the prompt, no training"),
]


def load(run, split):
    path = RESULTS / f"{run}_{split}_eval.json"
    if not path.exists():
        raise SystemExit(f"missing {path} - download it from Drive into results/")
    return json.loads(path.read_text(encoding="utf-8"))["examples"]


def wilson(k, n, z=1.96):
    """95% confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def score(rows, key="exec_correct"):
    k, n = sum(r[key] for r in rows), len(rows)
    lo, hi = wilson(k, n)
    return {"acc": k / n, "lo": lo, "hi": hi, "k": k, "n": n}


def fmt(s):
    return f"{s['acc']:.1%} ({s['lo']:.1%}-{s['hi']:.1%})"


def mcnemar_p(b, c):
    """Exact two-sided p-value: are the two models' disagreements lopsided?"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(rows_a, rows_b):
    a = {r["question"]: r["exec_correct"] for r in rows_a}
    b = {r["question"]: r["exec_correct"] for r in rows_b}
    common = a.keys() & b.keys()
    only_a = sum(a[q] and not b[q] for q in common)
    only_b = sum(b[q] and not a[q] for q in common)
    return only_a, only_b, mcnemar_p(only_a, only_b)


def main():
    api_test = load("api", "test")
    subset = {r["question"] for r in api_test}          # the 200 questions every model is scored on

    data = {}
    for label, run, _ in MODELS:
        test_all = api_test if run == "api" else load(run, "test")
        test_sub = [r for r in test_all if r["question"] in subset]
        if len(test_sub) != len(subset):
            print(f"warning: {label} has only {len(test_sub)} of the {len(subset)} shared test questions")
        data[label] = {"test_full": test_all, "test_sub": test_sub, "para": load(run, "paraphrase")}

    lines = ["| Model | Test, 200 shared questions | Paraphrase, 108 questions | Valid SQL (paraphrase) | Test, all 1,172 |",
             "|---|---|---|---|---|"]
    scores = {}
    for label, run, desc in MODELS:
        d = data[label]
        t, p = score(d["test_sub"]), score(d["para"])
        v = score(d["para"], "valid")
        scores[label] = {"test": t, "para": p}
        full = fmt(score(d["test_full"])) if run != "api" else "not run"
        lines.append(f"| {label} ({desc}) | {fmt(t)} | {fmt(p)} | {v['acc']:.1%} | {full} |")

    lines += ["", "Pairs compared on identical questions (McNemar exact test):", ""]
    labels = [m[0] for m in MODELS]
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            for name, key in [("test (200)", "test_sub"), ("paraphrase (108)", "para")]:
                a_only, b_only, p = paired(data[labels[i]][key], data[labels[j]][key])
                verdict = "clear difference" if p < 0.05 else "no clear difference"
                lines.append(f"- {labels[i]} vs {labels[j]} on {name}: only the first right {a_only}, "
                             f"only the second right {b_only}, p = {p:.3g} ({verdict})")

    text = "\n".join(lines)
    print(text)
    (RESULTS / "comparison.md").write_text(text + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25
    for idx, label in enumerate(labels):
        accs = [scores[label]["test"]["acc"] * 100, scores[label]["para"]["acc"] * 100]
        lows = [accs[0] - scores[label]["test"]["lo"] * 100, accs[1] - scores[label]["para"]["lo"] * 100]
        highs = [scores[label]["test"]["hi"] * 100 - accs[0], scores[label]["para"]["hi"] * 100 - accs[1]]
        xs = [x + (idx - 1) * width for x in range(2)]
        ax.bar(xs, accs, width, yerr=[lows, highs], capsize=4, label=label)
    ax.set_xticks(range(2))
    ax.set_xticklabels(["Test (template wordings)", "Paraphrase (new wordings)"])
    ax.set_ylabel("execution accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Same questions, three models (bars: 95% confidence range)")
    ax.legend(loc="lower left")
    fig.savefig(RESULTS / "comparison.png", dpi=150, bbox_inches="tight")
    print("\nsaved results/comparison.md and results/comparison.png")


if __name__ == "__main__":
    main()