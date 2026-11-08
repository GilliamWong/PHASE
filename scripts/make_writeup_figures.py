"""Draws the Q-OIL write-up's figures from the runs' saved metrics.

Each run is scored as in scripts.aggregate_runs (success averaged over
training, Q-OIL's Fig. 10 metric); bars and points show the mean over seeds
with a 95% t-interval. The paper's Fig. 10 values are read off the published
figure by eye, so they are approximate.

Usage: python -m scripts.make_writeup_figures [--out Baselines/figures]
"""

import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from scripts.aggregate_runs import NAME, run_score

# One color per method in every figure (a colorblind-safe categorical order).
COLORS = {
    "Q-OIL": "#2a78d6",
    "BC+RL": "#eb6834",
    "HIL": "#1baf7a",
    "Q-OIL w/o BC": "#eda100",
    "No decoupling": "#e87ba4",
    "No decoupling + BC": "#6250d6",
    "RLPD": "#008300",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

BONUSES = ["0.01", "0.025", "0.05", "0.5"]
# Fig. 10 of the paper, read off the figure. At b = 0, Q-OIL is BC+RL and the
# other two curves start from HIL.
PAPER_FIG10 = {
    "Q-OIL": [0.71, 0.71, 0.71, 0.70],
    "Q-OIL w/o BC": [0.47, 0.47, 0.57, 0.64],
    "No decoupling": [0.30, 0.27, 0.14, 0.05],
}
PAPER_B0 = {"BC+RL": 0.38, "HIL": 0.29}


def scores(experiment, task="lift"):
    """Per-seed scores of every finished run of one experiment."""
    out = []
    for run_dir in sorted(glob.glob(f"runs/{experiment}_{task}_ph_s*")):
        m = NAME.match(os.path.basename(run_dir))
        if m and m["name"] == experiment:
            scored = run_score(run_dir)
            if scored:
                out.append(scored[0])
    if not out:
        raise FileNotFoundError(f"no finished runs of {experiment} ({task})")
    return np.array(out)


def mean_ci(experiment, task="lift"):
    s = scores(experiment, task)
    half = stats.t.ppf(0.975, len(s) - 1) * s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0
    return s.mean(), half


def style(ax):
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)


def grouped_bars(ax, groups, series, values, ylabel="Success (avg. over training)"):
    """values[(group, series)] = (mean, ci) or None; each bar is labeled with its mean."""
    n = len(series)
    width = 0.8 / n
    for i, name in enumerate(series):
        labeled = False
        for j, group in enumerate(groups):
            v = values.get((group, name))
            if v is None:
                continue
            x = j + (i - (n - 1) / 2) * width
            ax.bar(x, v[0], width * 0.92, color=COLORS[name], label=None if labeled else name)
            labeled = True
            ax.errorbar(x, v[0], yerr=v[1], color=INK, linewidth=0.8, capsize=0)
            text = f"{v[0]:.2f}".lstrip("0") if v[0] >= 0.005 else "0"  # ".41": narrow enough for crowded groups
            ax.text(x, v[0] + v[1] + 0.015, text, ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    style(ax)


def legend_below(fig, axes, ncol):
    """One legend under the figure for the series drawn on the given axes."""
    handles, labels = [], []
    for a in axes:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=9)


def fig10(out):
    """The paper's Fig. 10 next to ours (Lift, half of takeovers run to the episode's end)."""
    x = np.arange(len(BONUSES) + 1)  # b = 0 first, then the swept bonuses, evenly spaced
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    ax = axes[0]
    for name, ys in PAPER_FIG10.items():
        start = PAPER_B0["BC+RL" if name == "Q-OIL" else "HIL"]
        ax.plot(x, [start, *ys], color=COLORS[name], linewidth=2, marker="o", markersize=5, label=name)
    for name, y in PAPER_B0.items():
        ax.plot(0, y, linestyle="none", marker="D", markersize=7, color=COLORS[name], label=f"{name} (b = 0)")
    ax.set_title("Paper, Fig. 10 (read off the figure)", fontsize=10, color=INK)

    ax = axes[1]
    curves = {"Q-OIL": ("qoil", "f10_full_bcrl"), "Q-OIL w/o BC": ("nobc", "f10_full_hil"),
              "No decoupling": ("nodecouple", "f10_full_hil"),
              "No decoupling + BC": ("nodecouple_bc", "f10_full_bcrl")}
    for name, (arm, start) in curves.items():
        pts = [mean_ci(start)] + [mean_ci(f"f10_full_{arm}_b{b}") for b in BONUSES]
        m, h = np.array(pts).T
        ax.plot(x, m, color=COLORS[name], linewidth=2, marker="o", markersize=5, label=name)
        ax.fill_between(x, m - h, m + h, color=COLORS[name], alpha=0.12, linewidth=0)
    for name, exp in (("BC+RL", "f10_full_bcrl"), ("HIL", "f10_full_hil")):
        m, h = mean_ci(exp)
        ax.errorbar(0, m, yerr=h, linestyle="none", marker="D", markersize=7, color=COLORS[name],
                    ecolor=INK, elinewidth=0.8, label=f"{name} (b = 0)")
    m, _ = mean_ci("fixT_rlpd")
    ax.axhline(m, color=COLORS["RLPD"], linestyle="--", linewidth=1.5, label="RLPD, no human")
    ax.set_title("Ours: Lift, 6 seeds, 95% CI", fontsize=10, color=INK)

    for ax in axes:
        ax.set_xticks(x, ["0", *BONUSES])
        ax.set_xlabel("Optimism bonus b", color=INK, fontsize=9)
        ax.set_ylim(0, 1.0)
        style(ax)
    axes[0].set_ylabel("Success (avg. over training)", color=INK, fontsize=9)
    fig.subplots_adjust(bottom=0.32, wspace=0.08)
    legend_below(fig, axes, ncol=4)
    fig.savefig(os.path.join(out, "fig1_fig10_reproduction.png"), dpi=200, facecolor="white")


def imitation(out):
    """How close each final policy's action is to the corrector's, at the states where it took over."""
    arms = [("Q-OIL", "f10_full_qoil_b0.025"), ("BC+RL", "f10_full_bcrl"), ("HIL", "f10_full_hil"),
            ("Q-OIL w/o BC", "f10_full_nobc_b0.025"), ("No decoupling", "f10_full_nodecouple_b0.025")]
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    for i, (name, exp) in enumerate(arms):
        d = []
        for path in glob.glob(f"runs/{exp}_lift_ph_s*/imitation.csv"):
            rows = list(csv.DictReader(open(path)))
            d.append(float(rows[-1]["d_corrector"]))
        ax.barh(i, np.mean(d), 0.7, color=COLORS[name])
        ax.text(np.mean(d) + 0.03, i, f"{np.mean(d):.2f} ({len(d)} seeds)", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(arms)), [n for n, _ in arms])
    ax.invert_yaxis()
    ax.set_xlim(0, 2.2)
    ax.set_xlabel("Distance from the corrector's action at its takeover states (L2, final policy)",
                  color=INK, fontsize=9)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_imitation.png"), dpi=200, facecolor="white")


def imitation_over_training(out):
    """Distance from the corrector over training, and Q-OIL's interventions: the share of its replay
    buffer they make up against the per-episode intervention rate the paper's annealing argument uses."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    for name, exp in (("Q-OIL", "f10_full_qoil_b0.025"), ("BC+RL", "f10_full_bcrl"), ("HIL", "f10_full_hil"),
                      ("Q-OIL w/o BC", "f10_full_nobc_b0.025")):
        curves = [{int(r["step"]): float(r["d_corrector"]) for r in csv.DictReader(open(path))}
                  for path in glob.glob(f"runs/{exp}_lift_ph_s*/imitation.csv")]
        steps = sorted(set.intersection(*(set(c) for c in curves)))
        ax.plot(np.array(steps) / 1000, [np.mean([c[s] for c in curves]) for s in steps], color=COLORS[name],
                linewidth=2.5 if name == "Q-OIL" else 1.5, marker="o", markersize=4,
                label=f"{name} ({len(curves)} seeds)")
    ax.set_ylim(0, 2.0)
    ax.set_ylabel("Distance from the corrector's action", color=INK, fontsize=9)
    ax.set_title("Distance from the corrector at its takeover states", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="center right")

    ax = axes[1]
    bins = np.arange(0, 100_001, 10_000)
    share, rate = [], []
    for run_dir in sorted(glob.glob("runs/f10_full_qoil_b0.025_lift_ph_s*")):
        acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
        acc.Reload()
        events = acc.Scalars("training/total_intervention_steps")
        steps = np.array([e.step + 1 for e in events])
        total = np.array([e.value for e in events])
        at = np.interp(bins[1:], steps, total)  # cumulative intervention steps at each bin edge
        share.append(at / bins[1:])
        rate.append(np.diff(np.concatenate([[0], at])) / np.diff(bins))
    x = bins[1:] / 1000
    ax.plot(x, 100 * np.mean(share, 0), color=COLORS["Q-OIL"], linewidth=2.5,
            label="Share of the replay buffer")
    ax.plot(x, 100 * np.mean(rate, 0), color=COLORS["Q-OIL"], linewidth=1.5, linestyle="--",
            label="Share of steps in each 10k-step window")
    ax.set_ylim(0, 60)
    ax.set_ylabel("Intervention transitions (%)", color=INK, fontsize=9)
    ax.set_title(f"Q-OIL's interventions ({len(share)} seeds)", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for ax in axes:
        ax.set_xlim(0, 102)
        ax.set_xlabel("Environment steps (thousands)", color=INK, fontsize=9)
        style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig6_imitation_over_training.png"), dpi=200, facecolor="white")


def bc_form(out):
    """Q-OIL against BC+RL for each BC form and corrector detail the paper leaves open."""
    variants = {
        "Log-likelihood\n(paper's loss)": ("f10_full_bcrl", "f10_full_qoil_b0.025"),
        "MSE on the\nmean action": ("mse_bcrl", "mse_qoil_b0.025"),
        "Capped\nlog-likelihood": ("bccap_bcrl", "bccap_qoil_b0.025"),
        "Log-likelihood,\nsampled\ncorrector actions": ("hsample_bcrl", "hsample_qoil_b0.025"),
        "Log-likelihood,\ndemos counted\nas interventions": ("hdemo_bcrl", "hdemo_qoil_b0.025"),
    }
    values = {}
    for group, (bcrl, qoil) in variants.items():
        values[(group, "BC+RL")] = mean_ci(bcrl)
        values[(group, "Q-OIL")] = mean_ci(qoil)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [2.6, 1]})
    grouped_bars(axes[0], list(variants), ["BC+RL", "Q-OIL"], values)
    m, _ = mean_ci("f10_full_hil")
    axes[0].axhline(m, color=COLORS["HIL"], linestyle="--", linewidth=1.5, label=f"HIL, default ({m:.2f})")
    axes[0].set_title("BC weight 0.1: BC loss and unreported human details", fontsize=10, color=INK)
    axes[0].tick_params(axis="x", labelsize=8)

    ax = axes[1]
    sweeps = {  # BC+RL experiment for each BC weight, at two fixed temperatures
        "T = 0.01": {0.1: "f10_full_bcrl", 0.3: "f10_full_bcrl_l0.3", 1.0: "f10_full_bcrl_l1.0"},
        "T = 0.1": {0.03: "t01_bcrl_l0.03", 0.1: "tsweep_bcrl_T0.1_nobk", 0.3: "t01_bcrl_l0.3",
                    1.0: "t01_bcrl_l1.0"},
    }
    for (label, sweep), (marker, line) in zip(sweeps.items(), (("o", "-"), ("s", ":"))):
        m, h = np.array([mean_ci(exp) for exp in sweep.values()]).T
        xs = list(sweep)
        ax.errorbar(xs, m, yerr=h, color=COLORS["BC+RL"], linewidth=2, marker=marker, linestyle=line,
                    markersize=5, ecolor=INK, elinewidth=0.8, label=f"BC+RL, {label}")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_xscale("log")
    ax.set_xticks([0.03, 0.1, 0.3, 1.0], ["0.03", "0.1", "0.3", "1"])
    ax.set_xlim(0.02, 1.5)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("BC weight λ", color=INK, fontsize=9)
    ax.set_title("BC weight sweep", fontsize=10, color=INK)
    style(ax)
    fig.subplots_adjust(bottom=0.31, wspace=0.18)
    legend_below(fig, [axes[0]], ncol=3)
    fig.savefig(os.path.join(out, "fig3_bc_form.png"), dpi=200, facecolor="white")


def rl_settings(out):
    """The core comparison under RL settings the paper does not report."""
    settings = {
        "T = 0.01\n(our default)": {"RLPD": "fixT_rlpd", "HIL": "f10_full_hil", "Q-OIL w/o BC": "f10_full_nobc_b0.025",
                                    "BC+RL": "f10_full_bcrl", "Q-OIL": "f10_full_qoil_b0.025"},
        "T = 0.03": "tsweep_{}_T0.03_nobk",
        "T = 0.03\n+ entropy backup": "tsweep_{}_T0.03_bk",
        "T = 0.1": "tsweep_{}_T0.1_nobk",
        "Success\nreward 1": "r1_{}",
    }
    arms = {"RLPD": "rlpd", "HIL": "hil", "Q-OIL w/o BC": "nobc_b0.025", "BC+RL": "bcrl", "Q-OIL": "qoil_b0.025"}
    values = {}
    for group, spec in settings.items():
        for name, arm in arms.items():
            exp = spec[name] if isinstance(spec, dict) else spec.format(arm)
            values[(group, name)] = mean_ci(exp)
    values[("Paper\n(Fig. 10)", "HIL")] = (PAPER_B0["HIL"], 0)
    values[("Paper\n(Fig. 10)", "BC+RL")] = (PAPER_B0["BC+RL"], 0)
    values[("Paper\n(Fig. 10)", "Q-OIL w/o BC")] = (PAPER_FIG10["Q-OIL w/o BC"][1], 0)
    values[("Paper\n(Fig. 10)", "Q-OIL")] = (PAPER_FIG10["Q-OIL"][1], 0)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    grouped_bars(ax, ["Paper\n(Fig. 10)", *settings], list(arms), values)
    ax.axvline(0.5, color=MUTED, linewidth=0.8)
    ax.tick_params(axis="x", labelsize=8)
    fig.subplots_adjust(bottom=0.25)
    legend_below(fig, [ax], ncol=5)
    fig.savefig(os.path.join(out, "fig4_rl_settings.png"), dpi=200, facecolor="white")


def robustness(out):
    """Q-OIL against BC+RL when the human, the demos or the task change."""
    arms = {"RLPD": "rlpd", "HIL": "hil", "Q-OIL w/o BC": "nobc_b0.025", "BC+RL": "bcrl", "Q-OIL": "qoil_b0.025"}
    lift = {"Main setting": "f10_full_{}", "10 demos": "d10_{}", "75% corrector": "c75_{}",
            "β = 0.005": "beta_{}"}
    values = {}
    for group, pattern in lift.items():
        for name, arm in arms.items():
            exp = "fixT_rlpd" if (group == "Main setting" and arm == "rlpd") else pattern.format(arm)
            if glob.glob(f"runs/{exp}_lift_ph_s*"):
                values[(group, name)] = mean_ci(exp)
    other = {"Can": ("can", "can_full_{}", "can_rlpd"), "Can, demos as\ninterventions": ("can", "can_hdemo_{}", None),
             "Square": ("square", "square_cap100_{}", "square_rlpd"),
             "Square, demos as\ninterventions": ("square", "square_hdemo_{}", None)}
    values2 = {}
    for group, (task, pattern, rlpd) in other.items():
        for name, arm in arms.items():
            exp = rlpd if arm == "rlpd" else pattern.format(arm)
            if exp and glob.glob(f"runs/{exp}_{task}_ph_s*"):
                values2[(group, name)] = mean_ci(exp, task)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    grouped_bars(axes[0], list(lift), list(arms), values)
    axes[0].set_title("Lift (6 seeds)", fontsize=10, color=INK)
    grouped_bars(axes[1], list(other), list(arms), values2, ylabel="")
    axes[1].set_title("Can (6 seeds) and Square (3 seeds)", fontsize=10, color=INK)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    fig.subplots_adjust(bottom=0.25, wspace=0.05)
    legend_below(fig, axes, ncol=5)
    fig.savefig(os.path.join(out, "fig5_robustness.png"), dpi=200, facecolor="white")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="Baselines/figures")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"
    for draw in (fig10, imitation, imitation_over_training, bc_form, rl_settings, robustness):
        draw(args.out)
        plt.close("all")
        print("drew", draw.__name__)


if __name__ == "__main__":
    main()
