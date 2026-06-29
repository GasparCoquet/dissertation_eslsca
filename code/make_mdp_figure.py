"""
make_mdp_figure.py
Regenerate fig_mdp_architecture.png (the agent-environment MDP loop diagram)
with the state/reward label placed clear of the lower return arrow.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)

NAVY = "#1b2f56"; STEEL = "#3b6ea5"; ARROW = "#1b3a6b"
GREY = "#9aa7b5"; LIGHT = "#e8edf3"; TITLE = "#2f4a6b"


def box(ax, cx, cy, w, h, facecolor, textcolor, lines, fs=13):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.3,rounding_size=2.0",
                 linewidth=1.2, edgecolor=NAVY, facecolor=facecolor, zorder=2))
    ax.text(cx, cy, "\n".join(lines), ha="center", va="center",
            color=textcolor, fontsize=fs, zorder=3)


def curved(ax, p0, p1, rad, color=ARROW, lw=2.2):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                 arrowstyle="-|>", mutation_scale=18, linewidth=lw,
                 color=color, zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(12.4, 6.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

    ax.text(50, 59, "Markov Decision Process formalisation of multi-venue execution",
            ha="center", va="center", fontsize=17, color=TITLE)

    # boxes
    box(ax, 24, 38, 32, 15, STEEL, "white",
        ["AGENT", "(execution policy π)", "linear in state features"])
    box(ax, 79, 38, 34, 15, NAVY, "white",
        ["MARKET ENVIRONMENT", "price • impact • venues", "(calibrated simulator)"])
    box(ax, 50, 11, 44, 12, LIGHT, "#22324d",
        [r"objective:  maximise  $\mathbb{E}[\sum_t r_t]$  $\Leftrightarrow$  minimise  IS"], fs=13)

    # top arrow: AGENT -> ENVIRONMENT (bulges up), label above the arc
    curved(ax, (40, 43.5), (62, 43.5), -0.35)
    ax.text(51, 51, r"action $a_t$:" + "\nper-venue execution rates",
            ha="center", va="center", fontsize=12, color="#22324d")

    # bottom arrow: ENVIRONMENT -> AGENT (bulges down), label BELOW the arc (cleared)
    curved(ax, (62, 32.5), (40, 32.5), -0.35)
    ax.text(51, 23.5, r"state $s_{t+1}$   $\bullet$   reward $r_t = -\,$IS increment",
            ha="center", va="center", fontsize=12, color="#22324d")

    # grey arrows down to objective box
    ax.add_patch(FancyArrowPatch((22, 30.5), (38, 17.2),
                 connectionstyle="arc3,rad=0.18", arrowstyle="-|>",
                 mutation_scale=16, linewidth=2.0, color=GREY, zorder=1))
    ax.add_patch(FancyArrowPatch((81, 30.5), (63, 17.2),
                 connectionstyle="arc3,rad=-0.18", arrowstyle="-|>",
                 mutation_scale=16, linewidth=2.0, color=GREY, zorder=1))

    fig.savefig(os.path.join(FIG, "fig_mdp_architecture.png"), dpi=100,
                bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    print("wrote fig_mdp_architecture.png")


if __name__ == "__main__":
    main()
