from .config import ROOT, run_dir

COLORS = {"dense_active": "#315b83", "dense_total": "#6e879b", "moe": "#b47719",
          "random": "#965f86", "oracle": "#606c38"}
COLORS.update({"attn_dense": "#315b83", "attn_active": "#6e879b",
               "attn_total": "#24384d", "attn_moh": "#b47719", "attn_random": "#965f86",
               "attn_ffn_moe": "#606c38", "attn_ffn_total": "#899276", "attn_both": "#bd7042"})


def style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.figsize": (9, 4.5), "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": .18, "figure.constrained_layout.use": True, "savefig.dpi": 160})


def save_figure(fig, cfg, name):
    path = ROOT / "results" / "figures" / run_dir(cfg).name
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / (name + ".png"), bbox_inches="tight")
    return path / (name + ".png")
