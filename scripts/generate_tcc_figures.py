"""Generate all figures referenced by tcc/cic-tc.tex.

Each ``make_<name>()`` function produces one PDF+PNG figure under
``tcc/images/<batch>/``. Run with ``python scripts/generate_tcc_figures.py``
or ``--only <name>`` to render a single figure.

Output layout::

    tcc/images/
        architecture/  — Batch A (pipeline + HSDC/SWHDC blocks + backbones)
        theory/        — Batch B (3D-rep, ERP, latitude distortion, EgoNeRF)
        methodology/   — Batch C (real-data RF-ERP, culling, augmentation)
        setup/         — Batch D (class distribution, LR schedule)
        related/       — Batch E (method taxonomy)
        data/          — per-class ERP gallery
        results/       — copied from experiments/figures/ in a separate step
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Rectangle
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "tcc" / "images"
PLY_ROOT = ROOT / "gs_data" / "modelsplat" / "modelsplat_ply"
ERP_CACHE = ROOT / "data" / "processed" / "modelnet10" / "radiance_field" / "ns8_H256_W512_c3.0_p5.0-95.0"

# Allow ``from src.preprocessing.ply_loader import load_gaussian_ply``
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

matplotlib.rcParams.update({
    "figure.dpi":          150,
    "figure.facecolor":    "white",
    "savefig.facecolor":   "white",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "font.family":         "DejaVu Sans",
    "font.size":           10,
    "axes.titlesize":      11,
    "axes.labelsize":      10,
    "legend.fontsize":     9,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "axes.grid":           False,
    # Embed TrueType fonts instead of Type 3 outlines.  Type 3 fonts can cause
    # "no glyph" warnings in some PDF viewers and are not preferred for printing.
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
})

PALETTE = {
    "hsdc":      "#2166ac",
    "swhdc":     "#d6604d",
    "resnet":    "#4dac26",
    "purple":    "#9970ab",
    "amber":     "#e08214",
    "grey":      "#808080",
    "red":       "#d73027",
    "lightblue": "#92c5de",
    "lightred":  "#f4a582",
    "edge":      "#333333",
    "bg":        "#f4f4f4",
    "bg2":       "#e8eef5",
}


def _flatten_transparency(fig: plt.Figure) -> None:
    """Eliminate PDF transparency operators by compositing semi-transparent artists
    over a white background before saving.

    matplotlib's PDF backend emits ExtGState entries with ``ca``/``CA < 1`` for
    every artist whose alpha is strictly between 0 and 1.  These transparency
    operators are rendered incorrectly (dark backgrounds, artefacts) by some PDF
    viewers — notably SumatraPDF and certain Acrobat versions — as well as by
    pdflatex's inclusion pipeline on older TeX Live installations.

    This function walks every artist in the figure tree and, for each artist
    whose alpha is in (0, 1), composites its fill/edge/line colour over white and
    resets the artist alpha to 1.  Vector text and fully-opaque artists are
    unchanged.  The 3D axis panes (``ax.xaxis.pane`` etc.) are handled
    separately because they are not exposed through ``get_children()`` traversal.
    """
    import matplotlib.collections as mc
    import matplotlib.lines as ml
    import matplotlib.patches as mp
    import matplotlib.colors as mcolors

    WHITE = np.array([1.0, 1.0, 1.0])

    def _composite(rgba: np.ndarray, alpha: float) -> np.ndarray:
        """Alpha-composite an RGBA colour over white; return opaque RGBA."""
        rgb = np.asarray(rgba)[:3]
        return np.append(alpha * rgb + (1.0 - alpha) * WHITE, 1.0)

    def _fix(artist) -> None:
        alpha = artist.get_alpha()
        if alpha is not None and 0.0 < alpha < 1.0:
            if isinstance(artist, (mp.Patch,)):
                fc = np.array(mcolors.to_rgba(artist.get_facecolor()))
                ec = np.array(mcolors.to_rgba(artist.get_edgecolor()))
                artist.set_facecolor(_composite(fc, alpha))
                artist.set_edgecolor(_composite(ec, alpha))
                artist.set_alpha(1.0)

            elif isinstance(artist, mc.Collection):
                # Collections store RGBA in _edgecolors/_facecolors internally.
                # mcolors.to_rgba_array multiplies the array by self._alpha, so
                # we must set _alpha = 1.0 BEFORE calling set_edgecolor /
                # set_facecolor to avoid double-application.
                raw_ec = artist.get_edgecolor()
                raw_fc = artist.get_facecolor()
                artist._alpha = 1.0
                if hasattr(raw_ec, "__len__") and len(raw_ec):
                    artist.set_edgecolor([_composite(c, alpha) for c in raw_ec])
                if hasattr(raw_fc, "__len__") and len(raw_fc):
                    try:
                        artist.set_facecolor([_composite(c, alpha) for c in raw_fc])
                    except Exception:
                        pass

            elif isinstance(artist, ml.Line2D):
                c = np.array(mcolors.to_rgba(artist.get_color()))
                artist.set_color(_composite(c, alpha))
                artist.set_alpha(1.0)

        for child in artist.get_children():
            _fix(child)

    _fix(fig)

    # 3D axis panes are not reachable via get_children() — handle them explicitly.
    for ax in fig.get_axes():
        for axis_name in ("xaxis", "yaxis", "zaxis"):
            axis = getattr(ax, axis_name, None)
            if axis is None or not hasattr(axis, "pane"):
                continue
            pane = axis.pane
            alpha = pane.get_alpha()
            if alpha is not None and 0.0 < alpha < 1.0:
                fc = np.array(mcolors.to_rgba(pane.get_facecolor()))
                ec = np.array(mcolors.to_rgba(pane.get_edgecolor()))
                pane.set_facecolor(_composite(fc, alpha))
                pane.set_edgecolor(_composite(ec, alpha))
                pane.set_alpha(1.0)


def save(fig: plt.Figure, category: str, name: str) -> None:
    out = IMG / category / name
    out.parent.mkdir(parents=True, exist_ok=True)
    # Composite all semi-transparent artists over white before writing the PDF so
    # that the output contains no PDF transparency operators (ca/CA < 1 in the
    # page ExtGState).  This prevents dark-background artefacts in SumatraPDF,
    # Acrobat, and pdflatex's \includegraphics on some PDF viewers.
    _flatten_transparency(fig)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.relative_to(ROOT)}.pdf")


def add_box(ax, x, y, w, h, text, fc=PALETTE["bg"], ec=PALETTE["edge"],
            fontsize=9, fontweight="normal"):
    """Draw a rounded box with centered multi-line text."""
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.05",
                           linewidth=1.0, facecolor=fc, edgecolor=ec)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, wrap=True)


def add_arrow(ax, x0, y0, x1, y1, color=PALETTE["edge"], lw=1.2,
              arrow_style="-|>", mutation=12, connectionstyle="arc3,rad=0"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=arrow_style,
                        mutation_scale=mutation, linewidth=lw, color=color,
                        connectionstyle=connectionstyle)
    ax.add_patch(a)


# ===========================================================================
# Batch A — Architecture diagrams
# ===========================================================================

def make_pipeline_overview() -> None:
    """7-stage end-to-end pipeline overview (replaces Ch.4 placeholder)."""
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 4.5); ax.axis("off")

    stages = [
        (0.2,  "3DGS PLY\n($N$ Gaussians)",                    PALETTE["bg"]),
        (2.2,  "Opacity-weighted\ncentroid $\\mathbf{C}$",        PALETTE["bg2"]),
        (4.2,  "EgoNeRF\nexp. shells $r_s$",                   PALETTE["bg2"]),
        (6.2,  "RF-ERP tensor\n$N_{\\rm sh}\\!\\times\\!H\\!\\times\\!W$", PALETTE["lightblue"]),
        (8.2,  "Transforms\nlog1p + derived\n+ aug.",         PALETTE["bg2"]),
        (10.2, "Backbone\nResNet +\nHSDC / SWHDC",            PALETTE["lightred"]),
        (12.2, "Classifier\nGAP + FC + softmax",              PALETTE["bg"]),
    ]
    box_w, box_h = 1.7, 1.8
    y_box = 1.4

    for x, text, fc in stages:
        add_box(ax, x, y_box, box_w, box_h, text, fc=fc, fontsize=9)

    # Connecting arrows
    for i in range(len(stages) - 1):
        x_src = stages[i][0] + box_w
        x_dst = stages[i + 1][0]
        add_arrow(ax, x_src, y_box + box_h / 2, x_dst, y_box + box_h / 2, lw=1.5)

    # Stage labels above
    for i, (x, _, _) in enumerate(stages):
        ax.text(x + box_w / 2, y_box + box_h + 0.25, f"Stage {i+1}",
                ha="center", va="bottom", fontsize=9, color=PALETTE["edge"],
                fontweight="bold")

    # Section brackets at the bottom
    bracket_y = 0.7
    x0 = stages[0][0]
    x1 = stages[3][0] + box_w
    ax.plot([x0, x0, x1, x1], [bracket_y + 0.1, bracket_y, bracket_y,
            bracket_y + 0.1], color=PALETTE["hsdc"], lw=1.4)
    ax.text((x0 + x1) / 2, bracket_y - 0.25, "RF-ERP preprocessing (Sec. 4.3)",
            ha="center", va="top", fontsize=9, color=PALETTE["hsdc"])
    x0 = stages[4][0]
    x1 = stages[6][0] + box_w
    ax.plot([x0, x0, x1, x1], [bracket_y + 0.1, bracket_y, bracket_y,
            bracket_y + 0.1], color=PALETTE["swhdc"], lw=1.4)
    ax.text((x0 + x1) / 2, bracket_y - 0.25,
            "Classifier-side transforms + network (Sec. 4.4-4.5)",
            ha="center", va="top", fontsize=9, color=PALETTE["swhdc"])

    save(fig, "architecture", "pipeline_overview")


def _draw_block_diagram(ax, color, combine_box_text, output_text,
                        combine_box_h=0.9, combine_box_w=1.6):
    """Shared HSDC/SWHDC schematic; differs only in the combine box + output."""
    ax.set_xlim(0, 11); ax.set_ylim(0, 6); ax.axis("off")

    # Input
    add_box(ax, 0.2, 2.6, 1.5, 0.8, "Input\n$C \\times H \\times W$",
            fc=PALETTE["bg"], fontsize=9)

    # 4 dilated branches — short labels so the connecting lines stay outside
    branch_y = [4.4, 3.6, 2.4, 1.2]
    rates = [1, 2, 3, 4]
    branch_x = 3.1
    branch_w, branch_h = 1.7, 0.7
    # Bus y-span: exactly from top branch centre to bottom branch centre
    bus_top = branch_y[0] + branch_h / 2
    bus_bot = branch_y[-1] + branch_h / 2

    # Left hub: short stub from input to a vertical line, then fanout
    hub_x = 2.2
    ax.plot([1.7, hub_x], [3.0, 3.0], color=PALETTE["edge"], lw=1.2)
    ax.plot([hub_x, hub_x], [bus_top, bus_bot], color=PALETTE["edge"], lw=1.2)

    for y_b, d in zip(branch_y, rates):
        add_box(ax, branch_x, y_b, branch_w, branch_h,
                f"dilation $d = {d}$", fc=color, fontsize=9)
        ax.plot([hub_x, branch_x], [y_b + branch_h / 2,
                y_b + branch_h / 2], color=PALETTE["edge"], lw=1.2)

    # Right hub: gather lines from each branch
    hub2_x = 5.4
    ax.plot([hub2_x, hub2_x], [bus_top, bus_bot], color=PALETTE["edge"], lw=1.2)
    for y_b in branch_y:
        ax.plot([branch_x + branch_w, hub2_x], [y_b + branch_h / 2,
                y_b + branch_h / 2], color=PALETTE["edge"], lw=1.2)

    # Main horizontal flow: hub2 -> combine box (with arrow head)
    cx = 5.9
    cy_box = 3.0 - combine_box_h / 2
    add_box(ax, cx, cy_box, combine_box_w, combine_box_h, combine_box_text,
            fc=PALETTE["bg2"], fontsize=9)
    add_arrow(ax, hub2_x, 3.0, cx, 3.0, lw=1.2)

    # Combine -> output (with arrow head)
    out_x = cx + combine_box_w + 0.5
    add_box(ax, out_x, 2.55, 1.9, 0.9, output_text, fc=color, fontsize=9)
    add_arrow(ax, cx + combine_box_w, 3.0, out_x, 3.0, lw=1.2)

    # Annotation footnote
    ax.text(5.5, 0.6,
            "All four branches are $3\\times3$ convolutions with "
            "circular padding and shared kernel weights.",
            ha="center", fontsize=9, color=PALETTE["edge"], style="italic")


def make_hsdc_block() -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_block_diagram(
        ax,
        color=PALETTE["lightblue"],
        combine_box_text="Concatenate\nalong channel\n+ BN + ReLU",
        output_text="Output\n$4C \\times H \\times W$",
    )
    ax.set_title("HSDC block — Horizontally Stacked Dilated Convolution\n"
                 "(Stringhini et al., ICIP 2024)", fontsize=11)
    save(fig, "architecture", "hsdc_block")


def make_swhdc_block() -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_block_diagram(
        ax,
        color=PALETTE["lightred"],
        combine_box_text=("Latitude-weighted sum\n"
                          "$F^* = \\sum_n W_n(\\varphi)\\, F_n$\n"
                          "$W_n = \\min(N, 1/\\sin\\varphi)$"),
        output_text="Output\n$C \\times H \\times W$\n(no extra params)",
        combine_box_h=1.3,
        combine_box_w=2.2,
    )
    ax.set_title("SWHDC block — Spherically-Weighted HDC\n"
                 "(Stringhini et al., SIBGRAPI 2024)", fontsize=11)
    save(fig, "architecture", "swhdc_block")


def _resnet_layout(ax, *, title, total_blocks, conv_label, block_kind,
                   block_colour, head_text):
    """Compact ResNet diagram with one box per macro stage."""
    ax.set_xlim(0, 16); ax.set_ylim(0, 5); ax.axis("off")

    stages = [
        ("Input\n$10 \\times 256 \\times 512$",                  PALETTE["bg"]),
        (f"Conv1\n7$\\times$7 stride 2\n+ {conv_label}",           block_colour),
        (f"Conv2$_x$\n($n_2$ residual blocks)\n+ {block_kind}",     block_colour),
        (f"Conv3$_x$\n($n_3$ residual blocks)\n+ {block_kind}",     block_colour),
        (f"Conv4$_x$\n($n_4$ residual blocks)\n+ {block_kind}",     block_colour),
        (f"Conv5$_x$\n($n_5$ residual blocks)\n+ {block_kind}",     block_colour),
        (head_text,                                              PALETTE["bg"]),
    ]

    box_w, box_h, gap = 2.0, 1.8, 0.20
    y = 1.6
    cursor = 0.2
    for txt, fc in stages:
        add_box(ax, cursor, y, box_w, box_h, txt, fc=fc, fontsize=8)
        cursor += box_w + gap

    cursor = 0.2
    for _ in range(len(stages) - 1):
        x_src = cursor + box_w
        x_dst = cursor + box_w + gap
        add_arrow(ax, x_src, y + box_h / 2, x_dst, y + box_h / 2, lw=1.3)
        cursor += box_w + gap

    resolutions = ["", "$128 \\times 256$", "$64 \\times 128$",
                   "$32 \\times 64$", "$16 \\times 32$", "$8 \\times 16$",
                   "softmax"]
    cursor = 0.2
    for r in resolutions:
        if r:
            ax.text(cursor + box_w / 2, y - 0.25, r, ha="center", va="top",
                    fontsize=8, color=PALETTE["edge"], style="italic")
        cursor += box_w + gap

    cursor = 0.2 + 2 * (box_w + gap)
    for n in total_blocks:
        ax.text(cursor + box_w / 2, y + box_h + 0.2, f"$n={n}$",
                ha="center", va="bottom", fontsize=8, color=PALETTE["edge"])
        cursor += box_w + gap

    ax.set_title(title, fontsize=11)


def make_hsdcnet_backbone() -> None:
    fig, ax = plt.subplots(figsize=(16, 4))
    _resnet_layout(
        ax,
        title="HSDCNet — ResNet-34 backbone with HSDC blocks ($\\approx$ 5.5 M params)",
        total_blocks=[3, 4, 6, 3],
        conv_label="HSDC",
        block_kind="2 $\\times$ HSDC / block",
        block_colour=PALETTE["lightblue"],
        head_text="GAP\n+ Dropout\n+ FC",
    )
    save(fig, "architecture", "hsdcnet_backbone")


def make_swhdcresnet_backbone() -> None:
    fig, ax = plt.subplots(figsize=(16, 4))
    _resnet_layout(
        ax,
        title="SWHDCResNet — ResNet-50 backbone with SWHDC blocks ($\\approx$ 23.6 M params)",
        total_blocks=[3, 4, 6, 3],
        conv_label="SWHDC",
        block_kind="SWHDC at 3$\\times$3 conv",
        block_colour=PALETTE["lightred"],
        head_text="GAP\n+ Dropout\n+ FC",
    )
    save(fig, "architecture", "swhdcresnet_backbone")


# ===========================================================================
# Batch B — Theory figures
# ===========================================================================

def make_3d_representations() -> None:
    """Six conceptual panels: point cloud / mesh / voxel / SDF / NeRF / 3DGS."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))

    rng = np.random.default_rng(0)
    # Sample points roughly on a chair-like silhouette: a seat + four legs + back
    pts = []
    for _ in range(120):  # seat top
        pts.append((rng.uniform(-1, 1), rng.uniform(-0.5, 0.5), 0.5))
    for x in (-0.85, 0.85):
        for y in (-0.4, 0.4):
            for z in np.linspace(0, 0.5, 6):
                pts.append((x + rng.normal(0, 0.03), y + rng.normal(0, 0.03), z))
    for _ in range(80):  # back rest
        pts.append((rng.uniform(-1, 1), 0.45 + rng.normal(0, 0.03), rng.uniform(0.5, 1.7)))
    pts = np.array(pts)

    # Panel 1: point cloud
    ax = axes[0, 0]
    ax.scatter(pts[:, 0], pts[:, 2], s=4, c=PALETTE["hsdc"])
    ax.set_title("Point cloud", fontsize=11, fontweight="bold")
    ax.text(0.5, -0.08, "Unordered set of\n3D coordinates",
            transform=ax.transAxes, ha="center", va="top", fontsize=9)

    # Panel 2: mesh — draw triangulated wireframe approximation
    ax = axes[0, 1]
    from matplotlib.tri import Triangulation
    try:
        tri = Triangulation(pts[:, 0], pts[:, 2])
        ax.triplot(tri, color=PALETTE["hsdc"], lw=0.4)
    except Exception:
        ax.scatter(pts[:, 0], pts[:, 2], s=4, c=PALETTE["hsdc"])
    ax.set_title("Triangle mesh", fontsize=11, fontweight="bold")
    ax.text(0.5, -0.08, "Vertices + faces;\nconnectivity is explicit",
            transform=ax.transAxes, ha="center", va="top", fontsize=9)

    # Panel 3: voxels
    ax = axes[0, 2]
    cell = 0.12
    for px, pz in zip(pts[:, 0], pts[:, 2]):
        ix = int(round(px / cell))
        iz = int(round(pz / cell))
        ax.add_patch(Rectangle((ix * cell - cell / 2, iz * cell - cell / 2),
                               cell, cell,
                               facecolor=PALETTE["hsdc"], edgecolor="white", lw=0.4))
    ax.set_title("Voxel grid", fontsize=11, fontweight="bold")
    ax.text(0.5, -0.08, "Discrete 3D grid;\nuniform spacing",
            transform=ax.transAxes, ha="center", va="top", fontsize=9)

    # Panel 4: implicit / SDF (level set)
    ax = axes[1, 0]
    xx, zz = np.meshgrid(np.linspace(-1.6, 1.6, 200), np.linspace(-0.2, 2.2, 200))
    from scipy.spatial import KDTree
    tree = KDTree(pts[:, [0, 2]])
    d, _ = tree.query(np.c_[xx.ravel(), zz.ravel()], k=1)
    sdf = d.reshape(xx.shape) - 0.07
    ax.contour(xx, zz, sdf, levels=[-0.05, 0.05, 0.15, 0.3, 0.5],
               colors=[PALETTE["hsdc"], PALETTE["lightblue"], PALETTE["grey"],
                       PALETTE["grey"], PALETTE["grey"]], linewidths=[1.3, 1, 0.6, 0.6, 0.6])
    ax.contour(xx, zz, sdf, levels=[0], colors=[PALETTE["red"]], linewidths=2)
    ax.set_title("Implicit (SDF / NeRF)", fontsize=11, fontweight="bold")
    ax.text(0.5, -0.08, "Continuous field;\nzero level set = surface",
            transform=ax.transAxes, ha="center", va="top", fontsize=9)

    # Panel 5: NeRF — volumetric density
    ax = axes[1, 1]
    density = np.exp(-3 * np.clip(d.reshape(xx.shape), 0, None)) * 0.9
    ax.imshow(density, extent=(-1.6, 1.6, -0.2, 2.2), origin="lower",
              cmap="magma", aspect="equal")
    ax.set_title("Neural radiance field", fontsize=11, fontweight="bold")
    ax.text(0.5, -0.08, "$\\sigma(\\mathbf{p})$ + view-dependent\nradiance, learned MLP",
            transform=ax.transAxes, ha="center", va="top", fontsize=9)

    # Panel 6: 3DGS — anisotropic ellipses
    ax = axes[1, 2]
    rng2 = np.random.default_rng(1)
    for px, _py, pz in pts[::3]:
        w = rng2.uniform(0.06, 0.16)
        h = rng2.uniform(0.06, 0.16)
        angle = rng2.uniform(0, 180)
        ax.add_patch(Ellipse((px, pz), w, h, angle=angle,
                             facecolor=PALETTE["hsdc"], alpha=0.5,
                             edgecolor="none"))
    ax.set_title("3D Gaussian Splatting", fontsize=11, fontweight="bold")
    ax.text(0.5, -0.08, "Anisotropic 3D Gaussians:\n$\\mathbf{\\mu}, \\mathbf{\\Sigma}, \\alpha$",
            transform=ax.transAxes, ha="center", va="top", fontsize=9)

    for ax in axes.flat:
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-0.2, 2.2)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color("#999"); s.set_linewidth(0.6)

    fig.suptitle("Common explicit and implicit representations of 3D objects",
                 fontsize=12, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.subplots_adjust(hspace=0.45)
    save(fig, "theory", "3d_representations")


def make_gaussian_primitive() -> None:
    """3D anisotropic Gaussian primitive, two-panel wide layout.

    Left: the continuous kernel rendered as nested opaque 1/2/3-sigma
    level-set fills (orthographic projection, hand-built so the red
    principal-axis arrows are never occluded), floating above a faint
    ground grid.  Right: the kernel profile along the Mahalanobis
    distance, showing the opacity as the peak amplitude and the 3-sigma
    support beyond which the contribution is negligible.  The per-Gaussian
    parameters are described in the text, not in the figure.
    """
    import matplotlib.colors as mcolors

    fig = plt.figure(figsize=(7.4, 2.85))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0],
                          left=0.005, right=0.97, top=0.96, bottom=0.19,
                          wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    axp = fig.add_subplot(gs[0, 1])
    ax.set_aspect("equal"); ax.axis("off")

    # ------------------------------------------------------------------
    # Left panel — orthographic 3D scene (hand-built projection).
    # ------------------------------------------------------------------
    def _n(v):
        return v / np.linalg.norm(v)
    az, el = np.deg2rad(30), np.deg2rad(16)
    view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = _n(np.cross(np.array([0.0, 0.0, 1.0]), view))
    up = np.cross(view, right)
    M = np.vstack([right, up])                      # 2x3 world -> screen
    proj = lambda P: np.atleast_2d(P) @ M.T         # noqa: E731
    o2 = proj(np.zeros(3))[0]

    # Ground grid on the z = 0 plane (spatial reference).
    g_lo, g_hi, g_step = 0.0, 2.4, 0.6
    grid = np.arange(g_lo, g_hi + 1e-9, g_step)
    for gx in grid:
        p = proj(np.array([[gx, g_lo, 0], [gx, g_hi, 0]]))
        ax.plot(p[:, 0], p[:, 1], color="#dbe0e4", lw=0.8, zorder=0)
    for gy in grid:
        p = proj(np.array([[g_lo, gy, 0], [g_hi, gy, 0]]))
        ax.plot(p[:, 0], p[:, 1], color="#dbe0e4", lw=0.8, zorder=0)

    # World xyz axes from the origin.
    AXL = 2.6
    for vec3, name in [((1, 0, 0), "x"), ((0, 1, 0), "y"), ((0, 0, 1), "z")]:
        v = np.array(vec3, float)
        tip = proj(AXL * v)[0]
        ax.annotate("", xy=(tip[0], tip[1]), xytext=(o2[0], o2[1]),
                    arrowprops=dict(arrowstyle="-|>", color=PALETTE["grey"],
                                    lw=1.2, mutation_scale=11,
                                    shrinkA=0, shrinkB=0), zorder=1)
        lab = proj(1.08 * AXL * v)[0]
        ax.text(lab[0], lab[1], name, color=PALETTE["edge"], fontsize=10,
                style="italic", ha="center", va="center", zorder=1,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    # Gaussian: floating centre (deliberately off the origin).
    mu = np.array([1.30, 1.15, 1.50])
    s = np.array([0.56, 0.38, 0.26])
    ang = np.deg2rad([35, 30, 22])
    cx, sx = np.cos(ang[0]), np.sin(ang[0])
    cy, sy = np.cos(ang[1]), np.sin(ang[1])
    cz, sz = np.cos(ang[2]), np.sin(ang[2])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    A = R @ np.diag(s)
    mu2 = proj(mu)[0]

    # Dashed drop-line to the floor — the "floating in 3D" cue.
    foot = np.array([mu[0], mu[1], 0.0])
    pf = proj(np.vstack([mu, foot]))
    ax.plot(pf[:, 0], pf[:, 1], color=PALETTE["grey"], lw=1.0, ls=(0, (4, 3)),
            alpha=0.7, zorder=1)
    fd = proj(foot)[0]
    ax.plot(fd[0], fd[1], "o", color=PALETTE["grey"], ms=4, alpha=0.7, zorder=1)

    # Orthographic projection of the 3D Gaussian is a 2D Gaussian with
    # covariance C2 = M (A A^T) M^T.  Nested opaque level-set fills at
    # 3, 2 and 1 sigma convey the continuous density falloff without
    # any PDF transparency (colours are pre-composited over white).
    C2 = M @ (A @ A.T) @ M.T
    wv, Vv = np.linalg.eigh(C2)
    tt = np.linspace(0, 2 * np.pi, 240)
    ring = Vv @ (np.sqrt(wv)[:, None] * np.vstack([np.cos(tt), np.sin(tt)]))
    blue = np.array(mcolors.to_rgb(PALETTE["hsdc"]))
    for k_sig, shade in [(3.0, 0.16), (2.0, 0.34), (1.0, 0.58)]:
        col = tuple(1.0 - shade * (1.0 - blue))      # composite over white
        out = mu2 + (k_sig * ring).T
        ax.fill(out[:, 0], out[:, 1], color=col, lw=0, zorder=2)
    out3 = mu2 + (3.0 * ring).T
    ax.plot(out3[:, 0], out3[:, 1], color=PALETTE["hsdc"], lw=1.0, zorder=3)

    # Sigma level annotations along the long principal direction.
    dir_long = _n(proj(mu + R[:, 0])[0] - mu2)
    for k_sig, lab in [(1.0, r"$1\sigma$"), (2.0, r"$2\sigma$"),
                       (3.0, r"$3\sigma$")]:
        # point on the k-sigma outline along the long axis direction
        pt = mu2 + dir_long * k_sig * np.sqrt(dir_long @ C2 @ dir_long)
        ax.text(pt[0] + 0.07, pt[1] + 0.05, lab, fontsize=8,
                color=PALETTE["hsdc"], ha="left", va="bottom", zorder=7,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    # Principal-axis arrows, drawn from the centre to the 3-sigma outline
    # so they span the visible blob, with labels just outside the tips.
    for k, off in zip(range(3), [3.45, 3.45, 3.7]):
        tip = proj(mu + 3.0 * s[k] * R[:, k])[0]
        ax.annotate("", xy=(tip[0], tip[1]), xytext=(mu2[0], mu2[1]),
                    arrowprops=dict(arrowstyle="-|>", color=PALETTE["red"],
                                    lw=2.0, mutation_scale=13,
                                    shrinkA=0, shrinkB=0), zorder=6)
        lab = proj(mu + off * s[k] * R[:, k])[0]
        ax.text(lab[0], lab[1], f"$\\mathbf{{v}}_{k+1}$",
                color=PALETTE["red"], fontsize=9, fontweight="bold",
                ha="center", va="center", zorder=7,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    # Mean.
    ax.plot(mu2[0], mu2[1], "o", color=PALETTE["edge"], ms=5, zorder=8)
    ax.text(mu2[0] - 0.07, mu2[1] - 0.09, r"$\boldsymbol{\mu}$", fontsize=11,
            color=PALETTE["edge"], ha="right", va="center", zorder=8,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    # Centre the scene.
    bound = [np.zeros(3), np.array([AXL, 0, 0]), np.array([0, AXL, 0]),
             np.array([0, 0, AXL]), np.array([g_hi, g_hi, 0]), mu, foot]
    for k in range(3):
        bound.append(mu + 3.2 * s[k] * R[:, k])
        bound.append(mu - 3.2 * s[k] * R[:, k])
    B = proj(np.array(bound))
    xc = (B[:, 0].min() + B[:, 0].max()) / 2
    yc = (B[:, 1].min() + B[:, 1].max()) / 2
    half = max(np.ptp(B[:, 0]), np.ptp(B[:, 1])) / 2 * 1.06
    ax.set_xlim(xc - half, xc + half); ax.set_ylim(yc - half, yc + half)

    # ------------------------------------------------------------------
    # Right panel — kernel profile vs. Mahalanobis distance.
    # ------------------------------------------------------------------
    alpha_op = 0.8
    D = np.linspace(0.0, 4.2, 400)
    y = alpha_op * np.exp(-0.5 * D ** 2)
    axp.plot(D, y, color=PALETTE["hsdc"], lw=2.2, zorder=4)
    axp.fill_between(D, 0, y, where=D <= 3.0,
                     color=tuple(1.0 - 0.16 * (1.0 - blue)), lw=0, zorder=1)

    # Opacity caps the peak.
    axp.axhline(alpha_op, color=PALETTE["red"], lw=1.0, ls=(0, (4, 3)),
                zorder=3)
    axp.text(2.45, alpha_op + 0.025, r"opacity $\alpha$ (peak value)",
             fontsize=8.5, color=PALETTE["red"], ha="left", va="bottom")

    # Sigma marks and the negligible tail beyond 3 sigma.
    for k_sig in (1, 2, 3):
        yk = alpha_op * np.exp(-0.5 * k_sig ** 2)
        axp.plot([k_sig, k_sig], [0, yk], color=PALETTE["grey"], lw=0.9,
                 ls=(0, (3, 3)), zorder=2)
    axp.axvspan(3.0, 4.2, color="#f0f0f0", zorder=0)
    axp.text(3.6, 0.30, "negligible\ncontribution",
             fontsize=8.5, color=PALETTE["edge"], ha="center", va="center")
    axp.annotate("", xy=(3.0, 0.105), xytext=(3.6, 0.21),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["grey"],
                                 lw=1.0))

    axp.set_xlim(0, 4.2); axp.set_ylim(0, 0.97)
    axp.set_xticks([0, 1, 2, 3, 4])
    axp.set_xticklabels(["0", r"$1\sigma$", r"$2\sigma$", r"$3\sigma$",
                         r"$4\sigma$"])
    axp.set_yticks([0, 0.4, 0.8])
    axp.set_xlabel(r"Mahalanobis distance $D$ from the centre $\boldsymbol{\mu}$",
                   fontsize=9)
    axp.set_ylabel(r"contribution $\alpha\,e^{-D^2/2}$", fontsize=9)
    axp.tick_params(labelsize=8)

    save(fig, "theory", "gaussian_primitive")


def make_erp_camera() -> None:
    """Spherical camera model: 3D sphere -> 2D ERP grid."""
    fig = plt.figure(figsize=(11, 5))
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)

    # 3D sphere with great circles
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    uu, vv = np.meshgrid(u, v)
    R = 1.0
    x = R * np.cos(uu) * np.sin(vv)
    y = R * np.sin(uu) * np.sin(vv)
    z = R * np.cos(vv)
    ax3d.plot_wireframe(x, y, z, color=PALETTE["grey"], lw=0.3, alpha=0.5)
    # Highlight equator and meridian
    eq_u = np.linspace(0, 2 * np.pi, 200)
    ax3d.plot(np.cos(eq_u), np.sin(eq_u), np.zeros_like(eq_u),
              color=PALETTE["hsdc"], lw=1.8)
    ax3d.plot(np.cos(eq_u), np.zeros_like(eq_u), np.sin(eq_u),
              color=PALETTE["swhdc"], lw=1.8)

    # Sample direction d
    theta, phi = np.deg2rad(40), np.deg2rad(55)
    dx = np.sin(phi) * np.cos(theta)
    dy = np.sin(phi) * np.sin(theta)
    dz = np.cos(phi)
    ax3d.quiver(0, 0, 0, dx, dy, dz, color=PALETTE["red"], lw=2,
                arrow_length_ratio=0.12)
    ax3d.scatter([dx], [dy], [dz], color=PALETTE["red"], s=40)
    ax3d.text(dx * 1.1, dy * 1.1, dz * 1.05,
              r"$\mathbf{d}(\theta,\varphi)$", color=PALETTE["red"], fontsize=10)
    ax3d.text(0, -0.1, -1.18, "south pole ($\\varphi=\\pi$)", ha="center",
              fontsize=8, color=PALETTE["edge"])
    ax3d.text(0, 0, 1.18, "north pole ($\\varphi=0$)", ha="center",
              fontsize=8, color=PALETTE["edge"])
    ax3d.text(1.18, 0, 0, "equator", color=PALETTE["hsdc"], fontsize=8)

    ax3d.set_box_aspect((1, 1, 1))
    ax3d.view_init(elev=18, azim=30)
    ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
    ax3d.set_title("Spherical camera at object centroid", fontsize=11)

    # 2D ERP grid (unfolded sphere)
    H, W = 8, 16
    for i in range(H + 1):
        ax2d.axhline(i, color=PALETTE["grey"], lw=0.4)
    for j in range(W + 1):
        ax2d.axvline(j, color=PALETTE["grey"], lw=0.4)
    # Equator
    ax2d.axhline(H / 2, color=PALETTE["hsdc"], lw=2, label="equator $\\varphi=\\pi/2$")
    # Prime meridian
    ax2d.axvline(W / 2, color=PALETTE["swhdc"], lw=2, label="meridian $\\theta=0$")
    # Sample point (u, v) corresponding to the 3D direction
    u_sample = (theta + np.pi) / (2 * np.pi) * W
    v_sample = phi / np.pi * H
    ax2d.scatter([u_sample], [v_sample], color=PALETTE["red"], s=80, zorder=10)
    ax2d.text(u_sample + 0.3, v_sample + 0.2, "$(u, v)$",
              color=PALETTE["red"], fontsize=10)

    ax2d.set_xlim(0, W); ax2d.set_ylim(H, 0)
    ax2d.set_aspect("equal")
    ax2d.set_xlabel("column $u$  ($\\theta \\in [-\\pi, \\pi]$)")
    ax2d.set_ylabel("row $v$  ($\\varphi \\in [0, \\pi]$)")
    ax2d.set_xticks([0, W / 2, W])
    ax2d.set_xticklabels(["$-\\pi$", "$0$", "$\\pi$"])
    ax2d.set_yticks([0, H / 2, H])
    ax2d.set_yticklabels(["$0$", "$\\pi/2$", "$\\pi$"])
    ax2d.set_title("Equirectangular projection (ERP)", fontsize=11)
    ax2d.legend(loc="upper right", fontsize=8)

    fig.suptitle("Forward map from a sphere to a $H\\!\\times\\!W$ ERP image",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    save(fig, "theory", "erp_camera")


def make_latitude_distortion() -> None:
    """1/sin(phi) pixel-density curve + receptive-field illustration."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: 1/sin curve
    phi = np.linspace(np.deg2rad(2), np.deg2rad(178), 400)
    rho = 1.0 / np.sin(phi)
    ax = axes[0]
    ax.plot(np.rad2deg(phi), rho, color=PALETTE["hsdc"], lw=2.2,
            label=r"$1/\sin\varphi$")
    ax.axhline(1.0, color=PALETTE["grey"], ls="--", lw=1,
               label="equator (no stretch)")
    ax.fill_between(np.rad2deg(phi), 1, rho, where=rho > 1, alpha=0.15,
                    color=PALETTE["hsdc"])
    # Mark the latitudes drawn as indicatrices on the right panel, so the two
    # views read together: each dot is one row of ellipses.
    for phi_deg in (20, 35, 60, 90, 120, 145, 160):
        r = 1.0 / np.sin(np.deg2rad(phi_deg))
        ax.plot(phi_deg, r, "o", ms=5, color=PALETTE["swhdc"], zorder=5)
    ax.set_xlim(0, 180); ax.set_ylim(0.8, 8)
    ax.set_xlabel("polar angle $\\varphi$ (degrees)")
    ax.set_ylabel("relative horizontal stretch")
    ax.set_title("Latitude-dependent horizontal stretching")
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax.legend(loc="upper center")
    ax.axes.spines["top"].set_visible(False)
    ax.axes.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)

    # Right: Tissot's indicatrix on the ERP map.  Circles of equal geodesic
    # radius on the sphere (each "sees" the same physical neighbourhood) project
    # to ellipses whose horizontal axis grows as 1/sin(phi) toward the poles,
    # while their vertical axis stays fixed.  A naive square convolution kernel
    # is a constant box in this (lon, lat) grid, so near the poles it spans a far
    # wider sphere region than at the equator — exactly the mismatch HSDC/SWHDC
    # correct for.
    ax = axes[1]
    LON_MAX = 360.0
    ax.set_facecolor("white")

    # Latitudes (rows) drawn as indicatrices; matched to the red dots on the left.
    phi_rows = np.array([20, 35, 60, 90, 120, 145, 160])      # polar angle
    lat_rows = 90.0 - phi_rows                                # latitude for y-axis
    lon_cols = np.linspace(45, LON_MAX - 45, 4)

    # Graticule
    for lat in np.linspace(-90, 90, 13):
        ax.plot([0, LON_MAX], [lat, lat], color="#e0e6ea", lw=0.6, zorder=0)
    for lon in np.linspace(0, LON_MAX, 13):
        ax.plot([lon, lon], [-90, 90], color="#e0e6ea", lw=0.6, zorder=0)

    base_h = 7.0          # fixed vertical semi-extent of every indicatrix (deg)
    cmap = plt.cm.YlOrRd
    max_stretch = 1.0 / np.sin(np.deg2rad(phi_rows.min()))
    for phi_deg, lat in zip(phi_rows, lat_rows):
        stretch = 1.0 / np.sin(np.deg2rad(phi_deg))
        col = cmap(0.25 + 0.6 * (stretch - 1.0) / (max_stretch - 1.0))
        for lon in lon_cols:
            ell = plt.matplotlib.patches.Ellipse(
                (lon, lat), 2 * base_h * stretch, 2 * base_h,
                facecolor=col, edgecolor=PALETTE["edge"], lw=0.7,
                alpha=0.9, zorder=2)
            ax.add_patch(ell)

    # Call out the equator (circle) vs near-pole (wide ellipse) contrast, with
    # text placed in the empty graticule gaps so it never overlaps an ellipse.
    ax.annotate("equator:\nstays circular ($\\times 1$)",
                xy=(lon_cols[1] + base_h, 0), xytext=(LON_MAX * 0.5, 17),
                ha="center", va="bottom", fontsize=8, color=PALETTE["edge"],
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color=PALETTE["edge"], lw=0.7))
    ax.annotate("near pole:\nstretched $\\times\\,1/\\sin\\varphi$",
                xy=(lon_cols[0], lat_rows[0] - base_h),
                xytext=(LON_MAX * 0.5, 47),
                ha="center", va="center", fontsize=8, color=PALETTE["swhdc"],
                fontweight="bold", annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color=PALETTE["swhdc"], lw=0.7))

    ax.set_xlim(0, LON_MAX); ax.set_ylim(-95, 95)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax.set_xlabel("longitude (degrees)")
    ax.set_ylabel("latitude (degrees)")
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("#999"); s.set_linewidth(0.6)
    ax.set_title("Equal sphere neighbourhoods, stretched on the ERP map")

    fig.suptitle("Why naive 2D convolution is ill-matched to ERP",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    save(fig, "theory", "latitude_distortion")


def make_swhdc_latitude_weights() -> None:
    """W_n(phi) = min(N, 1/sin(phi)) for N = 1..4."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    phi = np.linspace(np.deg2rad(0.5), np.deg2rad(179.5), 600)
    colours = [PALETTE["hsdc"], PALETTE["resnet"], PALETTE["amber"],
               PALETTE["swhdc"]]
    for N, c in zip([1, 2, 3, 4], colours):
        w = np.minimum(N, 1.0 / np.sin(phi))
        ax.plot(np.rad2deg(phi), w, color=c, lw=2, label=f"$N={N}$")
    ax.set_xlim(0, 180); ax.set_ylim(0, 5)
    ax.set_xlabel("polar angle $\\varphi$ (degrees)")
    ax.set_ylabel("dilation weight $W_n(\\varphi)$")
    ax.set_title("SWHDC latitude-dependent dilation weights "
                 "$W_n(\\varphi) = \\min(N,\\, 1/\\sin\\varphi)$")
    ax.legend(title="branch index $n$", loc="upper center", ncol=4)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    save(fig, "theory", "swhdc_latitude_weights")


def make_egonerf_shells() -> None:
    """Linear vs exponential (EgoNeRF) shell spacing."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    N = 8
    r_near, r_far = 0.2, 1.0

    # Linear
    radii_lin = np.linspace(r_near, r_far, N)
    # Exponential (EgoNeRF)
    radii_exp = r_near * (r_far / r_near) ** (np.arange(N) / (N - 1))

    for ax, radii, title, colour in zip(
            axes,
            [radii_lin, radii_exp],
            ["Uniform spacing\n$r_s = r_{\\rm near} + s \\cdot \\Delta r$",
             "Exponential spacing (EgoNeRF)\n"
             "$r_s = r_{\\rm near}(r_{\\rm far}/r_{\\rm near})^{s/(N-1)}$"],
            [PALETTE["grey"], PALETTE["hsdc"]]):
        for r in radii:
            ax.add_patch(Circle((0, 0), r, fill=False, edgecolor=colour, lw=1.5))
        # Object silhouette (illustrative)
        ax.add_patch(Circle((0, 0), 0.05, color=PALETTE["red"], zorder=10))
        ax.text(0, -0.09, "$\\mathbf{C}$", ha="center", fontsize=10,
                color=PALETTE["red"])
        # Shell labels: place at evenly spread angles around the circle so the
        # labels for adjacent shells do not overlap, even for tightly spaced
        # inner shells in the exponential schedule.
        angles_deg = np.linspace(20, 160, len(radii))
        for s_idx, (r, ang_deg) in enumerate(zip(radii, angles_deg)):
            ang = np.deg2rad(ang_deg)
            lx = r * np.cos(ang)
            ly = r * np.sin(ang)
            ax.plot([0, lx], [0, ly], color=colour, lw=0.4, ls=":",
                    alpha=0.4)
            ax.annotate(f"$r_{s_idx}$",
                        xy=(lx, ly),
                        xytext=(lx * 1.08, ly * 1.08),
                        fontsize=9, color=colour,
                        ha="center", va="center")
        ax.set_xlim(-r_far - 0.1, r_far + 0.1)
        ax.set_ylim(-r_far - 0.1, r_far + 0.1)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)

    fig.suptitle("Radial sampling locations: $N=8$ shells",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    save(fig, "theory", "egonerf_shells")


# ===========================================================================
# Batch C — Methodology figures (uses real data)
# ===========================================================================

def _load_ply(category: str, split: str, name: str):
    from src.preprocessing.ply_loader import load_gaussian_ply
    p = PLY_ROOT / category / split / name / "point_cloud.ply"
    return load_gaussian_ply(p)


def _load_erp(category: str, split: str, name: str) -> np.ndarray:
    return np.load(ERP_CACHE / category / split / f"{name}.npy")


def make_rf_erp_pipeline_steps() -> None:
    """Real objects: PLY render -> centroid + shells overlay -> 8-shell ERP.

    Three classes (airplane, car, flower_pot) from gs_data/. The 8-shell ERPs
    are loaded from the production cache under data/processed/modelnet40/.
    """
    samples = [
        ("airplane",   "train", "airplane_0001"),
        ("car",        "train", "car_0001"),
        ("flower_pot", "train", "flower_pot_0001"),
    ]
    mn40_cache = (ROOT / "data" / "processed" / "modelnet40" /
                  "radiance_field" / "ns8_H256_W512_c3.0_p5.0-95.0")

    rows = []
    for cat, split, name in samples:
        from src.preprocessing.ply_loader import load_gaussian_ply
        g = load_gaussian_ply(PLY_ROOT / cat / split / name / "point_cloud.ply")
        erp = np.load(mn40_cache / cat / split / f"{name}.npy")
        rows.append((cat, g, erp))

    fig = plt.figure(figsize=(15, 10.5))
    n_rows = len(rows)
    n_cols = 3  # 1: 3D points, 2: 2D projection + shells, 3: 8-shell ERP grid

    column_titles = [
        "(a) 3DGS PLY",
        "(b) Centroid + 8 exp. shells (xz-projection)",
        "(c) 8-shell RF-ERP (log-scaled density)",
    ]

    for r_idx, (cat, g, erp) in enumerate(rows):
        # Column 1 — 3D point cloud render
        ax = fig.add_subplot(n_rows, n_cols, r_idx * n_cols + 1, projection="3d")
        xyz = g["xyz"]
        rgb = np.clip(g["rgb"], 0, 1)
        sub = np.random.default_rng(0).choice(len(xyz),
                                              size=min(4000, len(xyz)),
                                              replace=False)
        ax.scatter(xyz[sub, 0], xyz[sub, 1], xyz[sub, 2],
                   c=rgb[sub], s=2, alpha=0.6)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=45)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        # Row label on the left of column 1
        ax.text2D(-0.12, 0.5,
                  f"{cat}\n({len(xyz):,} Gaussians)",
                  transform=ax.transAxes,
                  rotation=90, ha="center", va="center",
                  fontsize=11, fontweight="bold")
        # Column header only on the top row
        if r_idx == 0:
            ax.set_title(column_titles[0], fontsize=11, pad=14)

        # Column 2 — 2D projection (xz plane) + centroid + shells
        ax = fig.add_subplot(n_rows, n_cols, r_idx * n_cols + 2)
        alpha = g["opacity"]
        centroid = (alpha[:, None] * xyz).sum(0) / alpha.sum()
        radii = np.linalg.norm(xyz - centroid, axis=1)
        r_near = np.percentile(radii, 5)
        r_far  = np.percentile(radii, 95)
        shells = r_near * (r_far / r_near) ** (np.arange(8) / 7)

        ax.scatter(xyz[sub, 0] - centroid[0], xyz[sub, 2] - centroid[2],
                   c=rgb[sub], s=1.5, alpha=0.45)
        for s_r in shells:
            ax.add_patch(Circle((0, 0), s_r, fill=False,
                                edgecolor=PALETTE["hsdc"], lw=1.0, alpha=0.8))
        ax.scatter([0], [0], color=PALETTE["red"], s=40, zorder=10)
        ax.text(0, -r_far * 0.07, "$\\mathbf{C}$", ha="center",
                color=PALETTE["red"], fontsize=10, fontweight="bold")
        lim = r_far * 1.1
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        if r_idx == 0:
            ax.set_title(column_titles[1], fontsize=11, pad=14)

        # Column 3 — 8-shell ERP grid
        ax = fig.add_subplot(n_rows, n_cols, r_idx * n_cols + 3)
        N_sh, H_e, W_e = erp.shape
        pad = 6  # blank pixels between tiles to host the per-shell label
        grid = np.zeros((H_e * 2 + pad, W_e * 4 + 3 * pad))
        for i in range(N_sh):
            rr, cc = divmod(i, 4)
            y0 = rr * (H_e + pad // 2)
            x0 = cc * (W_e + pad)
            grid[y0:y0 + H_e, x0:x0 + W_e] = erp[i]
        im = ax.imshow(np.log1p(grid), cmap="magma", aspect="auto")
        # Per-shell label written into a top corner of each tile in white
        # with a black outline so it stays readable on bright and dark ERP
        # regions alike.
        for i in range(N_sh):
            rr, cc = divmod(i, 4)
            y0 = rr * (H_e + pad // 2)
            x0 = cc * (W_e + pad)
            t = ax.text(x0 + 6, y0 + 14,
                        f"shell {i}", color="white", fontsize=8,
                        ha="left", va="top", fontweight="bold")
            t.set_path_effects([
                pe.withStroke(linewidth=2.0, foreground="black"),
            ])
        ax.set_xticks([]); ax.set_yticks([])
        if r_idx == 0:
            ax.set_title(column_titles[2], fontsize=11, pad=14)

    fig.suptitle(
        "RF-ERP preprocessing: real ModelSplat objects through the pipeline",
        fontsize=13, y=0.995,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.subplots_adjust(hspace=0.18, wspace=0.12, left=0.06)
    save(fig, "methodology", "rf_erp_pipeline_steps")


def make_log_compression() -> None:
    """Empirical justification for the log1p density transform.

    Aggregates the raw shell-density values of a random sample of ModelNet10
    objects from the production cache and shows, in three panels, why an
    element-wise ``log(1 + rho)`` compression is applied:

      (a) the raw density is extremely heavy-tailed (a large spike at rho=0
          plus a long right tail), so a linear input would let a handful of
          high-density splats dominate the dynamic range;
      (b) after log1p the populated densities occupy a well-spread, bounded
          band, stabilising the input statistics;
      (c) the transfer curve log(1+rho) has its steepest slope near rho=0
          (derivative 1/(1+rho)), which amplifies the low-density boundary
          signal while compressing the high-density tail.

    Statistics are computed once and printed so the numbers can be quoted in
    the text. The cache stores the RAW density ERP (log1p is applied at load
    time in the dataset), so the values here are the un-transformed densities.
    """
    import glob
    import random

    files = glob.glob(str(ERP_CACHE / "*" / "*" / "*.npy"))
    if not files:
        print("  ** no ERP cache found; skipping log_compression figure",
              file=sys.stderr)
        return
    random.Random(0).shuffle(files)
    files = files[:150]

    # Aggregate a subsampled pixel population (keep memory modest).
    rng = np.random.default_rng(0)
    pooled: list[np.ndarray] = []
    per_obj_max: list[float] = []
    for f in files:
        a = np.load(f).astype(np.float32).ravel()
        per_obj_max.append(float(a.max()))
        # Subsample 60k pixels/object so the histogram is representative
        # without holding the full 1M-pixel ERP for every object.
        idx = rng.choice(a.size, size=min(60_000, a.size), replace=False)
        pooled.append(a[idx])
    rho = np.concatenate(pooled)
    nz = rho[rho > 1e-6]

    frac_empty = float((rho <= 1e-6).mean())
    p99 = float(np.percentile(rho, 99))
    rho_max = float(rho.max())
    ratio = rho_max / max(p99, 1e-6)
    print(f"  log1p stats | empty={frac_empty:.1%}  p99={p99:.2f}  "
          f"max={rho_max:.2f}  max/p99={ratio:.1f}  "
          f"log1p(max)={np.log1p(rho_max):.2f}  "
          f"median per-obj max={np.median(per_obj_max):.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # --- (a) raw density histogram, log-scaled counts -------------------
    ax = axes[0]
    ax.hist(rho, bins=120, range=(0, np.percentile(rho, 99.95)),
            color=PALETTE["hsdc"], edgecolor="none", alpha=0.85)
    ax.set_yscale("log")
    ax.axvline(p99, color=PALETTE["red"], lw=1.4, ls="--")
    ax.text(p99, ax.get_ylim()[1] * 0.5, f"  $p_{{99}}\\approx{p99:.0f}$",
            color=PALETTE["red"], fontsize=9, va="top")
    ax.set_xlabel(r"raw shell density $\rho$")
    ax.set_ylabel("pixel count (log)")
    ax.set_title(r"(a) Raw density $\rho$")
    ax.text(0.97, 0.92,
            f"{frac_empty:.0%} of pixels empty\n"
            f"$\\rho_{{\\max}}\\approx{rho_max:.0f}$ "
            f"($\\approx{ratio:.0f}\\times\\,p_{{99}}$)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=PALETTE["grey"], alpha=0.9))

    # --- (b) log1p density histogram ------------------------------------
    ax = axes[1]
    ax.hist(np.log1p(nz), bins=120, color=PALETTE["resnet"],
            edgecolor="none", alpha=0.85)
    ax.axvline(np.log1p(p99), color=PALETTE["red"], lw=1.4, ls="--")
    ax.set_xlabel(r"$\tilde\rho = \log(1+\rho)$")
    ax.set_ylabel("populated-pixel count")
    ax.set_title(r"(b) After $\log(1+\rho)$")
    ax.text(0.97, 0.92,
            f"range $[0,\\;{np.log1p(rho_max):.1f}]$\n"
            f"bulk in $[0,\\;{np.log1p(p99):.1f}]$",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=PALETTE["grey"], alpha=0.9))

    # --- (c) transfer curve ---------------------------------------------
    ax = axes[2]
    xs = np.linspace(0, rho_max, 400)
    ax.plot(xs, np.log1p(xs), color=PALETTE["edge"], lw=2.0,
            label=r"$\log(1+\rho)$")
    ax.plot([0, rho_max], [0, rho_max / rho_max * np.log1p(rho_max)],
            color=PALETTE["grey"], lw=1.0, ls=":", alpha=0.0)  # spacer
    # Mark how p99 and max map through the curve
    for x, lab, col in [(p99, "$p_{99}$", PALETTE["red"]),
                        (rho_max, "$\\rho_{\\max}$", PALETTE["swhdc"])]:
        y = float(np.log1p(x))
        ax.plot([x, x], [0, y], color=col, lw=1.0, ls="--", alpha=0.7)
        ax.plot([0, x], [y, y], color=col, lw=1.0, ls="--", alpha=0.7)
        ax.scatter([x], [y], color=col, s=22, zorder=5)
    ax.set_xlabel(r"raw density $\rho$")
    ax.set_ylabel(r"$\tilde\rho$")
    ax.set_title(r"(c) Transfer curve $\log(1+\rho)$")

    plt.tight_layout()
    save(fig, "methodology", "log_compression")


def make_spatial_culling() -> None:
    """Spatial-culling 3σ illustration on a real airplane PLY."""
    g = _load_ply("airplane", "train", "airplane_0001")
    xyz = g["xyz"]
    alpha = g["opacity"]
    scales = g["scale"]
    centroid = (alpha[:, None] * xyz).sum(0) / alpha.sum()
    radii = np.linalg.norm(xyz - centroid, axis=1)
    r_near = np.percentile(radii, 5)
    r_far  = np.percentile(radii, 95)
    shells = r_near * (r_far / r_near) ** (np.arange(8) / 7)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: bird's-eye view with one shell highlighted and the 3-sigma band
    ax = axes[0]
    sub = np.random.default_rng(0).choice(len(xyz), size=4000, replace=False)
    in_band = np.zeros(len(xyz), dtype=bool)
    target_shell = shells[3]
    sigma = 3 * scales.max(axis=1)
    in_band = np.abs(radii - target_shell) < sigma
    ax.scatter(xyz[sub, 0] - centroid[0], xyz[sub, 2] - centroid[2],
               c=np.where(in_band[sub], PALETTE["red"], PALETTE["grey"]),
               s=2, alpha=0.55)
    ax.add_patch(Circle((0, 0), target_shell, fill=False,
                        edgecolor=PALETTE["hsdc"], lw=2, label="shell $s$"))
    ax.add_patch(Circle((0, 0), target_shell + np.median(sigma), fill=False,
                        edgecolor=PALETTE["hsdc"], lw=1, ls="--",
                        label="$\\pm 3\\sigma$ band"))
    ax.add_patch(Circle((0, 0), max(0.0, target_shell - np.median(sigma)),
                        fill=False, edgecolor=PALETTE["hsdc"], lw=1, ls="--"))
    lim = r_far * 1.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Per-shell pre-filter on $r_{\\rm dist}$")
    legend_handles = [
        mpatches.Patch(color=PALETTE["red"], label="Gaussian within $3\\sigma$ of shell"),
        mpatches.Patch(color=PALETTE["grey"], label="culled (skipped entirely)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

    # Right: histogram of |r_dist - r_s| with cutoff
    ax = axes[1]
    delta = np.abs(radii - target_shell)
    ax.hist(delta, bins=80, color=PALETTE["grey"], alpha=0.55,
            label="all Gaussians")
    ax.hist(delta[in_band], bins=80, color=PALETTE["red"], alpha=0.85,
            label="evaluated for shell $s$")
    ax.axvline(np.median(sigma), color=PALETTE["hsdc"], ls="--", lw=1.5,
               label="$3\\sigma$ cutoff")
    ax.set_xlabel("$|r_{\\rm dist} - r_s|$")
    ax.set_ylabel("Gaussian count")
    ax.set_title("3$\\sigma$-cutoff distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    speedup = len(xyz) / max(1, in_band.sum())
    fig.suptitle(
        f"Spatial culling reduces work by ~{speedup:.1f}$\\times$ at shell $s$ "
        f"(airplane / 3$\\sigma$ band)", fontsize=11, y=1.02)
    plt.tight_layout()
    save(fig, "methodology", "spatial_culling")


def _erp_to_rgb(erp: np.ndarray) -> np.ndarray:
    """8-channel ERP -> 1-channel pseudo-depth for display."""
    norm = erp / max(erp.max(), 1e-6)
    weights = np.arange(erp.shape[0])[:, None, None]
    depth = (norm * weights).sum(0) / max(1e-6, norm.sum(0).max())
    return depth / max(depth.max(), 1e-6)


def make_augmentation_samples() -> None:
    """Original + each augmentation on a real toilet ERP."""
    erp = _load_erp("toilet", "train", "toilet_0001")
    base = _erp_to_rgb(erp)
    H, W = base.shape

    rng = np.random.default_rng(2026)
    # Build augmented variants
    flip = base[:, ::-1]
    # 3D rotation: simulate as horizontal circular shift + slight vertical warp
    shift = W // 5
    rot = np.concatenate([base[:, shift:], base[:, :shift]], axis=1)
    # Blur via box filter
    from scipy.ndimage import gaussian_filter
    blur = gaussian_filter(base, sigma=2.5)
    noise = np.clip(base + rng.normal(0, 0.08, base.shape), 0, 1)
    erase = base.copy()
    h_e, w_e = 40, 60
    y_e = rng.integers(40, H - h_e - 40)
    x_e = rng.integers(40, W - w_e - 40)
    erase[y_e:y_e + h_e, x_e:x_e + w_e] = 0.0

    panels = [
        ("Original",                base),
        ("Horiz. flip\n(azimuth $\\pi$)", flip),
        ("3D rotation\n(circular shift)", rot),
        ("Gaussian blur",           blur),
        ("Gaussian noise",          noise),
        ("Random erasing",          erase),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.5))
    for ax, (title, img) in zip(axes.flat, panels):
        ax.imshow(img, cmap="magma", aspect="auto", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Single-sample augmentation primitives on a toilet RF-ERP",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    save(fig, "methodology", "augmentation_samples")


def make_mixup_cutmix() -> None:
    """MixUp + CutMix illustrated on two ERPs."""
    a = _erp_to_rgb(_load_erp("chair", "train", "chair_0001"))
    b = _erp_to_rgb(_load_erp("table", "train", "table_0001"))

    lam = 0.6
    mixup = lam * a + (1 - lam) * b
    cutmix = a.copy()
    H, W = a.shape
    ch, cw = H // 3, W // 3
    y0 = H // 2 - ch // 2
    x0 = W // 2 - cw // 2
    cutmix[y0:y0 + ch, x0:x0 + cw] = b[y0:y0 + ch, x0:x0 + cw]

    fig, axes = plt.subplots(2, 2, figsize=(10, 5.5))
    for ax, img, t in zip(axes.flat,
                          [a, b, mixup, cutmix],
                          ["Sample A (chair)", "Sample B (table)",
                           f"MixUp ($\\lambda = {lam}$)",
                           "CutMix (central patch)"]):
        ax.imshow(img, cmap="magma", aspect="auto", vmin=0, vmax=1)
        ax.set_title(t, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Sample-pair augmentation: MixUp and CutMix",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    save(fig, "methodology", "mixup_cutmix")


def make_per_class_erp_gallery() -> None:
    """One representative pseudo-depth per ModelNet10 class."""
    categories = ["bathtub", "bed", "chair", "desk", "dresser",
                  "monitor", "night_stand", "sofa", "table", "toilet"]
    fig, axes = plt.subplots(2, 5, figsize=(14, 5))
    for ax, cat in zip(axes.flat, categories):
        first = next(iter((ERP_CACHE / cat / "train").glob("*.npy")), None)
        if first is None:
            ax.axis("off"); continue
        erp = np.load(first)
        ax.imshow(_erp_to_rgb(erp), cmap="magma", aspect="auto", vmin=0, vmax=1)
        ax.set_title(cat, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Per-class pseudo-depth RF-ERP gallery — ModelNet10",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    save(fig, "data", "per_class_erp_gallery")


def make_per_class_gallery() -> None:
    """10 ModelNet10 classes (rows) x 8 density shells (columns)."""
    categories = ["bathtub", "bed", "chair", "desk", "dresser",
                  "monitor", "night_stand", "sofa", "table", "toilet"]
    n_rows = len(categories)
    n_cols = 8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.7, n_rows * 1.05))

    for r, cat in enumerate(categories):
        first = next(iter((ERP_CACHE / cat / "train").glob("*.npy")), None)
        if first is None:
            for c in range(n_cols):
                axes[r, c].axis("off")
            axes[r, 0].text(0.5, 0.5, f"{cat}\n(no cache)", ha="center",
                            va="center", transform=axes[r, 0].transAxes,
                            fontsize=9, color=PALETTE["red"])
            continue
        erp = np.load(first)
        # Per-row normalisation so faint shells are still visible
        vmax = max(erp.max(), 1e-6)
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(erp[c], cmap="magma", aspect="auto",
                      vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_linewidth(0.4); s.set_color("#bbb")
            if r == 0:
                ax.set_title(f"shell {c + 1}", fontsize=9)
        axes[r, 0].set_ylabel(cat, rotation=0, ha="right", va="center",
                              labelpad=22, fontsize=10)

    fig.suptitle("Per-class radiance-field ERP gallery — ModelNet10",
                 fontsize=12, y=1.0)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    save(fig, "results", "per_class_gallery")


# ===========================================================================
# Batch D — Setup figures
# ===========================================================================

def make_class_distribution() -> None:
    """Bar charts of training-set sample count per class.

    Counts are the canonical ModelNet train split sizes
    (Wu et al., CVPR 2015), used as the source of truth in the TCC.
    """
    MN10_COUNTS = {
        "bathtub": 106, "bed": 515, "chair": 889, "desk": 200,
        "dresser": 200, "monitor": 465, "night_stand": 200, "sofa": 680,
        "table": 392, "toilet": 344,
    }
    MN40_COUNTS = {
        "airplane": 626, "bathtub": 106, "bed": 515, "bench": 173,
        "bookshelf": 572, "bottle": 335, "bowl": 64, "car": 197,
        "chair": 889, "cone": 167, "cup": 79, "curtain": 138,
        "desk": 200, "door": 109, "dresser": 200, "flower_pot": 149,
        "glass_box": 171, "guitar": 155, "keyboard": 145, "lamp": 124,
        "laptop": 149, "mantel": 284, "monitor": 465, "night_stand": 200,
        "person": 88, "piano": 231, "plant": 240, "radio": 104,
        "range_hood": 115, "sink": 128, "sofa": 680, "stairs": 124,
        "stool": 90, "table": 392, "tent": 163, "toilet": 344,
        "tv_stand": 267, "vase": 475, "wardrobe": 87, "xbox": 103,
    }
    mn10 = list(MN10_COUNTS.keys())
    mn40 = list(MN40_COUNTS.keys())
    c10 = list(MN10_COUNTS.values())
    c40 = list(MN40_COUNTS.values())

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), gridspec_kw={"height_ratios": [1, 1.4]})

    ax = axes[0]
    ax.bar(mn10, c10, color=PALETTE["hsdc"])
    ax.set_title(f"ModelNet10 — training set ({sum(c10):,} objects)",
                 fontsize=11)
    ax.set_ylabel("# objects")
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.bar(mn40, c40, color=PALETTE["swhdc"])
    ax.set_title(f"ModelNet40 — training set ({sum(c40):,} objects)",
                 fontsize=11)
    ax.set_ylabel("# objects")
    ax.tick_params(axis="x", labelrotation=70, labelsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Class distribution of the ModelSplat training partitions",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    save(fig, "setup", "class_distribution")


def make_lr_schedule() -> None:
    """10-epoch linear warmup -> cosine annealing to lr_min."""
    base, lr_min, total, warmup = 1e-4, 1e-6, 500, 10
    epochs = np.arange(total + 1)
    lr = np.zeros_like(epochs, dtype=float)
    for e in epochs:
        if e < warmup:
            lr[e] = base * (e + 1) / warmup
        else:
            t = (e - warmup) / max(1, total - warmup)
            lr[e] = lr_min + 0.5 * (base - lr_min) * (1 + np.cos(np.pi * t))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, lr, color=PALETTE["hsdc"], lw=1.8)
    ax.axvspan(0, warmup, color=PALETTE["amber"], alpha=0.18,
               label="warmup")
    ax.axvspan(warmup, total, color=PALETTE["lightblue"], alpha=0.18,
               label="cosine anneal")
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_yscale("log")
    ax.set_title("Warmup + cosine learning-rate schedule "
                 "(base $10^{-4}$, floor $10^{-6}$)", fontsize=11)
    ax.set_xlim(0, total)
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    save(fig, "setup", "lr_schedule")


# ===========================================================================
# Batch E — Related-work taxonomy
# ===========================================================================

def make_method_taxonomy() -> None:
    """Bubble chart: x = input modality, y = MN40 accuracy, marker = arch family."""
    methods = [
        # (name, input_modality, mn40_oa, params_M, pretrained, family)
        ("PointNet",    "Point cloud",      89.2, 3.5,  False, "MLP"),
        ("PointNet++",  "Point cloud",      91.9, 1.7,  False, "MLP"),
        ("PointMLP",    "Point cloud",      94.1, 13.2, False, "MLP"),
        ("PointNeXt",   "Point cloud",      93.2, 4.5,  False, "MLP"),
        ("Point-BERT",  "Point cloud",      93.2, 22.1, True,  "Transformer"),
        ("Point-MAE",   "Point cloud",      93.2, 22.1, True,  "Transformer"),
        ("Gaussian-MAE","3DGS params",      93.35, 22.1, True, "Transformer"),
        ("GS-PT",       "PC (3DGS-aug.)",   94.4, 22.1, True,  "Transformer"),
        ("3DGPE",       "Point cloud",      93.6, 8.0,  False, "Gauss+Mamba"),
        ("HSDC (paper)","Geometric ERP",    93.9, 5.3,  False, "CNN"),
        ("SWHDC (paper)","Geometric ERP",   91.9, 25.5, False, "CNN"),
        ("HSDC (ours)", "RF-ERP",           87.64, 5.47, False, "CNN"),
        ("SWHDC (ours)","RF-ERP",           88.65, 23.55, False, "CNN"),
    ]

    families = {
        "MLP": ("o",  PALETTE["resnet"]),
        "Transformer": ("s", PALETTE["purple"]),
        "Gauss+Mamba": ("D", PALETTE["amber"]),
        "CNN": ("^", PALETTE["hsdc"]),
    }
    modalities = ["Point cloud", "PC (3DGS-aug.)", "3DGS params",
                  "Geometric ERP", "RF-ERP"]
    mod_x = {m: i for i, m in enumerate(modalities)}

    # Per-method (x_jitter, label_dx_pts, label_dy_pts) — hand-tuned to avoid
    # overlap in the cluttered "Point cloud" and "3DGS params" columns.
    layout = {
        "PointNet":     (-0.22,  8,  -2),
        "PointNet++":   (-0.22,  8,   2),
        "PointMLP":     ( 0.20,  8,   3),
        "PointNeXt":    ( 0.22,  8,  -8),
        "Point-BERT":   (-0.30, -55, -3),
        "Point-MAE":    (-0.05,  8,  -6),
        "Gaussian-MAE": ( 0.00,  9,   4),
        "GS-PT":        ( 0.00,  9,   4),
        "3DGPE":        ( 0.32,  8,   3),
        "HSDC (paper)": ( 0.00,  9,   3),
        "SWHDC (paper)":( 0.00,  9,  -8),
        "HSDC (ours)":  (-0.05,  9,  -10),
        "SWHDC (ours)": ( 0.05,  9,   4),
    }

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for name, modality, oa, p, pretrained, fam in methods:
        marker, color = families[fam]
        face = color if not pretrained else "white"
        edge = color
        jx, dx, dy = layout.get(name, (0.0, 8, 4))
        x = mod_x[modality] + jx
        size = 60 + 6 * p
        ax.scatter(x, oa, s=size, marker=marker, facecolor=face,
                   edgecolor=edge, linewidth=1.5, zorder=10)
        ax.annotate(name, (x, oa), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, color=PALETTE["edge"],
                    ha="left" if dx >= 0 else "right")

    # Horizontal accuracy bands
    for y, lab in [(90, "$90\\%$"), (93, "$93\\%$"), (95, "$95\\%$")]:
        ax.axhline(y, color=PALETTE["grey"], ls="--", lw=0.5, alpha=0.6)
        ax.text(len(modalities) - 0.45, y + 0.05, lab, fontsize=8,
                color=PALETTE["grey"])

    ax.set_xticks(list(range(len(modalities))))
    ax.set_xticklabels(modalities)
    ax.set_ylim(86, 96)
    ax.set_ylabel("ModelNet40 overall accuracy (\\%)")
    ax.set_title("Method taxonomy on ModelNet40\n"
                 "marker = architecture family, fill = from scratch, hollow = pretrained, "
                 "size = parameter count", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Legends
    fam_handles = [plt.Line2D([0], [0], marker=m, color="white",
                              markerfacecolor=c, markeredgecolor=c,
                              markersize=10, label=name, linestyle="")
                   for name, (m, c) in families.items()]
    leg1 = ax.legend(handles=fam_handles, title="Architecture",
                     loc="lower left", fontsize=9)
    ax.add_artist(leg1)
    fill_handles = [plt.Line2D([0], [0], marker="o", color="white",
                               markerfacecolor=PALETTE["grey"],
                               markeredgecolor=PALETTE["grey"],
                               markersize=10, label="from scratch", linestyle=""),
                    plt.Line2D([0], [0], marker="o", color="white",
                               markerfacecolor="white",
                               markeredgecolor=PALETTE["grey"],
                               markersize=10, label="pretrained", linestyle="")]
    ax.legend(handles=fill_handles, title="Training regime",
              loc="lower right", fontsize=9)

    plt.tight_layout()
    save(fig, "related", "method_taxonomy")


# ===========================================================================
# CLI
# ===========================================================================

ALL_FIGURES = {
    # Batch A
    "pipeline_overview":      make_pipeline_overview,
    "hsdc_block":             make_hsdc_block,
    "swhdc_block":            make_swhdc_block,
    "hsdcnet_backbone":       make_hsdcnet_backbone,
    "swhdcresnet_backbone":   make_swhdcresnet_backbone,
    # Batch B
    "3d_representations":     make_3d_representations,
    "gaussian_primitive":     make_gaussian_primitive,
    "erp_camera":             make_erp_camera,
    "latitude_distortion":    make_latitude_distortion,
    "swhdc_latitude_weights": make_swhdc_latitude_weights,
    "egonerf_shells":         make_egonerf_shells,
    # Batch C
    "rf_erp_pipeline_steps":  make_rf_erp_pipeline_steps,
    "log_compression":        make_log_compression,
    "spatial_culling":        make_spatial_culling,
    "augmentation_samples":   make_augmentation_samples,
    "mixup_cutmix":           make_mixup_cutmix,
    "per_class_erp_gallery":  make_per_class_erp_gallery,
    "per_class_gallery":      make_per_class_gallery,
    # Batch D
    "class_distribution":     make_class_distribution,
    "lr_schedule":            make_lr_schedule,
    # Batch E
    "method_taxonomy":        make_method_taxonomy,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None,
                        help="Render only the named figure(s); default = all.")
    args = parser.parse_args()

    todo = args.only if args.only else list(ALL_FIGURES.keys())
    for name in todo:
        if name not in ALL_FIGURES:
            print(f"!! unknown figure: {name}", file=sys.stderr)
            continue
        print(f"[{name}]")
        try:
            ALL_FIGURES[name]()
        except Exception as exc:  # pragma: no cover
            print(f"  ** failed: {exc}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
