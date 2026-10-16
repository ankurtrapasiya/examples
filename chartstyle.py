"""The blog's chart stylesheet. Every script calls cs.apply() first and cs.save() last.

The charts are SVG, inlined into the page at build time (astro.config.mjs), so text is real
text: the site's CSS sets its font and its colour for light and dark mode. Line colours are
fixed and chosen to read on both backgrounds. What this module fixes:
  - palette: maroon for the thing the chart is about, blue for the comparison, grey for context,
    then purple, amber, green, matching the colour language of the formulas
  - type: one size scale (title 11 bold, axis labels 10, ticks and legend 9), left-aligned
    numbered titles, legend without a frame
  - layout: left and bottom spines only, faint horizontal grid, air between stacked panels
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

MAROON, MAROON_LIGHT, BLUE, GREY = "#b6262c", "#e0676b", "#3b6ea5", "#a3a19c"
PURPLE, AMBER, GREEN = "#7c5fd3", "#c0781a", "#3f8f45"
INK = "#7d7b77"                 # text and spines: a mid grey that reads on both backgrounds
PALETTE = [MAROON, BLUE, GREY, PURPLE, AMBER, GREEN]
# the page defines these per theme (global.css); the hex is the light-mode fallback
CSS_VARS = {MAROON: "--chart-maroon", BLUE: "--chart-blue", GREY: "--chart-grey",
            PURPLE: "--chart-purple", AMBER: "--chart-amber", GREEN: "--chart-green"}


def css_colour(hex_):
    """'#b6262c' -> 'var(--chart-maroon, #b6262c)' for palette colours, unchanged otherwise."""
    h = hex_.lower()
    return f"var({CSS_VARS[h]}, {h})" if h in CSS_VARS else h


def apply():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "svg.fonttype": "none",                       # keep text as text, so the page can style it
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.labelsize": 10, "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": INK, "grid.alpha": 0.18, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "lines.linewidth": 1.6, "lines.markersize": 4,
        "legend.frameon": False, "legend.fontsize": 9, "legend.handlelength": 1.8,
        "legend.loc": "best",                         # the emptiest corner; save() moves it out if none is empty
        "axes.facecolor": "none", "figure.facecolor": "none",
    })


def title(ax, text, n=None):
    """A panel title, centred, numbered inline when the figure has several panels ("2. What
    10,000 dollars became…"). The page's CSS gives it the heading face and ink via the gid."""
    label = f"{n}. {text}" if n is not None else text
    t = ax.set_title(label, fontsize=11, loc="center", fontweight="bold", pad=10)
    t.set_gid("chart-title")


def out_dir(post):
    """Where a post's charts go. Inside the blog repo that is public/posts/<post>, where the site
    reads them; in the standalone examples repo, or anywhere else, it is charts/<post> next to
    the scripts (the Docker image mounts that folder)."""
    here = Path(__file__).resolve().parent
    blog = here.parent / "public" / "posts"
    return (blog if blog.is_dir() else here / "charts") / post


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        for t in (ax.title,):
            if t.get_text() and not t.get_gid():
                t.set_gid("chart-title")
        for lbl in (ax.xaxis.label, ax.yaxis.label):
            lbl.set_gid("chart-axis")
    _place_legends(fig)
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg:
            leg.set_gid("chart-legend")
    fig.tight_layout(h_pad=4.0)          # clear air between stacked panels
    fig.savefig(path, transparent=True, dpi=200)   # dpi only affects rasterised layers (big scatters)
    svg = path.read_text()
    for h, var in CSS_VARS.items():                 # palette colours follow the page's theme
        svg = svg.replace(f": {h}", f": var({var}, {h})")
    path.write_text(svg)
    _embed_points(fig, path)
    print("wrote", path)


def _place_legends(fig):
    """A legend never covers data. Matplotlib's "best" picks the emptiest corner; if every corner
    has data under it, the legend moves below the panel, in a row, under the x-axis title."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    below = 0
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            continue
        leg = ax.legend(handles, labels, loc="best")
        fig.canvas.draw()                                 # "best" is only decided at draw time
        box = leg.get_window_extent(r)
        pts = []
        for line in ax.get_lines():
            xy = line.get_xydata()
            if len(xy):
                pts.append(line.get_transform().transform(xy))
        for c in ax.collections:
            if hasattr(c, "get_offsets") and len(c.get_offsets()):
                pts.append(c.get_offset_transform().transform(np.asarray(c.get_offsets())))
        if pts:
            p = np.vstack(pts)
            inside = ((p[:, 0] >= box.x0) & (p[:, 0] <= box.x1) & (p[:, 1] >= box.y0) & (p[:, 1] <= box.y1)).sum()
            if inside > 0:
                ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.26),
                          ncol=min(len(handles), 2))
                below += 1
    if below:                                             # give the panels their height back
        fig.set_figheight(fig.get_figheight() + 0.6 * below)


def _embed_points(fig, path, limit=2500):
    """Write every labelled series' points into the SVG as JSON, in SVG coordinates, with the
    series colour and each panel's plotting box, so the page can show a tooltip for the point
    under the cursor. Big clouds are skipped: a tooltip on one of 50,000 dots says nothing."""
    from matplotlib.colors import to_hex
    k_ = 72.0 / fig.dpi                   # the SVG is drawn in points; transforms give pixels
    H = fig.bbox.height * k_
    series, panels = [], []
    for k, ax in enumerate(fig.axes):
        (x0, y0), (x1, y1) = ax.bbox.get_points() * k_
        panels.append({"x0": round(float(x0), 2), "x1": round(float(x1), 2), "y0": round(float(H - y1), 2), "y1": round(float(H - y0), 2),
                       "title": ax.get_title() or "", "xlabel": ax.get_xlabel() or "x", "ylabel": ax.get_ylabel() or "y",
                       "xlog": ax.get_xscale() == "log"})
        arts = [(a.get_label(), a.get_xydata(), to_hex(a.get_color())) for a in ax.get_lines()]
        for c in ax.collections:
            if hasattr(c, "get_offsets") and len(c.get_offsets()):
                fc = c.get_facecolor()
                arts.append((c.get_label(), np.asarray(c.get_offsets()), to_hex(fc[0]) if len(fc) else "#888888"))
        for label, xy, color in arts:
            if not label or label.startswith("_") or len(xy) == 0 or len(xy) > limit:
                continue
            xy = np.asarray(xy, dtype=float)
            xy = xy[np.isfinite(xy).all(axis=1)]
            if len(xy) == 0:
                continue
            order = np.argsort(xy[:, 0], kind="stable")
            xy = xy[order]
            disp = ax.transData.transform(xy) * k_
            if "Date" in type(getattr(ax.xaxis, "converter", None)).__name__:   # a date axis: send dates, not day numbers
                from matplotlib.dates import num2date
                xs = [num2date(v).strftime("%Y-%m-%d") for v in xy[:, 0]]
            else:
                xs = [float(f"{v:.6g}") for v in xy[:, 0]]
            series.append({
                "panel": k, "label": label, "color": css_colour(color),
                "x": xs, "y": [float(f"{v:.6g}") for v in xy[:, 1]],
                "sx": [round(float(v), 2) for v in disp[:, 0]], "sy": [round(float(H - v), 2) for v in disp[:, 1]],
            })
    if not series:
        return
    svg = path.read_text()
    payload = json.dumps({"panels": panels, "series": series}, separators=(",", ":")).replace("<", "&lt;")
    meta = "<metadata id=\"chart-data\">" + payload + "</metadata>"
    i = svg.index(">", svg.index("<svg")) + 1
    path.write_text(svg[:i] + meta + svg[i:])
