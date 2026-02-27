"""
scripts/render_diagrams.py

Renders all Mermaid (.mmd) diagrams in docs/diagrams/ to PNG and SVG,
then copies the outputs to tcc/images/diagrams/ for use in the TCC document.

Requirements:
    Node.js >= 18  (https://nodejs.org)
    npx (bundled with npm — no global install needed)

Usage:
    python scripts/render_diagrams.py                     # PNG + SVG
    python scripts/render_diagrams.py --format png        # PNG only
    python scripts/render_diagrams.py --format svg        # SVG only
    python scripts/render_diagrams.py --width 3200        # custom width
    python scripts/render_diagrams.py --no-copy           # skip tcc/images copy

Output layout:
    docs/diagrams/rendered/    <- primary rendered files
    tcc/images/diagrams/       <- copy for LaTeX \\includegraphics{}
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ── Project root (two levels up from this script) ─────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIAGRAMS_DIR = PROJECT_ROOT / "docs" / "diagrams"
RENDERED_DIR = DIAGRAMS_DIR / "rendered"
TCC_IMAGES_DIR = PROJECT_ROOT / "tcc" / "images" / "diagrams"
CONFIG_FILE = DIAGRAMS_DIR / "mmdc_config.json"

# ── Mermaid CLI invocation ─────────────────────────────────────────────────────
MMDC_PACKAGE = "@mermaid-js/mermaid-cli"


def _npx_cmd() -> list[str]:
    """Return the base npx command for the current OS."""
    if sys.platform == "win32":
        return ["npx.cmd", "--yes", MMDC_PACKAGE]
    return ["npx", "--yes", MMDC_PACKAGE]


def check_node() -> bool:
    """Return True if Node.js >= 14 is available."""
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        version = result.stdout.strip().lstrip("v")
        major = int(version.split(".")[0])
        if major < 14:
            print(f"[warn] Node.js {version} found — v14+ recommended for mmdc")
        else:
            print(f"[ok]  Node.js {version}")
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return False


def render_one(
    mmd_path: Path,
    out_dir: Path,
    fmt: str,
    width: int,
    bg_color: str,
) -> Path | None:
    """Render a single .mmd file to `fmt` (png or svg).

    Returns the output path on success, None on failure.
    """
    out_path = out_dir / mmd_path.with_suffix(f".{fmt}").name

    cmd = _npx_cmd() + [
        "--input",  str(mmd_path),
        "--output", str(out_path),
    ]

    if CONFIG_FILE.exists():
        cmd += ["--configFile", str(CONFIG_FILE)]

    if fmt == "png":
        cmd += [
            "--width",           str(width),
            "--backgroundColor", bg_color,
        ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            print(f"  [fail] {mmd_path.name}")
            print(f"         stdout: {result.stdout.strip()}")
            print(f"         stderr: {result.stderr.strip()}")
            return None
        print(f"  [ok]  {out_path.name}")
        return out_path
    except subprocess.TimeoutExpired:
        print(f"  [timeout] {mmd_path.name}")
        return None
    except FileNotFoundError:
        print("[error] npx not found — make sure Node.js is installed")
        return None


def render_all(
    fmt: str = "both",
    width: int = 2400,
    bg_color: str = "white",
    copy_to_tcc: bool = True,
) -> None:
    mmd_files = sorted(DIAGRAMS_DIR.glob("*.mmd"))
    if not mmd_files:
        print(f"No .mmd files found in {DIAGRAMS_DIR}")
        return

    if not check_node():
        print(
            "\n[error] Node.js not found.\n"
            "  Install from https://nodejs.org and re-run this script.\n"
            "  Alternatively, paste .mmd content into https://mermaid.live "
            "  to export manually."
        )
        _generate_html_preview(mmd_files)
        return

    RENDERED_DIR.mkdir(parents=True, exist_ok=True)
    if copy_to_tcc:
        TCC_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    formats = ["png", "svg"] if fmt == "both" else [fmt]
    rendered: list[Path] = []

    print(f"\nRendering {len(mmd_files)} diagram(s) -> {RENDERED_DIR}\n")
    for mmd in mmd_files:
        for f in formats:
            out = render_one(mmd, RENDERED_DIR, f, width, bg_color)
            if out:
                rendered.append(out)

    if copy_to_tcc and rendered:
        print(f"\nCopying {len(rendered)} file(s) -> {TCC_IMAGES_DIR}")
        for src in rendered:
            dst = TCC_IMAGES_DIR / src.name
            shutil.copy2(src, dst)
            print(f"  -> {dst.name}")

    print(f"\nDone.  {len(rendered)} file(s) rendered.")
    _print_latex_snippets(rendered)


def _generate_html_preview(mmd_files: list[Path]) -> None:
    """Fallback: generate a single HTML file with embedded Mermaid.js
    that can be opened in a browser for manual export."""
    html_path = RENDERED_DIR / "preview.html"
    RENDERED_DIR.mkdir(parents=True, exist_ok=True)

    diagrams_html = ""
    for mmd in mmd_files:
        content = mmd.read_text(encoding="utf-8")
        # Strip %%{init:...}%% directive for browser rendering (optional)
        stem = mmd.stem
        diagrams_html += f"""
    <section>
      <h2>{stem}</h2>
      <div class="mermaid">
{content}
      </div>
    </section>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>ERP-ViT Diagram Preview</title>
  <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
  <script>mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});</script>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; }}
    section {{ margin-bottom: 60px; border-top: 2px solid #ccc; padding-top: 20px; }}
    h2 {{ color: #333; }}
    .mermaid {{ background: #fafafa; padding: 20px; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>ERP-ViT Architecture Diagrams</h1>
  <p>Open this file in a browser to view all diagrams.<br/>
     Right-click any diagram and "Save image as..." to export manually.</p>
{diagrams_html}
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    print(f"\n[fallback] HTML preview generated: {html_path}")
    print("  Open in a browser to view and manually export diagrams.")


def _print_latex_snippets(rendered: list[Path]) -> None:
    """Print ready-to-use LaTeX \\includegraphics{} snippets."""
    pngs = [p for p in rendered if p.suffix == ".png"]
    if not pngs:
        return

    print("\n-- LaTeX \\includegraphics snippets " + "-" * 38)
    for p in pngs:
        latex_path = f"images/diagrams/{p.name}"
        stem = p.stem.replace("_", " ").title()
        print(f"""
\\begin{{figure}}[htb]
  \\caption{{{stem}}}
  \\begin{{center}}
    \\includegraphics[width=0.95\\textwidth]{{{latex_path}}}
  \\end{{center}}
  \\legend{{Source: The Authors.}}
  \\label{{fig:{p.stem}}}
\\end{{figure}}""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render Mermaid diagrams to PNG/SVG for the TCC document."
    )
    parser.add_argument(
        "--format",
        choices=["png", "svg", "both"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=2400,
        help="PNG output width in pixels (default: 2400)",
    )
    parser.add_argument(
        "--bg",
        default="white",
        help="PNG background colour (default: white)",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Skip copying outputs to tcc/images/diagrams/",
    )
    args = parser.parse_args()

    render_all(
        fmt=args.format,
        width=args.width,
        bg_color=args.bg,
        copy_to_tcc=not args.no_copy,
    )
