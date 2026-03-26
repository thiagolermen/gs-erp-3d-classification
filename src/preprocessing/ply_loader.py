"""
Load 3D Gaussian Splat PLY files (ShapeSplats/ModelSplats format).

PLY properties (in order): x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2,
f_rest_0..N (variable number, depends on spherical harmonic degree),
opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3

The parser reads binary little-endian PLY files produced by the 3DGS training
pipeline.  Raw stored values require the following transformations before use:
  - opacity:  sigmoid(raw_opacity)
  - scale:    exp(raw_scale)  (log-space storage)
  - rgb:      clip(0.5 + 0.28209479177387814 * f_dc, 0, 1)
              (SH DC coefficient to linear RGB; SH_C0 = 0.28209479177387814)

References:
    Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
    SIGGRAPH 2023.
    ShapeSplats/ModelNet_Splats dataset, HuggingFace.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# Zeroth-order SH coefficient (DC term): Y_0^0 = 1 / (2 * sqrt(pi))
_SH_C0: float = 0.28209479177387814


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_ply_header(data: bytes) -> tuple[int, list[str], int]:
    """Parse a binary-little-endian PLY header.

    Returns:
        n_vertices: number of vertices (Gaussians).
        properties: ordered list of property names for the vertex element.
        header_end: byte offset of the first data byte (after 'end_header\n').

    Raises:
        ValueError: If the PLY format is not binary_little_endian or if the
                    vertex element cannot be found.
    """
    # Decode header lines up to 'end_header'
    header_str = data[:4096].decode("ascii", errors="replace")
    if "end_header" not in header_str:
        # Try a larger window for files with many properties
        header_str = data[:16384].decode("ascii", errors="replace")

    lines = header_str.split("\n")

    format_ok = False
    n_vertices = 0
    in_vertex = False
    properties: list[str] = []
    header_end = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("format"):
            if "binary_little_endian" not in line:
                raise ValueError(
                    f"Only binary_little_endian PLY is supported; got: '{line}'"
                )
            format_ok = True
        elif line.startswith("element vertex"):
            n_vertices = int(line.split()[-1])
            in_vertex = True
        elif line.startswith("element") and in_vertex:
            # A new element — stop collecting vertex properties
            in_vertex = False
        elif line.startswith("property") and in_vertex:
            # e.g. "property float x" or "property float f_rest_0"
            parts = line.split()
            if len(parts) >= 3:
                properties.append(parts[-1])
        elif line == "end_header":
            # Byte offset: length of all lines up to and including 'end_header\n'
            header_end = sum(len(l) + 1 for l in lines[: i + 1])
            break

    if not format_ok:
        raise ValueError("PLY format line not found or not binary_little_endian.")
    if n_vertices == 0:
        raise ValueError("PLY file contains no vertices (element vertex = 0).")

    return n_vertices, properties, header_end


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_gaussian_ply(path: Path) -> dict:
    """Load a 3D Gaussian Splat PLY file and return processed Gaussian parameters.

    Reads the binary little-endian PLY, applies standard transformations to the
    raw stored values (sigmoid for opacity, exp for scale, SH DC to RGB), and
    returns a dictionary ready for radiance field computation.

    The number of f_rest_* properties is detected automatically from the header,
    so files trained with different SH degrees (0, 1, 2, 3) are all supported.

    Args:
        path: Absolute path to the point_cloud.ply file.

    Returns:
        A dict with:
            'xyz'         : (N, 3) float32 — Gaussian centre positions.
            'opacity'     : (N,)   float32 — sigmoid(raw_opacity), in [0, 1].
            'scale'       : (N, 3) float32 — exp(raw_scale), actual scale per axis.
            'rgb'         : (N, 3) float32 — SH DC to linear RGB, clipped to [0, 1].
            'rotation'    : (N, 4) float32 — raw quaternion (w, x, y, z), NOT
                            normalised (caller should normalise before use).
            'n_gaussians' : int             — number of Gaussians.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError:        If the file is not a valid binary_little_endian PLY,
                           or if required properties are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PLY file not found: '{path}'")

    with path.open("rb") as fh:
        raw = fh.read()

    n_vertices, properties, header_end = _parse_ply_header(raw)

    # Each property is a float32 (4 bytes)
    n_props = len(properties)
    expected_bytes = n_vertices * n_props * 4
    available_bytes = len(raw) - header_end

    if available_bytes < expected_bytes:
        raise ValueError(
            f"PLY data too short: expected {expected_bytes} bytes for "
            f"{n_vertices} vertices × {n_props} float32 props, "
            f"got {available_bytes} bytes."
        )

    # Fast binary read: treat all vertices as a flat float32 buffer
    flat = np.frombuffer(raw, dtype=np.float32, offset=header_end)
    # Trim to exact vertex data (ignore any trailing bytes)
    flat = flat[: n_vertices * n_props]
    data = flat.reshape(n_vertices, n_props)  # (N, n_props)

    # Build property-name → column index map
    prop_idx: dict[str, int] = {name: i for i, name in enumerate(properties)}

    # ------------------------------------------------------------------
    # Required properties — raise early with a clear message if missing
    # ------------------------------------------------------------------
    required = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3",
                "f_dc_0", "f_dc_1", "f_dc_2"]
    missing = [p for p in required if p not in prop_idx]
    if missing:
        raise ValueError(
            f"PLY file '{path}' is missing required properties: {missing}. "
            f"Available: {list(prop_idx.keys())}"
        )

    # ------------------------------------------------------------------
    # Extract xyz
    # ------------------------------------------------------------------
    xyz = data[:, [prop_idx["x"], prop_idx["y"], prop_idx["z"]]].astype(np.float32)

    # ------------------------------------------------------------------
    # Extract opacity: sigmoid(raw) — Kerbl et al. eq. (2)
    # ------------------------------------------------------------------
    raw_opacity = data[:, prop_idx["opacity"]].astype(np.float64)
    opacity = (1.0 / (1.0 + np.exp(-raw_opacity))).astype(np.float32)

    # ------------------------------------------------------------------
    # Extract scale: exp(raw) — stored in log-space
    # ------------------------------------------------------------------
    scale = np.exp(
        data[:, [prop_idx["scale_0"], prop_idx["scale_1"], prop_idx["scale_2"]]]
        .astype(np.float64)
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # Extract RGB from DC SH coefficients
    # f_dc are the zeroth-order SH coefficients; convert to colour:
    #   colour = 0.5 + SH_C0 * f_dc   (Kerbl et al., supplemental)
    # ------------------------------------------------------------------
    f_dc = data[
        :, [prop_idx["f_dc_0"], prop_idx["f_dc_1"], prop_idx["f_dc_2"]]
    ].astype(np.float32)
    rgb = np.clip(0.5 + _SH_C0 * f_dc, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Extract quaternion (w, x, y, z) — NOT normalised
    # ------------------------------------------------------------------
    rotation = data[
        :, [prop_idx["rot_0"], prop_idx["rot_1"],
            prop_idx["rot_2"], prop_idx["rot_3"]]
    ].astype(np.float32)

    return {
        "xyz": xyz,
        "opacity": opacity,
        "scale": scale,
        "rgb": rgb,
        "rotation": rotation,
        "n_gaussians": int(n_vertices),
    }
