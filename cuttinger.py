#!/usr/bin/env python3
"""
Setzkasten pipeline without OpenSCAD.

What this script does
---------------------
1) Loads a scan STL.
2) Applies the STL transform you already found.
3) Reads the transformed mesh bounds.
4) Automatically sizes the box so the mesh sits centered inside it.
5) Computes planar slice contours for every divider using trimesh/shapely.
6) Writes laser-ready SVG files for:
   - outer panels (back, top, bottom, left, right)
   - vertical dividers with object cutouts
   - horizontal dividers with object cutouts
   - one combined layout.svg with all parts arranged

Install
-------
    pip install numpy trimesh shapely svgwrite

Usage
-----
    python setzkasten_boxes_pipeline.py

Output
------
Files are written to ./box_output/

Notes
-----
- Coordinates are in mm.
- The STL transform below uses the values you already validated.
- The script auto-centers the transformed STL in the box interior.
- The outer box size is derived from the mesh size plus configurable padding.
- If a clean closed section polygon is not available, the script falls back
  to a hull generated from the section points.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import svgwrite
import trimesh
from shapely.affinity import translate
from shapely.geometry import LineString, MultiPoint, MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union

try:
    from shapely import concave_hull  # Shapely 2.x
except Exception:
    concave_hull = None


# -----------------------------------------------------------------------------
# Input STL and transform
# -----------------------------------------------------------------------------
stl_file = "model.stl"
size_scale = 1.05 # 5% size increase to give more clearance, and place for a sheet of felt or foam on the divider faces if desired.
stl_scale = [size_scale, size_scale, size_scale]
stl_rot_deg = [18.0, 0.0, 0.0]
stl_pos = [-200.0, 0.0, 300.0]

# -----------------------------------------------------------------------------
# Auto-fit box parameters
# -----------------------------------------------------------------------------
# Padding from mesh to inner box walls
pad_x = 5.0
pad_y = 5.0
pad_z = 5.0

# Divider grid
cells_x = 4
cells_y = 3

# Material / manufacturing
mat = 4.0
kerf = 0.15
finger_target = 18.0
fit_clearance = 0.8
simplify_tol = 0.15
min_polygon_area = 5.0
concave_ratio = 0.4  # smaller = tighter hull, larger = smoother hull


divider_section_samples = 5
remove_undercuts = True
top_insert_step = 0.5
top_insert_support_step = 0.5
top_insert_side_margin = 1.0
top_insert_smooth_window = 8 # disabled by default, can cause issues with small dividers or tight fit
debug_top_contact_svg = False

# Output
output_dir = Path("box_output")
svg_stroke_width = 1.5
layout_spacing = 15.0
layout_max_row_width = 1200.0


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def oddify(n: int) -> int:
    return n + 1 if n % 2 == 0 else n


def finger_count(length: float) -> int:
    return oddify(max(3, int(math.floor(length / finger_target))))


def finger_pitch(length: float) -> float:
    return length / finger_count(length)


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]],
        dtype=float,
    )


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]],
        dtype=float,
    )


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=float,
    )


def scale_matrix(s: Sequence[float]) -> np.ndarray:
    sx, sy, sz = s
    return np.array(
        [[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]],
        dtype=float,
    )


def translate_matrix(t: Sequence[float]) -> np.ndarray:
    tx, ty, tz = t
    return np.array(
        [[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]],
        dtype=float,
    )


def openscad_like_transform(
    scale_xyz: Sequence[float],
    rot_xyz_deg: Sequence[float],
    pos_xyz: Sequence[float],
) -> np.ndarray:
    rx, ry, rz = rot_xyz_deg
    return (
        translate_matrix(pos_xyz)
        @ rot_z(rz)
        @ rot_y(ry)
        @ rot_x(rx)
        @ scale_matrix(scale_xyz)
    )


def transform_points(
    points: Iterable[Tuple[float, float]],
    dx: float = 0.0,
    dy: float = 0.0,
    flip_y_height: float | None = None,
) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for x, y in points:
        if flip_y_height is not None:
            y = flip_y_height - y
        out.append((x + dx, y + dy))
    return out


def catmull_rom_spline_points(
    points: Sequence[Tuple[float, float]],
    samples_per_segment: int = 12,
    edge_linear_segments: int = 3,
) -> List[Tuple[float, float]]:
    """Interpolate a smooth open curve through points without edge overshoot.

    Uses a shape-preserving cubic Hermite interpolation over x/y samples.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return [tuple(p) for p in pts]

    # Remove consecutive duplicates so slope calculations stay stable.
    deduped = [pts[0]]
    for p in pts[1:]:
        if not np.allclose(p, deduped[-1]):
            deduped.append(p)
    pts = np.asarray(deduped, dtype=float)

    if len(pts) < 2:
        return [tuple(p) for p in pts]
    if len(pts) == 2:
        a, b = pts
        samples = max(2, int(samples_per_segment))
        return [tuple((1.0 - t) * a + t * b) for t in np.linspace(0.0, 1.0, samples + 1)]

    x = pts[:, 0]
    y = pts[:, 1]
    h = np.diff(x)

    # If x is not strictly increasing, fall back to the original points.
    if np.any(h <= 0.0):
        return [tuple(p) for p in pts]

    delta = np.diff(y) / h
    d = np.zeros_like(y)

    # Endpoint slopes: one-sided, then clamped for shape preservation.
    d[0] = delta[0]
    d[-1] = delta[-1]
    if len(y) > 2:
        d[0] = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
        d[-1] = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])

    def _clamp_endpoint(idx: int, slope: float, secant: float) -> float:
        if secant == 0.0:
            return 0.0
        if slope * secant <= 0.0:
            return 0.0
        return float(np.clip(slope, -3.0 * abs(secant), 3.0 * abs(secant)))

    d[0] = _clamp_endpoint(0, d[0], delta[0])
    d[-1] = _clamp_endpoint(-1, d[-1], delta[-1])

    # Interior slopes: Fritsch-Carlson weighted harmonic mean.
    for i in range(1, len(y) - 1):
        if delta[i - 1] == 0.0 or delta[i] == 0.0 or delta[i - 1] * delta[i] <= 0.0:
            d[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    samples = max(2, int(samples_per_segment))
    edge_linear_segments = max(0, int(edge_linear_segments))
    out: List[Tuple[float, float]] = []
    for i in range(len(x) - 1):
        hi = h[i]
        x0 = x[i]
        y0 = y[i]
        y1 = y[i + 1]
        m0 = d[i]
        m1 = d[i + 1]
        use_linear = i < edge_linear_segments or i >= (len(x) - 1 - edge_linear_segments)

        for j in range(samples):
            t = j / float(samples)
            xx = x0 + hi * t
            if use_linear:
                yy = (1.0 - t) * y0 + t * y1
            else:
                t2 = t * t
                t3 = t2 * t
                h00 = 2.0 * t3 - 3.0 * t2 + 1.0
                h10 = t3 - 2.0 * t2 + t
                h01 = -2.0 * t3 + 3.0 * t2
                h11 = t3 - t2
                yy = h00 * y0 + h10 * hi * m0 + h01 * y1 + h11 * hi * m1
            out.append((float(xx), float(yy)))

    out.append((float(x[-1]), float(y[-1])))
    return out


def shift_geometry(geom: BaseGeometry, dx: float, dy: float) -> BaseGeometry:
    """
    Shift a shapely geometry by dx/dy.
    """
    return translate(geom, xoff=dx, yoff=dy)


def iter_polygons(geom: BaseGeometry):
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            yield from iter_polygons(poly)
    elif hasattr(geom, "geoms"):
        for child in geom.geoms:
            yield from iter_polygons(child)


def polygon_to_svg_points(poly: Polygon, height: float, dx: float = 0.0, dy: float = 0.0):
    ext = transform_points(poly.exterior.coords, dx=dx, dy=dy, flip_y_height=height)
    holes = [transform_points(ring.coords, dx=dx, dy=dy, flip_y_height=height) for ring in poly.interiors]
    return ext, holes


# -----------------------------------------------------------------------------
# SVG writer
# -----------------------------------------------------------------------------
def save_svg(width: float, height: float, path: Path, border: BaseGeometry) -> None:
    dwg = svgwrite.Drawing(
        str(path),
        size=(f"{width}mm", f"{height}mm"),
        viewBox=f"0 0 {width} {height}",
    )
    grp = dwg.g(fill="none", stroke="black", stroke_width=svg_stroke_width)

    for poly in iter_polygons(border):
        ext, holes = polygon_to_svg_points(poly, height)
        grp.add(dwg.polygon(points=ext))
        for hole in holes:
            grp.add(dwg.polygon(points=hole))

    dwg.add(grp)
    dwg.save()


def save_svg_with_polyline(
    width: float,
    height: float,
    path: Path,
    border: BaseGeometry,
    polyline: List[Tuple[float, float]],
) -> None:
    """Save border plus one debug polyline overlay (red)."""
    dwg = svgwrite.Drawing(
        str(path),
        size=(f"{width}mm", f"{height}mm"),
        viewBox=f"0 0 {width} {height}",
    )

    grp_border = dwg.g(fill="none", stroke="black", stroke_width=svg_stroke_width)
    for poly in iter_polygons(border):
        ext, holes = polygon_to_svg_points(poly, height)
        grp_border.add(dwg.polygon(points=ext))
        for hole in holes:
            grp_border.add(dwg.polygon(points=hole))
    dwg.add(grp_border)

    if len(polyline) >= 2:
        dbg_pts = transform_points(polyline, flip_y_height=height)
        grp_dbg = dwg.g(fill="none", stroke="red", stroke_width=1.0)
        grp_dbg.add(dwg.polyline(points=dbg_pts))
        dwg.add(grp_dbg)

    dwg.save()


def save_vdiv_debug_svg(
    width: float,
    height: float,
    path: Path,
    border: BaseGeometry,
    section_points: np.ndarray,
    cutline: List[Tuple[float, float]],
) -> None:
    """Save detailed VDIV debug drawing: border, raw section points, and cutline."""
    dwg = svgwrite.Drawing(
        str(path),
        size=(f"{width}mm", f"{height}mm"),
        viewBox=f"0 0 {width} {height}",
    )

    grp_border = dwg.g(fill="none", stroke="black", stroke_width=svg_stroke_width)
    for poly in iter_polygons(border):
        ext, holes = polygon_to_svg_points(poly, height)
        grp_border.add(dwg.polygon(points=ext))
        for hole in holes:
            grp_border.add(dwg.polygon(points=hole))
    dwg.add(grp_border)

    if section_points.size > 0:
        pts = np.asarray(section_points, dtype=float)
        max_pts = 4000
        if len(pts) > max_pts:
            idx = np.linspace(0, len(pts) - 1, max_pts, dtype=int)
            pts = pts[idx]
        pts_svg = transform_points(pts.tolist(), flip_y_height=height)
        grp_pts = dwg.g(fill="#1d4ed8", stroke="none", opacity=0.45)
        for px, py in pts_svg:
            grp_pts.add(dwg.circle(center=(px, py), r=0.35))
        dwg.add(grp_pts)

    if len(cutline) >= 2:
        line_svg = transform_points(cutline, flip_y_height=height)
        grp_cut = dwg.g(fill="none", stroke="#dc2626", stroke_width=1.2)
        grp_cut.add(dwg.polyline(points=line_svg))
        dwg.add(grp_cut)

    dwg.save()


# -----------------------------------------------------------------------------
# Finger-joint panel outlines
# -----------------------------------------------------------------------------
def edge_tabs_horizontal(y: float, length: float, outward: int, mode: str) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = [(0.0, y)]
    n = finger_count(length)
    p = finger_pitch(length)
    tab_depth = mat + kerf

    if mode == "flat":
        pts.append((length, y))
        return pts
    if mode == "male":
        depth = outward * tab_depth
    elif mode == "female":
        depth = -outward * tab_depth
    else:
        raise ValueError(f"Unknown edge mode: {mode}")

    for i in range(n):
        x0 = i * p
        x1 = (i + 1) * p
        on = i % 2 == 0
        if on:
            pts.extend([(x0, y), (x0, y + depth), (x1, y + depth), (x1, y)])
        else:
            pts.append((x1, y))
    return pts


def edge_tabs_horizontal_centered(y: float, length: float, outward: int, mode: str) -> List[Tuple[float, float]]:
    """Generate horizontal tabs centered symmetrically around the middle."""
    n = finger_count(length)
    p = finger_pitch(length)
    tab_depth = mat + kerf

    if mode == "flat":
        return [(0.0, y), (length, y)]
    
    if mode == "male":
        depth = outward * tab_depth
    elif mode == "female":
        depth = -outward * tab_depth
    else:
        raise ValueError(f"Unknown edge mode: {mode}")

    # Calculate offset so first finger starts at center
    center = length / 2.0
    offset = center - (n // 2) * p
    
    pts: List[Tuple[float, float]] = [(offset, y)]
    
    for i in range(n):
        x0 = offset + i * p
        x1 = offset + (i + 1) * p
        on = i % 2 == 0
        if on:
            pts.extend([(x0, y), (x0, y + depth), (x1, y + depth), (x1, y)])
        else:
            pts.append((x1, y))
    
    return pts


def edge_tabs_vertical(x: float, length: float, outward: int, mode: str) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = [(x, 0.0)]
    n = finger_count(length)
    p = finger_pitch(length)
    tab_depth = mat + kerf

    if mode == "flat":
        pts.append((x, length))
        return pts
    if mode == "male":
        depth = outward * tab_depth
    elif mode == "female":
        depth = -outward * tab_depth
    else:
        raise ValueError(f"Unknown edge mode: {mode}")

    for i in range(n):
        y0 = i * p
        y1 = (i + 1) * p
        on = i % 2 == 0
        if on:
            pts.extend([(x, y0), (x + depth, y0), (x + depth, y1), (x, y1)])
        else:
            pts.append((x, y1))
    return pts


def panel_outline(
    w: float,
    h: float,
    top_mode: str,
    bottom_mode: str,
    left_mode: str,
    right_mode: str,
    panel_name: str = "",
) -> Polygon:
    pts: List[Tuple[float, float]] = []

    bottom = edge_tabs_horizontal(0.0, w, outward=-1, mode=bottom_mode)
    pts.extend(bottom)

    right_base = edge_tabs_vertical(w, h, outward=+1, mode=right_mode)
    pts.extend(right_base[1:])

    top = edge_tabs_horizontal(h, w, outward=+1, mode=top_mode)
    top_rev = [(w - x, y) for x, y in top]
    pts.extend(top_rev[1:])

    left = [(x, h - y) for x, y in edge_tabs_vertical(0.0, h, outward=-1, mode=left_mode)]
    pts.extend(left[1:])

    poly = Polygon(pts)
    return poly.buffer(0) if not poly.is_valid else poly


# -----------------------------------------------------------------------------
# STL slicing
# -----------------------------------------------------------------------------
def load_mesh_raw() -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(stl_file, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh.copy()


def transformed_mesh(mesh: trimesh.Trimesh, pos_xyz: Sequence[float]) -> trimesh.Trimesh:
    out = mesh.copy()
    out.apply_transform(openscad_like_transform(stl_scale, stl_rot_deg, pos_xyz))
    return out


def section_polygons(
    mesh: trimesh.Trimesh,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    axis_map: str,
) -> List[Polygon]:
    """
    Try to get a clean planar section first.
    If that fails or produces no polygons, fall back to a hull from section points.
    """
    section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    if section is None:
        return []

    out: List[Polygon] = []

    # First choice: intersect plane with mesh directly in world coordinates.
    # This avoids ambiguous local 2D section frames and keeps rotation handling
    # physically correct for divider contact extraction.
    try:
        segs = trimesh.intersections.mesh_plane(
            mesh=mesh,
            plane_normal=np.asarray(plane_normal, dtype=float),
            plane_origin=np.asarray(plane_origin, dtype=float),
        )

        if segs is not None and len(segs) > 0:
            lines_world: List[LineString] = []
            for seg in np.asarray(segs, dtype=float):
                if seg.shape != (2, 3):
                    continue
                a, b = seg[0], seg[1]
                if axis_map == "xy":
                    a2, b2 = (a[0], a[1]), (b[0], b[1])
                elif axis_map == "yz":
                    a2, b2 = (a[1], a[2]), (b[1], b[2])
                elif axis_map == "xz":
                    a2, b2 = (a[0], a[2]), (b[0], b[2])
                else:
                    raise ValueError(axis_map)

                if np.allclose(a2, b2):
                    continue
                lines_world.append(LineString([a2, b2]))

            if lines_world:
                polys_world = list(polygonize(unary_union(lines_world)))
                for p in polys_world:
                    if simplify_tol > 0:
                        p = p.simplify(simplify_tol, preserve_topology=True)
                    if fit_clearance != 0:
                        p = p.buffer(fit_clearance, join_style=1)
                    if p.is_empty:
                        continue
                    if isinstance(p, MultiPolygon):
                        out.extend([g for g in p.geoms if g.area >= min_polygon_area])
                    elif p.area >= min_polygon_area:
                        out.append(p)

                if out:
                    return out
    except Exception:
        pass

    # Prefer deterministic world-axis projection from discrete section lines.
    # `section.to_2D()` can choose an arbitrary in-plane basis which may rotate
    # or mirror the cut profile, causing wrong insertion-side behavior.
    try:
        lines_world: List[LineString] = []
        for polyline in section.discrete:
            pts = np.asarray(polyline, dtype=float)
            if pts.shape[0] < 2:
                continue

            if axis_map == "xy":
                pts2 = pts[:, [0, 1]]
            elif axis_map == "yz":
                pts2 = pts[:, [1, 2]]
            elif axis_map == "xz":
                pts2 = pts[:, [0, 2]]
            else:
                raise ValueError(axis_map)

            for a, b in zip(pts2[:-1], pts2[1:]):
                if not np.allclose(a, b):
                    lines_world.append(LineString([tuple(a), tuple(b)]))

        if lines_world:
            polys_world = list(polygonize(unary_union(lines_world)))
            for p in polys_world:
                if simplify_tol > 0:
                    p = p.simplify(simplify_tol, preserve_topology=True)
                if fit_clearance != 0:
                    p = p.buffer(fit_clearance, join_style=1)
                if p.is_empty:
                    continue
                if isinstance(p, MultiPolygon):
                    out.extend([g for g in p.geoms if g.area >= min_polygon_area])
                elif p.area >= min_polygon_area:
                    out.append(p)

            if out:
                return out
    except Exception:
        pass

    def path_lines(path: trimesh.path.Path2D) -> List[LineString]:
        lines: List[LineString] = []
        for entity in path.entities:
            try:
                pts = np.asarray(entity.discrete(path.vertices), dtype=float)
            except Exception:
                continue
            if pts.shape[0] < 2:
                continue
            for a, b in zip(pts[:-1], pts[1:]):
                if not np.allclose(a, b):
                    lines.append(LineString([tuple(a), tuple(b)]))
        return lines

    # 1) Best case: trimesh can provide true planar polygons
    try:
        planar, _ = section.to_2D()
        polys = list(planar.polygons_full)
        for p in polys:
            if simplify_tol > 0:
                p = p.simplify(simplify_tol, preserve_topology=True)
            if fit_clearance != 0:
                p = p.buffer(fit_clearance, join_style=1)
            if p.is_empty:
                continue
            if isinstance(p, MultiPolygon):
                out.extend([g for g in p.geoms if g.area >= min_polygon_area])
            elif p.area >= min_polygon_area:
                out.append(p)

        if out:
            return out

        lines = path_lines(planar)
        if lines:
            polys = list(polygonize(unary_union(lines)))
            for p in polys:
                if simplify_tol > 0:
                    p = p.simplify(simplify_tol, preserve_topology=True)
                if fit_clearance != 0:
                    p = p.buffer(fit_clearance, join_style=1)
                if p.is_empty:
                    continue
                if isinstance(p, MultiPolygon):
                    out.extend([g for g in p.geoms if g.area >= min_polygon_area])
                elif p.area >= min_polygon_area:
                    out.append(p)

            if out:
                return out

        pts_2d = [tuple(p) for p in np.asarray(planar.vertices, dtype=float)]
        if len(pts_2d) >= 3:
            mp = MultiPoint(pts_2d)

            try:
                if concave_hull is not None:
                    hull = concave_hull(mp, ratio=concave_ratio)
                else:
                    hull = mp.convex_hull
            except Exception:
                hull = mp.convex_hull

            if not hull.is_empty:
                if simplify_tol > 0:
                    hull = hull.simplify(simplify_tol, preserve_topology=True)
                if fit_clearance != 0:
                    hull = hull.buffer(fit_clearance, join_style=1)

                if isinstance(hull, MultiPolygon):
                    out.extend([g for g in hull.geoms if g.area >= min_polygon_area])
                elif isinstance(hull, Polygon) and hull.area >= min_polygon_area:
                    out.append(hull)

                if out:
                    return out
    except Exception:
        pass

    # 2) Fallback: line polygonization
    try:
        lines: List[LineString] = []
        for polyline in section.discrete:
            pts = np.asarray(polyline, dtype=float)
            if pts.shape[0] < 2:
                continue

            if axis_map == "xy":
                pts2 = pts[:, [0, 1]]
            elif axis_map == "yz":
                pts2 = pts[:, [1, 2]]
            elif axis_map == "xz":
                pts2 = pts[:, [0, 2]]
            else:
                raise ValueError(axis_map)

            for a, b in zip(pts2[:-1], pts2[1:]):
                if not np.allclose(a, b):
                    lines.append(LineString([tuple(a), tuple(b)]))

        polys = list(polygonize(unary_union(lines)))
        for p in polys:
            if simplify_tol > 0:
                p = p.simplify(simplify_tol, preserve_topology=True)
            if fit_clearance != 0:
                p = p.buffer(fit_clearance, join_style=1)
            if p.is_empty:
                continue
            if isinstance(p, MultiPolygon):
                out.extend([g for g in p.geoms if g.area >= min_polygon_area])
            elif p.area >= min_polygon_area:
                out.append(p)

        if out:
            return out
    except Exception:
        pass

    # 3) Final fallback: hull of section points
    pts_2d: List[Tuple[float, float]] = []
    for polyline in section.discrete:
        pts = np.asarray(polyline, dtype=float)
        if pts.shape[0] < 2:
            continue

        if axis_map == "xy":
            pts2 = pts[:, [0, 1]]
        elif axis_map == "yz":
            pts2 = pts[:, [1, 2]]
        elif axis_map == "xz":
            pts2 = pts[:, [0, 2]]
        else:
            raise ValueError(axis_map)

        pts_2d.extend([tuple(p) for p in pts2])

    if len(pts_2d) < 3:
        return []

    mp = MultiPoint(pts_2d)

    try:
        if concave_hull is not None:
            hull = concave_hull(mp, ratio=concave_ratio)
        else:
            hull = mp.convex_hull
    except Exception:
        hull = mp.convex_hull

    if hull.is_empty:
        return []

    if simplify_tol > 0:
        hull = hull.simplify(simplify_tol, preserve_topology=True)
    if fit_clearance != 0:
        hull = hull.buffer(fit_clearance, join_style=1)

    if hull.is_empty:
        return []

    if isinstance(hull, MultiPolygon):
        return [g for g in hull.geoms if g.area >= min_polygon_area]
    if isinstance(hull, Polygon) and hull.area >= min_polygon_area:
        return [hull]

    return []


def sampled_section_polygons(
    mesh: trimesh.Trimesh,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    axis_map: str,
    thickness: float,
    samples: int,
) -> List[Polygon]:
    """
    Collect section polygons across a slab thickness around the divider center plane.
    This creates more reliable clearance for real divider thickness.
    """
    n = max(int(samples), 1)
    normal = np.asarray(plane_normal, dtype=float)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return []
    normal = normal / norm

    if n == 1 or thickness <= 0:
        offsets = [0.0]
    else:
        offsets = np.linspace(-thickness / 2.0, thickness / 2.0, n)

    polys: List[Polygon] = []
    for off in offsets:
        origin = plane_origin + normal * float(off)
        polys.extend(section_polygons(mesh, origin, normal, axis_map))

    if not polys:
        return []

    merged = unary_union(polys).buffer(0)
    out = [p for p in iter_polygons(merged) if p.area >= min_polygon_area]
    return out

def sampled_section_points(
    mesh: trimesh.Trimesh,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    axis_map: str,
    thickness: float,
    samples: int,
) -> np.ndarray:
    """Collect world-space 2D section points across the divider thickness slab."""
    n = max(int(samples), 1)
    normal = np.asarray(plane_normal, dtype=float)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return np.empty((0, 2), dtype=float)
    normal = normal / norm

    if n == 1 or thickness <= 0:
        offsets = [0.0]
    else:
        offsets = np.linspace(-thickness / 2.0, thickness / 2.0, n)

    pts_out: List[np.ndarray] = []
    for off in offsets:
        origin = plane_origin + normal * float(off)
        try:
            segs = trimesh.intersections.mesh_plane(
                mesh=mesh,
                plane_normal=normal,
                plane_origin=origin,
            )
        except Exception:
            segs = None

        if segs is None or len(segs) == 0:
            continue

        segs = np.asarray(segs, dtype=float)
        if axis_map == "xy":
            pts2 = segs[:, :, [0, 1]].reshape(-1, 2)
        elif axis_map == "yz":
            pts2 = segs[:, :, [1, 2]].reshape(-1, 2)
        elif axis_map == "xz":
            pts2 = segs[:, :, [0, 2]].reshape(-1, 2)
        else:
            raise ValueError(axis_map)

        if len(pts2) > 0:
            pts_out.append(pts2)

    if not pts_out:
        return np.empty((0, 2), dtype=float)

    return np.vstack(pts_out)


def build_divider_cut_geometry(cuts: List[Polygon], clip_rect: Polygon) -> BaseGeometry:
    """
    Build one robust cut geometry from many slice polygons.
    """
    clipped: List[BaseGeometry] = []
    for c in cuts:
        g = c.intersection(clip_rect)
        if not g.is_empty:
            clipped.append(g)

    if not clipped:
        return Polygon()

    geom = unary_union(clipped).buffer(0)

    if remove_undercuts:
        # Conservative cleanup: remove concave re-entrant areas that cause undercuts.
        hulls = [p.convex_hull for p in iter_polygons(geom) if not p.is_empty]
        if hulls:
            geom = unary_union(hulls).buffer(0)

    return geom


def build_top_insertion_cut_geometry(
    cuts: List[Polygon],
    clip_rect: Polygon,
    step: float = 0.5,
    support_step: float | None = None,
    section_points: np.ndarray | None = None,
    side_margin: float | None = None,
) -> Tuple[BaseGeometry, List[Tuple[float, float]]]:
    """
    Build VDIV cut geometry for top-down insertion.

    For each depth column (Y), the cut opens from the local highest intersection
    up to the panel top, so the part can be lowered from above and then rests
    on the divider profile.
    In panel coords for VDIVs: X=depth(Y), Y=height(Z).
    """
    minx, miny, maxx, maxy = clip_rect.bounds
    margin = top_insert_side_margin if side_margin is None else float(side_margin)
    minx_eff = minx + max(0.0, margin)
    maxx_eff = maxx - max(0.0, margin)
    if maxx_eff <= minx_eff:
        minx_eff, maxx_eff = minx, maxx
    dx = max(float(step), 0.1)
    support_dx = dx if support_step is None else max(float(support_step), 0.1)

    strips: List[BaseGeometry] = []
    top_profile: List[Tuple[float, float]] = []

    def _build_strips_from_samples(xs: np.ndarray, ys: np.ndarray) -> None:
        nonlocal strips, top_profile
        for i in range(len(xs)):
            x = float(xs[i])
            x1 = min(x + dx, maxx_eff)
            if x1 <= x:
                continue
            sy1 = float(ys[i])
            strips.append(box(x, sy1, x1, maxy))
            top_profile.append(((x + x1) * 0.5, sy1))

    def _build_strips_from_profile(profile: Sequence[Tuple[float, float]]) -> None:
        """Build cut strips from a monotonic profile polyline."""
        nonlocal strips, top_profile
        if len(profile) < 2:
            return
        for i in range(len(profile) - 1):
            x0, y0 = profile[i]
            x1, _y1 = profile[i + 1]
            x0 = float(x0)
            x1 = float(x1)
            if x1 <= x0:
                continue
            strips.append(box(x0, float(y0), x1, maxy))
        top_profile.extend((float(x), float(y)) for x, y in profile)

    def _smooth_profile(values: np.ndarray) -> np.ndarray:
        """Remove short upward and downward spikes from the profile."""
        if len(values) < 3:
            return values

        win = int(top_insert_smooth_window)
        if win < 3:
            return values
        if win % 2 == 0:
            win += 1
        win = min(win, len(values) if len(values) % 2 == 1 else len(values) - 1)
        if win < 3:
            return values

        pad = win // 2
        padded = np.pad(values, (pad, pad), mode='edge')
        med = np.array([np.median(padded[i:i + win]) for i in range(len(values))], dtype=float)
        kernel = np.ones(win, dtype=float) / float(win)
        med_padded = np.pad(med, (pad, pad), mode='edge')
        avg = np.convolve(med_padded, kernel, mode='valid')
        return avg

    # Preferred path: derive first-hit-from-top directly from raw section points.
    if section_points is not None and len(section_points) > 0:
        pts = np.asarray(section_points, dtype=float)
        in_clip = (
            (pts[:, 0] >= minx_eff) & (pts[:, 0] <= maxx_eff) &
            (pts[:, 1] >= miny) & (pts[:, 1] <= maxy)
        )
        pts = pts[in_clip]
        if len(pts) >= 2:
            x_start = float(np.min(pts[:, 0]))
            x_stop = float(np.max(pts[:, 0]))

            if x_stop > x_start:
                support: List[Tuple[float, float]] = []
                x = x_start
                while x < x_stop:
                    x1 = min(x + support_dx, x_stop)
                    in_bin = (pts[:, 0] >= x) & (pts[:, 0] <= x1)
                    if np.any(in_bin):
                        y_mean = float(np.mean(pts[in_bin, 1]))
                        support.append(((x + x1) * 0.5, y_mean))
                    x = x1

                if len(support) >= 2:
                    support_arr = np.asarray(support, dtype=float)
                    sy = _smooth_profile(support_arr[:, 1])
                    support_smoothed = [(float(support_arr[i, 0]), float(sy[i])) for i in range(len(support_arr))]
                    spline_profile = catmull_rom_spline_points(
                        support_smoothed,
                        samples_per_segment=8,
                        edge_linear_segments=1,
                    )

                    # Keep only the actual support span so the profile ends with the blue-point extent.
                    spline_profile = [
                        (float(px), float(py))
                        for px, py in spline_profile
                        if x_start <= float(px) <= x_stop
                    ]
                    if len(spline_profile) >= 2:
                        _build_strips_from_profile(spline_profile)

    # Fallback: derive from polygonized section geometry.
    if not strips:
        clipped: List[BaseGeometry] = []
        for c in cuts:
            g = c.intersection(clip_rect)
            if not g.is_empty:
                clipped.append(g)

        if not clipped:
            return Polygon(), []

        base = unary_union(clipped).buffer(0)
        if base.is_empty:
            return Polygon(), []

        xs = np.arange(minx_eff, maxx_eff, dx, dtype=float)
        ys = np.full(xs.shape, np.nan, dtype=float)
        for i, x in enumerate(xs):
            x1 = min(x + dx, maxx_eff)
            slab = base.intersection(box(x, miny, x1, maxy))
            if slab.is_empty:
                continue
            ys[i] = float(slab.bounds[3])

        valid = np.isfinite(ys)
        if np.any(valid):
            valid_idx = np.where(valid)[0]
            i0 = int(valid_idx[0])
            i1 = int(valid_idx[-1])
            ys_seg = ys[i0:i1 + 1]
            seg_valid = np.isfinite(ys_seg)
            seg_idx = np.arange(len(ys_seg), dtype=float)
            ys_interp = np.interp(seg_idx, seg_idx[seg_valid], ys_seg[seg_valid])
            ys_smooth = _smooth_profile(ys_interp)
            _build_strips_from_samples(xs[i0:i1 + 1], ys_smooth)

    if not strips:
        return Polygon(), top_profile

    geom = unary_union(strips).buffer(0)
    if simplify_tol > 0:
        geom = geom.simplify(min(simplify_tol, dx), preserve_topology=True)
    return geom, top_profile


# -----------------------------------------------------------------------------
# Auto-fit box computation
# -----------------------------------------------------------------------------
raw_mesh = load_mesh_raw()
mesh_precenter = transformed_mesh(raw_mesh, stl_pos)
mesh_min, mesh_max = mesh_precenter.bounds
mesh_size = mesh_max - mesh_min
mesh_center = (mesh_min + mesh_max) / 2.0

inner_w = float(mesh_size[0] + 2 * pad_x)
inner_d = float(mesh_size[1] + 2 * pad_y)
inner_h = float(mesh_size[2] + 2 * pad_z)

outer_w = inner_w + 2 * mat
outer_d = inner_d + 2 * mat
outer_h = inner_h + 2 * mat

side_d = outer_d - mat
side_wall_w = side_d + mat
cell_w = inner_w / cells_x
cell_h = inner_h / cells_y
num_vdiv_x = max(cells_x - 1, 0)
num_vdiv_y = max(cells_y - 1, 0)
# Kept for compatibility with older visualization code paths.
num_hdiv = 0
slot_w = mat - kerf
cross_slot_depth = inner_h / 2
# Positive value makes cross-slots wider (easier assembly), negative tighter fit.
assembly_fit_clearance = 0.1
# Vertical overlap between X- and Y-slot ends in mm.
# > 0.0 => overlap, < 0.0 => gap (distance) between slot ends.
cross_slot_overlap_mm = 0.5
# Y offset for X-divider cross-slot centers (negative moves towards back).
x_div_cross_slot_y_offset = 0.0
# X offset for Y-divider cross-slot centers (negative moves towards left).
y_div_cross_slot_x_offset = 0.0

tab_len = max(10.0, min(20.0, inner_d / 4))
tab_t = mat + kerf
panel_tab_clearance = 0.0
panel_slot_w = tab_t + panel_tab_clearance
vdiv_panel_slot_x_offset = 0.0
tab_pos1 = inner_d * 0.25 - tab_len / 2
tab_pos2 = inner_d * 0.75 - tab_len / 2

# Side tenons for divider retention in box side walls.
# Tab depth should match wall thickness; kerf belongs to slot width, not protrusion depth.
divider_side_tab_depth = mat
divider_side_tab_height = max(10.0, min(20.0, inner_h / 5.0))
divider_side_tab_margin = max(6.0, inner_h * 0.12)

# Shared outer panel edge modes: (w, h, top_mode, bottom_mode, left_mode, right_mode)
OUTER_PANEL_SPECS = {
    "back":   (inner_w, outer_h, "flat", "female", "male", "male"),
    "front":  (inner_w, outer_h, "flat", "female", "male", "male"),
    "top":    (inner_w, inner_d, "flat", "flat", "flat", "flat"),
    "bottom": (outer_w, outer_d - mat, "male", "flat", "flat", "flat"),
    "left":   (side_wall_w, outer_h, "flat", "flat", "female", "female"),
    "right":  (side_wall_w, outer_h, "flat", "flat", "female", "female"),
}

# New STL position: center mesh in the box interior.
new_stl_pos = [
    stl_pos[0] + (inner_w / 2.0 - mesh_center[0]),
    stl_pos[1] + (inner_d / 2.0 - mesh_center[1]),
    stl_pos[2] + (inner_h / 2.0 - mesh_center[2]),
]

mesh = transformed_mesh(raw_mesh, new_stl_pos)
mesh_final_min, mesh_final_max = mesh.bounds


# -----------------------------------------------------------------------------
# Divider base polygons with slots and tabs
# -----------------------------------------------------------------------------
def vertical_divider_base() -> Polygon:
    return box(0, 0, inner_d, inner_h)


def y_vertical_divider_base() -> Polygon:
    return box(0, 0, inner_w, inner_h)


def side_tab_positions(height: float) -> List[float]:
    """Return y-start positions for two side tabs on divider edges."""
    h = max(0.1, float(divider_side_tab_height))
    m = max(0.0, float(divider_side_tab_margin))
    if height <= h:
        return [0.0]
    p1 = min(max(0.0, m), max(0.0, height - h))
    p2 = max(0.0, min(height - h, height - m - h))
    if abs(p2 - p1) < 1.0:
        return [max(0.0, (height - h) * 0.5)]
    return [p1, p2]


def add_side_tabs_to_div_x(poly: BaseGeometry) -> BaseGeometry:
    """Add left/right side tenons on X-dividers."""
    out = poly
    d = max(0.1, float(divider_side_tab_depth))
    for y0 in side_tab_positions(inner_h):
        y1 = y0 + divider_side_tab_height
        out = out.union(box(-d, y0, 0.0, y1))
        out = out.union(box(inner_d, y0, inner_d + d, y1))
    return out.buffer(0)


def add_side_tabs_to_div_y(poly: BaseGeometry) -> BaseGeometry:
    """Add left/right side tenons on Y-dividers."""
    out = poly
    d = max(0.1, float(divider_side_tab_depth))
    for y0 in side_tab_positions(inner_h):
        y1 = y0 + divider_side_tab_height
        out = out.union(box(-d, y0, 0.0, y1))
        out = out.union(box(inner_w, y0, inner_w + d, y1))
    return out.buffer(0)


def profile_height_at(profile: List[Tuple[float, float]], coord: float) -> float:
    """Interpolate remaining divider height at a given in-plane coordinate."""
    if not profile:
        return float("nan")
    arr = np.asarray(profile, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 2:
        return float("nan")
    order = np.argsort(arr[:, 0])
    xs = arr[order, 0]
    ys = arr[order, 1]
    if coord < xs[0] or coord > xs[-1]:
        return float("nan")
    return float(np.interp(coord, xs, ys))


def crossing_heights(
    x_profile: List[Tuple[float, float]],
    y_profile: List[Tuple[float, float]],
    y_coord: float,
    x_coord: float,
) -> Tuple[float, float, float]:
    """Return (x_height, y_height, common_height) at a divider crossing."""
    zx = profile_height_at(x_profile, y_coord)
    zy = profile_height_at(y_profile, x_coord)
    vals = [v for v in (zx, zy) if np.isfinite(v)]
    if not vals:
        return 0.0, 0.0, 0.0
    common = max(0.0, min(vals))
    return float(zx if np.isfinite(zx) else 0.0), float(zy if np.isfinite(zy) else 0.0), common


def add_cross_slots_to_div_x(
    poly: BaseGeometry,
    x_pos: float,
    x_profile: List[Tuple[float, float]],
    y_entries: List[Tuple[float, List[Tuple[float, float]]]],
) -> BaseGeometry:
    """Cut bottom-up slots in X-dividers at every Y-divider crossing."""
    out = poly
    if not y_entries:
        return out
    slot_w_cross = max(slot_w + assembly_fit_clearance, 0.1)
    overlap = float(cross_slot_overlap_mm)
    half_w = slot_w_cross / 2.0
    for y0, y_profile in y_entries:
        y_slot = float(y0) + float(x_div_cross_slot_y_offset)
        if y_slot + half_w <= 0.0 or y_slot - half_w >= inner_d:
            continue
        _zx, _zy, common_h = crossing_heights(x_profile, y_profile, y0, x_pos)
        split_h = max(0.0, min(common_h * 0.5, inner_h))
        x_top = max(0.0, min(split_h + 0.5 * overlap, inner_h))
        if x_top <= 0.0:
            continue
        out = out.difference(box(y_slot - half_w, 0.0, y_slot + half_w, x_top))
    return out.buffer(0)


def add_cross_slots_to_div_y(
    poly: BaseGeometry,
    y_pos: float,
    y_profile: List[Tuple[float, float]],
    x_entries: List[Tuple[float, List[Tuple[float, float]]]],
) -> BaseGeometry:
    """Cut top-down slots in Y-dividers at every X-divider crossing."""
    out = poly
    if not x_entries:
        return out
    slot_w_cross = max(slot_w + assembly_fit_clearance, 0.1)
    overlap = float(cross_slot_overlap_mm)
    half_w = slot_w_cross / 2.0
    for x0, x_profile in x_entries:
        x_slot = float(x0) + float(y_div_cross_slot_x_offset)
        if x_slot + half_w <= 0.0 or x_slot - half_w >= inner_w:
            continue
        _zx, zy, common_h = crossing_heights(x_profile, y_profile, y_pos, x0)
        split_h = max(0.0, min(common_h * 0.5, inner_h))
        # Measure the actual contour at the shifted slot center so the opening
        # reaches the profile even when the slot is offset in X.
        contour_h = profile_height_at(y_profile, x_slot)
        if not np.isfinite(contour_h):
            contour_h = zy
        top_h = max(common_h, max(0.0, min(contour_h, inner_h)))
        y_bottom = max(0.0, min(split_h - 0.5 * overlap, inner_h))
        if top_h <= y_bottom:
            continue
        # Y-divider slot must be open to the local upper contour for assembly.
        out = out.difference(box(x_slot - half_w, y_bottom, x_slot + half_w, top_h))
    return out.buffer(0)


def add_vdivider_tab_holes_to_panel(poly: BaseGeometry) -> BaseGeometry:
    """Cut top/bottom panel slots for vertical-divider tabs."""
    out = poly
    if num_vdiv_x <= 0:
        return out

    for idx in range(num_vdiv_x):
        x_center = mat + (idx + 1) * cell_w + vdiv_panel_slot_x_offset
        x0 = x_center - panel_slot_w / 2
        x1 = x_center + panel_slot_w / 2
        for y0 in (tab_pos1, tab_pos2):
            out = out.difference(box(x0, y0, x1, y0 + tab_len))

    return out.buffer(0)


def add_divider_side_tab_holes_to_panel(name: str, poly: BaseGeometry) -> BaseGeometry:
    """Cut receiving slots in side walls for divider side tenons."""
    out = poly
    half_w = max(panel_slot_w, 0.1) / 2.0
    z_starts = side_tab_positions(inner_h)

    if name in ("back", "front") and num_vdiv_x > 0:
        z_offset = mat if name in ("front", "back") else 0.0
        for idx in range(num_vdiv_x):
            x_center = (idx + 1) * cell_w
            for z0 in z_starts:
                y0 = z0 + z_offset
                y1 = y0 + divider_side_tab_height
                out = out.difference(box(x_center - half_w, y0, x_center + half_w, y1))

    if name in ("left", "right") and num_vdiv_y > 0:
        for idx in range(num_vdiv_y):
            y_center = (idx + 1) * (inner_d / cells_y) + mat
            for z0 in z_starts:
                yy0 = z0 + mat
                yy1 = yy0 + divider_side_tab_height
                out = out.difference(box(y_center - half_w, yy0, y_center + half_w, yy1))

    return out.buffer(0)


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------
def generate_outer_panels(panel_names: Sequence[str] | None = None) -> None:
    panels = OUTER_PANEL_SPECS
    selected = set(panel_names) if panel_names is not None else None

    for name, (w, h, top_mode, bot_mode, left_mode, right_mode) in panels.items():
        if selected is not None and name not in selected:
            continue
        panel_w = float(w)
        panel_h = float(h)

        if name == "bottom":
            # Compensate male tab protrusion so final bottom contour stays on target dimensions.
            tab_depth = mat + kerf
            panel_w = max(0.1, panel_w - 2.0 * tab_depth)
            panel_h = max(0.1, panel_h - (1.0 * tab_depth + kerf))

        poly = panel_outline(panel_w, panel_h, top_mode, bot_mode, left_mode, right_mode, panel_name=name)
        poly = add_divider_side_tab_holes_to_panel(name, poly)
        minx, miny, maxx, maxy = poly.bounds
        shifted = shift_geometry(poly, -minx, -miny)
        save_svg(maxx - minx, maxy - miny, output_dir / f"{name}.svg", shifted)


def generate_dividers() -> None:
    vbase = vertical_divider_base()
    ybase = y_vertical_divider_base()

    def div_x_name(x_pos_mm: float) -> str:
        token = f"{x_pos_mm:.2f}".replace('-', 'm').replace('.', 'p')
        return f"div_x_{token}"

    def div_y_name(y_pos_mm: float) -> str:
        token = f"{y_pos_mm:.2f}".replace('-', 'm').replace('.', 'p')
        return f"div_y_{token}"

    x_data: List[Tuple[float, BaseGeometry, List[Tuple[float, float]], np.ndarray]] = []
    y_data: List[Tuple[float, BaseGeometry, List[Tuple[float, float]], np.ndarray]] = []

    # Build X-divider base geometry + top-insert profile.
    for idx in range(num_vdiv_x):
        x0 = (idx + 1) * cell_w
        poly = vbase

        cuts = sampled_section_polygons(
            mesh,
            np.array([x0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            "yz",
            thickness=mat,
            samples=divider_section_samples,
        )
        cut_points = sampled_section_points(
            mesh,
            np.array([x0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            "yz",
            thickness=mat,
            samples=divider_section_samples,
        )
        clip_rect = box(0, 0, inner_d, inner_h)
        cut_geom, top_profile = build_top_insertion_cut_geometry(
            cuts,
            clip_rect,
            step=top_insert_step,
            support_step=top_insert_support_step,
            section_points=cut_points,
        )
        if not cut_geom.is_empty:
            poly = poly.difference(cut_geom)
        x_data.append((x0, poly, top_profile, cut_points))

        minx, miny, maxx, maxy = poly.bounds
        shifted = shift_geometry(poly, -minx, -miny)
        out_name = div_x_name(x0)
        save_svg(maxx - minx, maxy - miny, output_dir / f"{out_name}.svg", shifted)

    # Build Y-divider base geometry + top-insert profile.
    for idx in range(num_vdiv_y):
        y0 = (idx + 1) * (inner_d / cells_y)
        poly = ybase

        cuts = sampled_section_polygons(
            mesh,
            np.array([0.0, y0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            "xz",
            thickness=mat,
            samples=divider_section_samples,
        )
        cut_points = sampled_section_points(
            mesh,
            np.array([0.0, y0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            "xz",
            thickness=mat,
            samples=divider_section_samples,
        )
        clip_rect = box(0, 0, inner_w, inner_h)
        cut_geom, top_profile = build_top_insertion_cut_geometry(
            cuts,
            clip_rect,
            step=top_insert_step,
            support_step=top_insert_support_step,
            section_points=cut_points,
        )
        if not cut_geom.is_empty:
            poly = poly.difference(cut_geom)
        y_data.append((y0, poly, top_profile, cut_points))

    x_profiles = [(x0, prof) for x0, _poly, prof, _pts in x_data]
    y_profiles = [(y0, prof) for y0, _poly, prof, _pts in y_data]

    # Export X-dividers with dynamic cross-slots from bottom.
    for x0, poly, top_profile, cut_points in x_data:
        poly = add_cross_slots_to_div_x(poly, x0, top_profile, y_profiles)
        poly = add_side_tabs_to_div_x(poly)

        minx, miny, maxx, maxy = poly.bounds
        shifted = shift_geometry(poly, -minx, -miny)
        out_name = div_x_name(x0)
        save_svg(maxx - minx, maxy - miny, output_dir / f"{out_name}.svg", shifted)

        if debug_top_contact_svg and top_profile:
            profile_shifted = [(px - minx, py - miny) for px, py in top_profile]
            pts_shifted = np.asarray(cut_points, dtype=float).copy()
            if pts_shifted.size > 0:
                pts_shifted[:, 0] -= minx
                pts_shifted[:, 1] -= miny
            save_vdiv_debug_svg(
                maxx - minx,
                maxy - miny,
                output_dir / f"{out_name}_debug_top_contact.svg",
                shifted,
                pts_shifted,
                profile_shifted,
            )

    # Export Y-dividers with dynamic cross-slots from top.
    for y0, poly, top_profile, cut_points in y_data:
        poly = add_cross_slots_to_div_y(poly, y0, top_profile, x_profiles)
        poly = add_side_tabs_to_div_y(poly)

        minx, miny, maxx, maxy = poly.bounds
        shifted = shift_geometry(poly, -minx, -miny)
        out_name = div_y_name(y0)
        save_svg(maxx - minx, maxy - miny, output_dir / f"{out_name}.svg", shifted)

        if debug_top_contact_svg and top_profile:
            profile_shifted = [(px - minx, py - miny) for px, py in top_profile]
            pts_shifted = np.asarray(cut_points, dtype=float).copy()
            if pts_shifted.size > 0:
                pts_shifted[:, 0] -= minx
                pts_shifted[:, 1] -= miny
            save_vdiv_debug_svg(
                maxx - minx,
                maxy - miny,
                output_dir / f"{out_name}_debug_top_contact.svg",
                shifted,
                pts_shifted,
                profile_shifted,
            )



def build_all_parts() -> List[Tuple[str, BaseGeometry]]:
    parts: List[Tuple[str, BaseGeometry]] = []

    def div_x_name(x_pos_mm: float) -> str:
        token = f"{x_pos_mm:.2f}".replace('-', 'm').replace('.', 'p')
        return f"div_x_{token}"

    def div_y_name(y_pos_mm: float) -> str:
        token = f"{y_pos_mm:.2f}".replace('-', 'm').replace('.', 'p')
        return f"div_y_{token}"

    vbase = vertical_divider_base()
    ybase = y_vertical_divider_base()
    x_data: List[Tuple[float, BaseGeometry, List[Tuple[float, float]]]] = []
    y_data: List[Tuple[float, BaseGeometry, List[Tuple[float, float]]]] = []
    for idx in range(num_vdiv_x):
        x0 = (idx + 1) * cell_w
        poly = vbase
        cuts = sampled_section_polygons(
            mesh,
            np.array([x0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            "yz",
            thickness=mat,
            samples=divider_section_samples,
        )
        cut_points = sampled_section_points(
            mesh,
            np.array([x0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            "yz",
            thickness=mat,
            samples=divider_section_samples,
        )
        cut_geom, _top_profile = build_top_insertion_cut_geometry(
            cuts,
            box(0, 0, inner_d, inner_h),
            step=top_insert_step,
            support_step=top_insert_support_step,
            section_points=cut_points,
        )
        if not cut_geom.is_empty:
            poly = poly.difference(cut_geom)
        x_data.append((x0, poly, _top_profile))

    for idx in range(num_vdiv_y):
        y0 = (idx + 1) * (inner_d / cells_y)
        poly = ybase
        cuts = sampled_section_polygons(
            mesh,
            np.array([0.0, y0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            "xz",
            thickness=mat,
            samples=divider_section_samples,
        )
        cut_points = sampled_section_points(
            mesh,
            np.array([0.0, y0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            "xz",
            thickness=mat,
            samples=divider_section_samples,
        )
        cut_geom, _top_profile = build_top_insertion_cut_geometry(
            cuts,
            box(0, 0, inner_w, inner_h),
            step=top_insert_step,
            support_step=top_insert_support_step,
            section_points=cut_points,
        )
        if not cut_geom.is_empty:
            poly = poly.difference(cut_geom)
        y_data.append((y0, poly, _top_profile))

    x_profiles = [(x0, prof) for x0, _poly, prof in x_data]
    y_profiles = [(y0, prof) for y0, _poly, prof in y_data]

    for x0, poly, top_profile in x_data:
        poly = add_cross_slots_to_div_x(poly, x0, top_profile, y_profiles)
        poly = add_side_tabs_to_div_x(poly)
        minx, miny, _, _ = poly.bounds
        parts.append((div_x_name(x0), shift_geometry(poly, -minx, -miny)))

    for y0, poly, top_profile in y_data:
        poly = add_cross_slots_to_div_y(poly, y0, top_profile, x_profiles)
        poly = add_side_tabs_to_div_y(poly)
        minx, miny, _, _ = poly.bounds
        parts.append((div_y_name(y0), shift_geometry(poly, -minx, -miny)))

    return parts


def generate_layout_svg() -> None:
    parts = build_all_parts()

    placed: List[Tuple[str, BaseGeometry, float, float, float, float]] = []
    cursor_x = layout_spacing
    cursor_y = layout_spacing
    row_height = 0.0

    for name, poly in parts:
        minx, miny, maxx, maxy = poly.bounds
        w = maxx - minx
        h = maxy - miny

        if cursor_x + w + layout_spacing > layout_max_row_width and cursor_x > layout_spacing:
            cursor_x = layout_spacing
            cursor_y += row_height + layout_spacing
            row_height = 0.0

        placed.append((name, poly, cursor_x, cursor_y, w, h))
        cursor_x += w + layout_spacing
        row_height = max(row_height, h)

    total_width = max((x + w + layout_spacing for _, _, x, _, w, _ in placed), default=layout_spacing * 2)
    total_height = max((y + h + layout_spacing for _, _, _, y, _, h in placed), default=layout_spacing * 2)

    dwg = svgwrite.Drawing(
        str(output_dir / "layout.svg"),
        size=(f"{total_width}mm", f"{total_height}mm"),
        viewBox=f"0 0 {total_width} {total_height}",
    )
    grp = dwg.g(fill="none", stroke="black", stroke_width=svg_stroke_width)

    for name, poly, dx, dy, _, _ in placed:
        for part in iter_polygons(poly):
            ext = transform_points(part.exterior.coords, dx=dx, dy=dy, flip_y_height=total_height)
            grp.add(dwg.polygon(points=ext))

            for ring in part.interiors:
                hole = transform_points(ring.coords, dx=dx, dy=dy, flip_y_height=total_height)
                grp.add(dwg.polygon(points=hole))

        label_x = dx + 2
        label_y = total_height - dy - 4
        grp.add(dwg.text(name, insert=(label_x, label_y), font_size="4px", fill="black"))

    dwg.add(grp)
    dwg.save()


def cleanup_output_files() -> None:
    """Remove stale generated files that may no longer match current settings."""
    if not output_dir.exists():
        return

    patterns = [
        "layout.svg",
        "back.svg",
        "front.svg",
        "top.svg",
        "bottom.svg",
        "left.svg",
        "right.svg",
        "div_*.svg",
        "vdiv_*.svg",
        "ydiv_*.svg",
        "hdiv_*.svg",
        "*_debug_*.svg",
    ]
    for pattern in patterns:
        for file_path in output_dir.glob(pattern):
            try:
                file_path.unlink()
            except OSError:
                pass


def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_output_files()

    print(f"Loaded {stl_file}")
    print(f"Raw STL watertight: {raw_mesh.is_watertight}")
    print(f"Mesh bounds after current transform: min={mesh_min}, max={mesh_max}")
    print(f"Mesh size: {mesh_size[0]:.2f} x {mesh_size[1]:.2f} x {mesh_size[2]:.2f} mm")
    print(f"Final mesh bounds (used for cuts): min={mesh_final_min}, max={mesh_final_max}")
    print()
    print("Auto-fit box:")
    print(f"  inner_w = {inner_w:.2f}")
    print(f"  inner_d = {inner_d:.2f}")
    print(f"  inner_h = {inner_h:.2f}")
    print(f"  outer_w = {outer_w:.2f}")
    print(f"  outer_d = {outer_d:.2f}")
    print(f"  outer_h = {outer_h:.2f}")
    print()
    print("Centered STL transform for the box interior:")
    print(f"  stl_scale = {stl_scale}")
    print(f"  stl_rot   = {stl_rot_deg}")
    print(f"  stl_pos   = [{new_stl_pos[0]:.3f}, {new_stl_pos[1]:.3f}, {new_stl_pos[2]:.3f}]")

    generate_outer_panels(panel_names=["top", "bottom", "front", "back", "left", "right"])
    generate_dividers()

    print()
    print("Launching interactive visualization...")
    subprocess.run([sys.executable, "visualize_interactive.py"], check=True)

    print()
    print(f"Wrote SVG files to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()