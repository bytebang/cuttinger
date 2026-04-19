#!/usr/bin/env python3
"""Interactive 3D visualization using Plotly."""

import warnings
warnings.filterwarnings('ignore')

import sys
import re
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import plotly.graph_objects as go
import runpy

print("Loading cuttinger data...", file=sys.stderr)
ns = runpy.run_path('cuttinger.py')

mesh = ns['mesh']
outer_w = ns['outer_w']
outer_d = ns['outer_d']
outer_h = ns['outer_h']
inner_w = ns['inner_w']
inner_d = ns['inner_d']
inner_h = ns['inner_h']
cell_w = ns['cell_w']
cell_h = ns['cell_h']
num_vdiv_x = ns['num_vdiv_x']
num_hdiv = ns['num_hdiv']
num_vdiv_y = ns.get('num_vdiv_y', 0)
mesh_size = np.asarray(ns.get('mesh_size', [inner_w, inner_d, inner_h]), dtype=float)
outer_panel_specs = ns.get('OUTER_PANEL_SPECS', {})
# Geometry exported from cuttinger.py is expressed in inner-box coordinates.
# Shift by inner-box center so mesh and divider geometry aligns with the drawn inner box.
box_center = np.array([inner_w / 2.0, inner_d / 2.0, inner_h / 2.0], dtype=float)

print(f"Creating interactive visualization...", file=sys.stderr)

SVG_DIR = Path('box_output')
SVG_EXCLUDE = {'layout.svg'}

fig = go.Figure()

# Define box edges
edges = [
    [0,1], [1,2], [2,3], [3,0],  # bottom
    [4,5], [5,6], [6,7], [7,4],  # top
    [0,4], [1,5], [2,6], [3,7]   # vertical
]

def add_box_edges(fig, w, d, h, color, name, dash='solid', width=2, opacity=0.8, centered=False):
    """Add box edges to figure."""
    added_indices = []
    if centered:
        box_pts = np.array([
            [-w / 2.0, -d / 2.0, -h / 2.0], [w / 2.0, -d / 2.0, -h / 2.0], [w / 2.0, d / 2.0, -h / 2.0], [-w / 2.0, d / 2.0, -h / 2.0],
            [-w / 2.0, -d / 2.0, h / 2.0], [w / 2.0, -d / 2.0, h / 2.0], [w / 2.0, d / 2.0, h / 2.0], [-w / 2.0, d / 2.0, h / 2.0]
        ])
    else:
        box_pts = np.array([
            [0,0,0], [w,0,0], [w,d,0], [0,d,0],
            [0,0,h], [w,0,h], [w,d,h], [0,d,h]
        ])
    
    for e in edges:
        pts = box_pts[e]
        added_indices.append(len(fig.data))
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='lines',
            line=dict(color=color, width=width, dash=dash),
            name=name,
            legendgroup=name,
            opacity=opacity,
            hoverinfo='skip',
            showlegend=(e == edges[0])  # Only show in legend once
        ))
    return added_indices


def parse_svg_points(points_text):
    """Parse SVG polygon points string into an (N,2) float array."""
    vals = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', points_text)
    nums = np.asarray([float(v) for v in vals], dtype=float)
    if nums.size < 6:
        return None
    if nums.size % 2 != 0:
        nums = nums[:-1]
    return nums.reshape(-1, 2)


def load_svg_polygons(svg_path):
    """Load all polygon rings from an SVG file as 2D numpy arrays."""
    polygons = []
    try:
        root = ET.parse(svg_path).getroot()
    except Exception as exc:
        print(f"Warning: failed to parse {svg_path}: {exc}", file=sys.stderr)
        return polygons

    for poly in root.findall('.//{*}polygon'):
        pts = parse_svg_points(poly.attrib.get('points', ''))
        if pts is None or len(pts) < 3:
            continue
        polygons.append(pts)
    return polygons


def add_svg_plate(fig, polygons, map_front, map_back, color, name, legendgroup=None, show_legend_entry=True):
    """Render 2D SVG polygons as a thin 3D plate with front/back rings."""
    shown = not show_legend_entry
    group = legendgroup if legendgroup is not None else name

    for poly in polygons:
        if len(poly) < 3:
            continue
        pts = poly
        if np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        if len(pts) < 3:
            continue

        pts_closed = np.vstack([pts, pts[0]])

        x_front, y_front, z_front = map_front(pts)
        x_back, y_back, z_back = map_back(pts)
        x_front_c, y_front_c, z_front_c = map_front(pts_closed)
        x_back_c, y_back_c, z_back_c = map_back(pts_closed)

        fig.add_trace(go.Scatter3d(
            x=x_front_c, y=y_front_c, z=z_front_c,
            mode='lines',
            line=dict(color=color, width=2),
            opacity=0.9,
            name=name,
            legendgroup=group,
            showlegend=not shown,
            hoverinfo='skip'
        ))
        shown = True

        fig.add_trace(go.Scatter3d(
            x=x_back_c, y=y_back_c, z=z_back_c,
            mode='lines',
            line=dict(color=color, width=2),
            opacity=0.9,
            legendgroup=group,
            showlegend=False,
            hoverinfo='skip'
        ))

        # Connect front/back rings without surface shading.
        for idx in range(len(pts_closed)):
            fig.add_trace(go.Scatter3d(
            x=[x_front_c[idx], x_back_c[idx]],
            y=[y_front_c[idx], y_back_c[idx]],
            z=[z_front_c[idx], z_back_c[idx]],
                mode='lines',
            line=dict(color=color, width=0.8),
                opacity=0.65,
                legendgroup=group,
                showlegend=False,
                hoverinfo='skip'
            ))

# Draw outer box
add_box_edges(fig, outer_w, outer_d, outer_h, 'black', 'Outer Box', width=3, centered=True)

# Draw inner box
add_box_edges(fig, inner_w, inner_d, inner_h, 'blue', 'Inner Box', dash='dash', width=2, opacity=0.6, centered=True)

# Draw original object bounding box before padding.
mesh_w, mesh_d, mesh_h = (float(mesh_size[0]), float(mesh_size[1]), float(mesh_size[2]))
add_box_edges(fig, mesh_w, mesh_d, mesh_h, 'seagreen', 'Original BBox (No Padding)', dash='dot', width=2, opacity=0.9, centered=True)

# Draw vertical dividers (blue)
print("Adding vertical dividers...", file=sys.stderr)
for idx in range(num_vdiv_x):
    x_pos = (idx + 1) * cell_w
    x_pos = x_pos - box_center[0]
    legend_name = f'V-Divider {idx + 1}'
    # 4 vertical lines
    for y_raw in [0, inner_d]:
        y = y_raw - box_center[1]
        fig.add_trace(go.Scatter3d(
            x=[x_pos, x_pos], y=[y, y], z=[-box_center[2], inner_h - box_center[2]],
            mode='lines',
            line=dict(color='blue', width=1.5),
            name=legend_name if y_raw == 0 else '',
            legendgroup=legend_name,
            opacity=0.7,
            hoverinfo='skip',
            showlegend=bool(y_raw == 0)
        ))
    # Connect edges
    fig.add_trace(go.Scatter3d(
        x=[x_pos, x_pos], y=[-box_center[1], inner_d - box_center[1]], z=[-box_center[2], -box_center[2]],
        mode='lines',
        line=dict(color='blue', width=0.5),
        legendgroup=legend_name,
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter3d(
        x=[x_pos, x_pos], y=[-box_center[1], inner_d - box_center[1]], z=[inner_h - box_center[2], inner_h - box_center[2]],
        mode='lines',
        line=dict(color='blue', width=0.5),
        legendgroup=legend_name,
        showlegend=False,
        hoverinfo='skip'
    ))

# Draw Y-dividers (orange): vertical plates at y = const (between left and right).
print("Adding Y-dividers...", file=sys.stderr)
for idx in range(num_vdiv_y):
    y_pos_raw = (idx + 1) * (inner_d / ns['cells_y'])
    y_pos = y_pos_raw - box_center[1]
    legend_name = f'Y-Divider {idx + 1}'
    for x_pt_raw in [0, inner_w]:
        x_pt = x_pt_raw - box_center[0]
        fig.add_trace(go.Scatter3d(
            x=[x_pt, x_pt], y=[y_pos, y_pos], z=[-box_center[2], inner_h - box_center[2]],
            mode='lines',
            line=dict(color='darkorange', width=1.5),
            name=legend_name if x_pt_raw == 0 else '',
            legendgroup=legend_name,
            opacity=0.7,
            hoverinfo='skip',
            showlegend=bool(x_pt_raw == 0)
        ))
    for z_pt in [0, inner_h]:
        fig.add_trace(go.Scatter3d(
            x=[-box_center[0], inner_w - box_center[0]], y=[y_pos, y_pos], z=[z_pt - box_center[2], z_pt - box_center[2]],
            mode='lines',
            line=dict(color='darkorange', width=0.7),
            legendgroup=legend_name,
            showlegend=False,
            hoverinfo='skip'
        ))

# Draw mesh as scatter
print("Adding mesh...", file=sys.stderr)
if mesh.vertices is not None and len(mesh.vertices) > 0:
    verts = mesh.vertices
    # Subsample for performance
    step = max(1, len(verts) // 5000)
    verts_sub = verts[::step] - box_center
    
    fig.add_trace(go.Scatter3d(
        x=verts_sub[:, 0], y=verts_sub[:, 1], z=verts_sub[:, 2],
        mode='markers',
        marker=dict(size=2, color='coral', opacity=0.7),
        name='Mesh',
        hoverinfo='skip'
    ))

# Draw generated SVGs as 3D plates at their assembly positions.
print("Adding SVG plate models...", file=sys.stderr)
svg_files = sorted(
    p for p in SVG_DIR.glob('*.svg')
    if p.name.lower() not in SVG_EXCLUDE and '_debug_' not in p.stem.lower()
)

plate_thickness = float(ns.get('mat', 1.0))
if plate_thickness <= 0.0:
    plate_thickness = 1.0
tab_t = float(ns.get('tab_t', plate_thickness + ns.get('kerf', 0.0)))
divider_side_tab_depth = float(ns.get('divider_side_tab_depth', 0.0))

type_colors = {
    'vdiv': '#0B6E4F',
    'ydiv': '#B02E0C',
    'hdiv': '#B02E0C',
    'panel': '#355070',
    'other': '#7C6A0A',
}

for svg_path in svg_files:
    polygons = load_svg_polygons(svg_path)
    if not polygons:
        continue

    all_pts = np.vstack(polygons)
    min_x, min_y = all_pts.min(axis=0)
    max_x, max_y = all_pts.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y
    if width <= 0.0 or height <= 0.0:
        continue

    stem = svg_path.stem.lower()
    m_div_x = re.match(r'^div_x_([mp\d]+(?:p\d+)?)$', stem)
    m_div_y = re.match(r'^div_y_([mp\d]+(?:p\d+)?)$', stem)
    m_hdiv = re.match(r'^hdiv_(\d+)$', stem)

    # Reconstruct panel-local coordinates before min-bounds shift in SVG export.
    local_min_x = 0.0
    local_min_y = 0.0
    if stem in outer_panel_specs:
        _w, _h, _top_mode, bottom_mode, left_mode, _right_mode = outer_panel_specs[stem]
        if left_mode == 'male':
            local_min_x = -tab_t
        if bottom_mode == 'male':
            local_min_y = -tab_t

    if m_div_x or m_div_y:
        # Divider SVGs are shifted to min bounds on export; restore the original
        # negative tab extent before mapping into 3D assembly coordinates.
        local_min_x = -divider_side_tab_depth

    # SVG export flips Y for cutter coordinates; flip back to world-up for Z/Y.
    normalized = []
    for poly in polygons:
        p = poly - np.array([min_x, min_y])
        p[:, 1] = height - p[:, 1]
        p[:, 0] += local_min_x
        p[:, 1] += local_min_y
        normalized.append(p)

    if m_div_x:
        x_token = m_div_x.group(1)
        try:
            x_center = float(x_token.replace('m', '-').replace('p', '.'))
        except ValueError:
            continue

        def map_front(pts):
            return (
                np.full(pts.shape[0], x_center - plate_thickness / 2.0 - box_center[0]),
                pts[:, 0] - box_center[1],
                pts[:, 1] - box_center[2],
            )

        def map_back(pts):
            return (
                np.full(pts.shape[0], x_center + plate_thickness / 2.0 - box_center[0]),
                pts[:, 0] - box_center[1],
                pts[:, 1] - box_center[2],
            )

        add_svg_plate(
            fig,
            normalized,
            map_front,
            map_back,
            type_colors['vdiv'],
            f"SVG: {svg_path.stem}",
        )

    elif m_div_y:
        y_token = m_div_y.group(1)
        try:
            y_center = float(y_token.replace('m', '-').replace('p', '.'))
        except ValueError:
            continue

        def map_front(pts):
            return (
                pts[:, 0] - box_center[0],
                np.full(pts.shape[0], y_center - plate_thickness / 2.0 - box_center[1]),
                pts[:, 1] - box_center[2],
            )

        def map_back(pts):
            return (
                pts[:, 0] - box_center[0],
                np.full(pts.shape[0], y_center + plate_thickness / 2.0 - box_center[1]),
                pts[:, 1] - box_center[2],
            )

        add_svg_plate(
            fig,
            normalized,
            map_front,
            map_back,
            type_colors['ydiv'],
            f"SVG: {svg_path.stem}",
        )

    elif m_hdiv:
        idx = int(m_hdiv.group(1))
        if idx >= num_hdiv:
            continue
        z_center = (idx + 1) * cell_h

        def map_front(pts):
            return (
                pts[:, 0] - box_center[0],
                pts[:, 1] - box_center[1],
                np.full(pts.shape[0], z_center - plate_thickness / 2.0 - box_center[2]),
            )

        def map_back(pts):
            return (
                pts[:, 0] - box_center[0],
                pts[:, 1] - box_center[1],
                np.full(pts.shape[0], z_center + plate_thickness / 2.0 - box_center[2]),
            )

        add_svg_plate(fig, normalized, map_front, map_back, type_colors['hdiv'], f"SVG: {svg_path.stem}")

    elif stem == 'back':
        y_center = 0.0
        y_shift = -0.5 * plate_thickness
        z_shift = -plate_thickness

        def map_front(pts):
            return (
                pts[:, 0] - box_center[0],
                np.full(pts.shape[0], y_center - plate_thickness / 2.0 + y_shift - box_center[1]),
                pts[:, 1] - box_center[2] + z_shift,
            )

        def map_back(pts):
            return (
                pts[:, 0] - box_center[0],
                np.full(pts.shape[0], y_center + plate_thickness / 2.0 + y_shift - box_center[1]),
                pts[:, 1] - box_center[2] + z_shift,
            )

        add_svg_plate(fig, normalized, map_front, map_back, type_colors['panel'], f"SVG: {svg_path.stem}")

    elif stem == 'front':
        y_center = inner_d
        y_shift = 0.5 * plate_thickness
        z_shift = -plate_thickness

        def map_front(pts):
            return (
                pts[:, 0] - box_center[0],
                np.full(pts.shape[0], y_center - plate_thickness / 2.0 + y_shift - box_center[1]),
                pts[:, 1] - box_center[2] + z_shift,
            )

        def map_back(pts):
            return (
                pts[:, 0] - box_center[0],
                np.full(pts.shape[0], y_center + plate_thickness / 2.0 + y_shift - box_center[1]),
                pts[:, 1] - box_center[2] + z_shift,
            )

        add_svg_plate(fig, normalized, map_front, map_back, type_colors['panel'], f"SVG: {svg_path.stem}")

    elif stem == 'left':
        x_center = 0.0
        x_shift = -0.5 * plate_thickness
        y_shift = -plate_thickness
        z_shift = -plate_thickness

        def map_front(pts):
            return (
                np.full(pts.shape[0], x_center - plate_thickness / 2.0 + x_shift - box_center[0]),
                pts[:, 0] - box_center[1] + y_shift,
                pts[:, 1] - box_center[2] + z_shift,
            )

        def map_back(pts):
            return (
                np.full(pts.shape[0], x_center + plate_thickness / 2.0 + x_shift - box_center[0]),
                pts[:, 0] - box_center[1] + y_shift,
                pts[:, 1] - box_center[2] + z_shift,
            )

        add_svg_plate(fig, normalized, map_front, map_back, type_colors['panel'], f"SVG: {svg_path.stem}")

    elif stem == 'right':
        x_center = inner_w
        x_shift = 0.5 * plate_thickness
        y_shift = -plate_thickness
        z_shift = -plate_thickness

        def map_front(pts):
            return (
                np.full(pts.shape[0], x_center - plate_thickness / 2.0 + x_shift - box_center[0]),
                pts[:, 0] - box_center[1] + y_shift,
                pts[:, 1] - box_center[2] + z_shift,
            )

        def map_back(pts):
            return (
                np.full(pts.shape[0], x_center + plate_thickness / 2.0 + x_shift - box_center[0]),
                pts[:, 0] - box_center[1] + y_shift,
                pts[:, 1] - box_center[2] + z_shift,
            )

        add_svg_plate(fig, normalized, map_front, map_back, type_colors['panel'], f"SVG: {svg_path.stem}")

    elif stem == 'bottom':
        z_center = 0.0
        z_shift = -0.5 * plate_thickness

        def map_front(pts):
            return (
                pts[:, 0] - box_center[0],
                pts[:, 1] - box_center[1],
                np.full(pts.shape[0], z_center - plate_thickness / 2.0 + z_shift - box_center[2]),
            )

        def map_back(pts):
            return (
                pts[:, 0] - box_center[0],
                pts[:, 1] - box_center[1],
                np.full(pts.shape[0], z_center + plate_thickness / 2.0 + z_shift - box_center[2]),
            )

        add_svg_plate(fig, normalized, map_front, map_back, type_colors['panel'], f"SVG: {svg_path.stem}")

    elif stem == 'top':
        z_center = outer_h - 1.5 * plate_thickness

        def map_front(pts):
            return (
                pts[:, 0] - box_center[0],
                pts[:, 1] - box_center[1],
                np.full(pts.shape[0], z_center - plate_thickness / 2.0 - box_center[2]),
            )

        def map_back(pts):
            return (
                pts[:, 0] - box_center[0],
                pts[:, 1] - box_center[1],
                np.full(pts.shape[0], z_center + plate_thickness / 2.0 - box_center[2]),
            )

        add_svg_plate(fig, normalized, map_front, map_back, type_colors['panel'], f"SVG: {svg_path.stem}")

    else:
        # Fallback: show unknown SVGs as XZ plates near the front.
        y_center = outer_d + plate_thickness

        def map_front(pts):
            return (
                pts[:, 0] - box_center[0],
                np.full(pts.shape[0], y_center - plate_thickness / 2.0 - box_center[1]),
                pts[:, 1] - box_center[2],
            )

        def map_back(pts):
            return (
                pts[:, 0] - box_center[0],
                np.full(pts.shape[0], y_center + plate_thickness / 2.0 - box_center[1]),
                pts[:, 1] - box_center[2],
            )

        add_svg_plate(fig, normalized, map_front, map_back, type_colors['other'], f"SVG: {svg_path.stem}")

# Update layout
fig.update_layout(
    title='Setzkasten Box with Mesh - Interactive 3D View<br><sub>Rotate with mouse, scroll to zoom, drag to pan</sub>',
    legend=dict(
        x=1.0,
        xanchor='right',
        y=1.0,
        yanchor='top',
        orientation='v',
    ),
    autosize=True,
    margin=dict(l=0, r=0, t=80, b=0),
    scene=dict(
        xaxis=dict(title='X [mm]', gridcolor='lightgray', range=[-outer_w / 2.0 - plate_thickness, outer_w / 2.0 + plate_thickness]),
        yaxis=dict(title='Y [mm]', gridcolor='lightgray', range=[-outer_d / 2.0 - plate_thickness, outer_d / 2.0 + plate_thickness]),
        zaxis=dict(title='Z [mm]', gridcolor='lightgray', range=[-outer_h / 2.0 - plate_thickness, outer_h / 2.0 + plate_thickness]),
        camera_projection=dict(type='perspective'),
        camera=dict(
            eye=dict(x=1.8, y=1.8, z=1.2)
        ),
        # Keep millimeter proportions true across all axes.
        aspectmode='data'
    ),
    hovermode='closest',
    template='plotly_white'
)

output_file = 'box_output/visualization_interactive.html'
print(f"Saving interactive visualization...", file=sys.stderr)
fig.write_html(output_file, default_width='100%', default_height='100%', config={'responsive': True})
print(f"Interactive visualization saved to: {output_file}", file=sys.stderr)
print(f"Open in browser to interact with the 3D scene!", file=sys.stderr)
