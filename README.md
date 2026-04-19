# Cuttinger - Rebuild Guide

Dieses Projekt erzeugt aus einem 3D-Scan (`model.stl`) einen laserschneidbaren Setzkasten als SVG-Teilesatz.

<video src="media/example-output.webm" autoplay loop muted playsinline controls style="max-width: 100%; height: auto;"></video>

Die zentrale Datei ist [cuttinger.py](cuttinger.py).

## Was [cuttinger.py](cuttinger.py) macht

1. Laedt das STL-Modell.
2. Wendet den Transform (Scale, Rotation, Position) an.
3. Misst die Bounding Box des transformierten Meshes.
4. Berechnet automatisch Innen- und Aussenmasse des Kastens mit Padding.
5. Erzeugt Finger-Joint-Panels (Back, Top, Bottom, Left, Right).
6. Schneidet Divider-Ausschnitte ueber echte Mesh-Slices (Trimesh + Shapely).
7. Schreibt einzelne SVG-Teile fuer Panels und Divider.

## Projektstruktur

- [cuttinger.py](cuttinger.py): Generator fuer alle SVG-Bauteile
- [visualize_interactive.py](visualize_interactive.py): interaktive 3D-Visualisierung (Plotly)
- [model.stl](model.stl): Eingangs-Scan
- [box_output](box_output): Ausgabeordner mit erzeugten SVGs/Visualisierungen

## Voraussetzungen

- Python 3.10+
- Pakete:
  - `numpy`
  - `trimesh`
  - `shapely`
  - `svgwrite`
  - `plotly` (nur fuer [visualize_interactive.py](visualize_interactive.py))

## Wie komme ich zu einer 3D-STL-Datei?

Du brauchst als Input eine `model.stl`. Typische Wege:

1. Mit 3D-Scanner (am einfachsten)
   - Objekt scannen (z. B. Creality, EinScan, Revopoint, Artec oder MakerSpace-Scanner).
   - Mesh im Scanner-Tool bereinigen (Loecher schliessen, Rauschen entfernen, unnötige Teile abschneiden).
   - Als `STL` exportieren und im Projekt als `model.stl` ablegen.

2. Mit Handy per Photogrammetrie
   - Viele Fotos rund ums Objekt machen (gleichmaessiges Licht, wenig Spiegelungen).
   - In Apps/Tools wie Polycam, RealityScan oder Meshroom zu einem Mesh rechnen.
   - Mesh bereinigen und als `STL` exportieren.

3. Aus CAD oder Online-Bibliothek
   - Eigene CAD-Datei als `STL` exportieren (Fusion, FreeCAD, SolidWorks etc.).
   - Oder ein bestehendes Modell (z. B. Printables, Thingiverse) laden und als `model.stl` verwenden.

Wichtige Tipps vor dem Export:

- Geschlossenes Mesh (watertight) bevorzugen.
- Einheit im Blick behalten (meist mm).
- Modell moeglichst nahe an Realmass halten; Feintuning geht spaeter ueber `stl_scale` in [cuttinger.py](cuttinger.py).

## Setup

Im Projektordner:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy trimesh shapely svgwrite plotly
```

## Generierung starten

```powershell
python cuttinger.py
```

Ausgabe landet in [box_output](box_output), u. a.:

- `back.svg`
- `front.svg`
- `top.svg`
- `bottom.svg`
- `left.svg`
- `right.svg`
- `div_x_*.svg`
- `div_y_*.svg`

## Visualisierung

Interaktiv:

```powershell
python visualize_interactive.py
```

Hinweis: `python cuttinger.py` startet diese Visualisierung am Ende ebenfalls automatisch.

Erzeugt:

- [box_output/visualization_interactive.html](box_output/visualization_interactive.html)

## Wichtige Parameter in [cuttinger.py](cuttinger.py)

### STL und Transform

- `stl_file`
- `stl_scale`
- `stl_rot_deg`
- `stl_pos`

### Boxdimensionen / Raster

- `pad_x`, `pad_y`, `pad_z`: Abstand vom Objekt zur Innenwand
- `cells_x`, `cells_y`: Anzahl Faecher in X/Z-Richtung
- `mat`: Materialstaerke

### Fertigung / Passung

- `kerf`: Laser-Schnittbreite
- `finger_target`: Zielbreite fuer Finger-Joints
- `fit_clearance`: Zusatzspiel fuer Mesh-basierte Ausschnitte
- `slot_w`, `panel_slot_w`
- `vdiv_panel_slot_x_offset`

Hinweis: `vdiv_panel_slot_x_offset` ist als Feintuning-Parameter vorhanden, falls CAD-/SVG-Koordinaten und reale Montage leicht abweichen.

### Slice-Qualitaet

- `divider_section_samples`: Anzahl Slicing-Ebenen ueber Divider-Dicke
- `simplify_tol`, `min_polygon_area`, `concave_ratio`
- `remove_undercuts`

## Rebuild von Null (Checkliste)

1. `model.stl` ins Projekt legen.
2. In [cuttinger.py](cuttinger.py) `stl_*`-Transform einstellen.
3. `pad_*`, `cells_*`, `mat`, `kerf` setzen.
4. `python cuttinger.py` ausfuehren.
5. Teile in [box_output](box_output) pruefen (`front/back/top/bottom/left/right`, `div_x_*`, `div_y_*`).
6. `python visualize_interactive.py` zur 3D-Passformkontrolle ausfuehren.
7. Falls noetig Offsets/Passung anpassen und erneut generieren.

## Typische Probleme

- Teile passen nicht zusammen:
  - `mat`, `kerf`, `panel_slot_w` und Offsets pruefen.
- Divider-Ausschnitte wirken zu eng/weit:
  - `fit_clearance` und `divider_section_samples` anpassen.
- Mesh-Schnittflaechen sehen unruhig aus:
  - `simplify_tol`, `min_polygon_area`, `concave_ratio` feinjustieren.

## Ziel der Pipeline

Reproduzierbar aus einem beliebigen STL-Objekt einen steckbaren, laserfertigen Setzkasten erzeugen, inklusive automatischer Objekt-Cutouts und visueller 3D-Kontrolle.
