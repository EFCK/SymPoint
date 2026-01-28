import os
import xml.etree.ElementTree as ET
from svgpathtools import parse_path
import re
import argparse
import glob
import mmcv
import random
import warnings

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

UNNECESSARY_CLASSES = {
    3,  # folding_door
    5,  # rolling_door
    7,  # bay_window
    8,  # blind_window
    9,  # opening symbol
    11,  # bed
    14,  # TV_cabinet
    15,  # Wardrobe
    17,  # gas_stove
    19,  # refrigerator
    20,  # airconditioner
    21,  # bath
    22,  # bath_tub
    23,  # washing_machine
}


def _ns(tag):
    return f"{{{SVG_NS}}}{tag}"


def _float_attr(elem, name, default=None):
    v = elem.attrib.get(name)
    return default if v is None else float(v)


def _bbox_union(b1, b2):
    if b1 is None:
        return b2
    if b2 is None:
        return b1
    x0a, y0a, x1a, y1a = b1
    x0b, y0b, x1b, y1b = b2
    return (min(x0a, x0b), min(y0a, y0b), max(x1a, x1b), max(y1a, y1b))


def _bbox_path(d, filename=None, prim_id=None):
    if not d:
        return None
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", RuntimeWarning)
            p = parse_path(d)
            if len(p) == 0:
                return None
            xmin, xmax, ymin, ymax = p.bbox()

            # Print warning if any (deduplicated)
            if caught_warnings and filename:
                unique_warnings = list(set(str(w.message) for w in caught_warnings))
                print(
                    f"[warning] {filename}: primitiveId[{prim_id}] - {'; '.join(unique_warnings)}"
                )

            return (xmin, ymin, xmax, ymax)
    except Exception:
        return None


def safe_transform_path(d, x0, y0, s, px, py):
    """
    Safely transform a path. Returns (new_d, error_type) tuple.
    If successful, error_type is None. Otherwise, returns (None, error_type).
    """
    if not d:
        return None, "EMPTY_PATH"
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", RuntimeWarning)
            path = parse_path(d)
            if len(path) == 0:
                return None, "EMPTY_PATH"
            path = path.translated(-x0 - 1j * y0).scaled(s, s).translated(px + 1j * py)
            new_d = path.d()

            # Check if any warnings were caught (deduplicated)
            if caught_warnings:
                unique_warnings = list(set(str(w.message) for w in caught_warnings))
                return new_d, "; ".join(unique_warnings)

            return new_d, None
    except Exception as e:
        return None, f"TRANSFORM_ERROR: {type(e).__name__}"
    try:
        p = parse_path(d)
        if len(p) == 0:
            return None
        xmin, xmax, ymin, ymax = p.bbox()
        return (xmin, ymin, xmax, ymax)
    except Exception:
        return None


def _bbox_circle(cx, cy, r):
    return (cx - r, cy - r, cx + r, cy + r)


def _bbox_ellipse(cx, cy, rx, ry):
    return (cx - rx, cy - ry, cx + rx, cy + ry)


def parse_rotation(transform_str):
    """
    Extracts rotation angle and optional (cx, cy) from a transform string.
    Returns (angle, cx, cy) where cx, cy may be None.
    Works for e.g. 'rotate(-105, 350, 350)' or 'rotate(45)'.
    """
    if not transform_str:
        return None, None, None

    # Regex that matches rotate(angle) or rotate(angle, cx, cy)
    match = re.search(
        r"rotate\(\s*([-\d.]+)(?:[ ,]+([-\d.]+)[ ,]+([-\d.]+))?\s*\)", transform_str
    )
    if not match:
        return None, None, None

    angle = float(match.group(1))
    cx = float(match.group(2)) if match.group(2) is not None else None
    cy = float(match.group(3)) if match.group(3) is not None else None

    return angle, cx, cy


def should_convert_to_background(elem):
    """
    Check if element should be converted to background based on semanticId.
    Returns True if element should be converted to background.
    """
    semantic_id = elem.attrib.get("semanticId")

    if semantic_id is None:
        return False

    try:
        # SVG uses 1-based indexing, convert to 0-based
        sem_id = int(semantic_id) - 1
        return sem_id in UNNECESSARY_CLASSES
    except (ValueError, TypeError):
        return False


def convert_unnecessary_to_background(root):
    """
    Convert elements with unnecessary classes to background by removing their IDs.
    Returns the number of elements converted.
    """
    converted_count = 0

    # Find all groups
    groups = root.findall(".//" + _ns("g"))

    # If no groups, work directly with root
    if not groups:
        groups = [root]

    # Process elements in each group
    for group in groups:
        for elem in list(group):
            tag = elem.tag
            # Only process drawing elements
            if any(tag.endswith(t) for t in ["path", "circle", "ellipse"]):
                if should_convert_to_background(elem):
                    # Convert to background: remove both semanticId and instanceId
                    if "semanticId" in elem.attrib:
                        del elem.attrib["semanticId"]
                    if "instanceId" in elem.attrib:
                        del elem.attrib["instanceId"]
                    converted_count += 1

    return converted_count


def normalize_viewbox_with_padding_baked(
    svg_in,
    svg_out,
    target=700,
    pad_frac=0.08,
    scale_stroke=True,
    constant_stroke_width=None,
    remove_classes=False,
    use_union_bbox=False,
):
    tree = ET.parse(svg_in)
    root = tree.getroot()

    # Convert unnecessary classes to background if requested
    converted_count = 0
    if remove_classes:
        converted_count = convert_unnecessary_to_background(root)

    paths = root.findall(".//" + _ns("path"))
    circles = root.findall(".//" + _ns("circle"))
    ellips = root.findall(".//" + _ns("ellipse"))

    # ---- compute tight bbox
    svg_filename = os.path.basename(svg_in)
    content_bbox = None
    for idx, p in enumerate(paths):
        prim_id = p.get("primitiveId", idx)
        content_bbox = _bbox_union(
            content_bbox, _bbox_path((p.get("d") or "").strip(), svg_filename, prim_id)
        )

    for c in circles:
        cx = _float_attr(c, "cx")
        cy = _float_attr(c, "cy")
        r = _float_attr(c, "r")
        if None not in (cx, cy, r):
            content_bbox = _bbox_union(content_bbox, _bbox_circle(cx, cy, r))

    for e in ellips:
        cx = _float_attr(e, "cx")
        cy = _float_attr(e, "cy")
        rx = _float_attr(e, "rx")
        ry = _float_attr(e, "ry")
        if None not in (cx, cy, rx, ry):
            content_bbox = _bbox_union(content_bbox, _bbox_ellipse(cx, cy, rx, ry))

    if content_bbox is None:
        root.set("viewBox", f"0 0 {target} {target}")
        tree.write(svg_out)
        print(
            f"[baked] {os.path.basename(svg_in)} → {os.path.basename(svg_out)} (empty content)"
        )
        return

    if not use_union_bbox:
        # use viewBox if available and valid
        vb = root.get("viewBox")
        if vb:
            try:
                x, y, w, h = map(float, vb.strip().split())
                if w > 1e-9 and h > 1e-9:
                    content_bbox = (x, y, x + w, y + h)
            except Exception:
                pass

    x0, y0, x1, y1 = content_bbox
    cw, ch = max(1e-9, x1 - x0), max(1e-9, y1 - y0)

    pad = pad_frac * target
    avail = max(1e-9, target - 2 * pad)
    s = min(avail / cw, avail / ch)

    rw, rh = cw * s, ch * s
    px = (target - rw) * 0.5
    py = (target - rh) * 0.5

    # ---- paths
    for idx, p in enumerate(paths):
        d = (p.get("d") or "").strip()
        if not d:
            continue

        new_d, error = safe_transform_path(d, x0, y0, s, px, py)

        # Print warning/error if any
        if error:
            prim_id = p.get("primitiveId", idx)
            print(f"[warning] {svg_filename}: primitiveId[{prim_id}] - {error}")

        # Skip if transform failed (new_d is None)
        if new_d is None:
            continue

        p.set("d", new_d)

        if scale_stroke:
            sw = p.get("stroke-width")
            if sw:
                try:
                    p.set("stroke-width", str(float(sw) * s))
                except:
                    pass

        if constant_stroke_width is not None:
            sw = p.get("stroke-width")
            if sw:
                try:
                    p.set("stroke-width", constant_stroke_width)
                except:
                    pass

        if "transform" in p.attrib:
            del p.attrib["transform"]

    # ---- circles
    for idx, c in enumerate(circles):
        cx = _float_attr(c, "cx")
        cy = _float_attr(c, "cy")
        r = _float_attr(c, "r")
        if None in (cx, cy, r):
            prim_id = c.get("primitiveId", idx)
            print(f"[warning] {svg_filename}: primitiveId[{prim_id}] - MISSING_ATTR")
            continue
        cx_t = (cx - x0) * s + px
        cy_t = (cy - y0) * s + py
        r_t = r * s
        c.set("cx", f"{cx_t}")
        c.set("cy", f"{cy_t}")
        c.set("r", f"{r_t}")
        if scale_stroke:
            sw = c.get("stroke-width")
            if sw:
                try:
                    c.set("stroke-width", str(float(sw) * s))
                except:
                    pass

        if constant_stroke_width is not None:
            sw = c.get("stroke-width")
            if sw:
                try:
                    c.set("stroke-width", constant_stroke_width)
                except:
                    pass
        if "transform" in c.attrib:
            del c.attrib["transform"]

    # ---- ellipses
    # ---- ellipses
    for idx, e in enumerate(ellips):
        # preserve rotation
        original_transform = e.attrib.get("transform")
        angle, rcx, rcy = parse_rotation(original_transform)
        # extract rotation if any from format transform="rotate(angle cx cy)"
        # rotation = 0
        cx = _float_attr(e, "cx")
        cy = _float_attr(e, "cy")
        rx = _float_attr(e, "rx")
        ry = _float_attr(e, "ry")
        if None in (cx, cy, rx, ry):
            prim_id = e.get("primitiveId", idx)
            print(f"[warning] {svg_filename}: primitiveId[{prim_id}] - MISSING_ATTR")
            continue
        cx_t = (cx - x0) * s + px
        cy_t = (cy - y0) * s + py
        rx_t = rx * s
        ry_t = ry * s
        e.set("cx", f"{cx_t}")
        e.set("cy", f"{cy_t}")
        e.set("rx", f"{rx_t}")
        e.set("ry", f"{ry_t}")
        if scale_stroke:
            sw = e.get("stroke-width")
            if sw:
                try:
                    e.set("stroke-width", str(float(sw) * s))
                except:
                    pass

        if constant_stroke_width is not None:
            sw = e.get("stroke-width")
            if sw:
                try:
                    e.set("stroke-width", constant_stroke_width)
                except:
                    pass

        if "transform" in e.attrib:
            del e.attrib["transform"]
        # set back the original rotation with updated center
        if original_transform is not None:
            rotation = f"rotate({angle} {cx_t} {cy_t})"
            e.set("transform", rotation)

    root.set("viewBox", f"0 0 {target} {target}")
    root.set("width", str(target))
    root.set("height", str(target))

    os.makedirs(os.path.dirname(svg_out), exist_ok=True)
    tree.write(svg_out)

    if remove_classes and converted_count > 0:
        print(
            f"[baked] {os.path.basename(svg_in)} → {os.path.basename(svg_out)}  s={s:.6f}, pad={pad_frac:.2f}, converted_to_bg={converted_count}"
        )
    else:
        print(
            f"[baked] {os.path.basename(svg_in)} → {os.path.basename(svg_out)}  s={s:.6f}, pad={pad_frac:.2f}"
        )


def process(
    svg_file,
    output_dir,
    target,
    pad_frac,
    scale_stroke,
    constant_stroke_width,
    remove_classes=False,
    use_union_bbox=False,
):
    """
    Process a single SVG file - used for parallel processing.
    """
    svg_filename = os.path.basename(svg_file)
    if svg_filename.endswith("_predicted_editted.svg"):
        svg_filename = svg_filename.replace("_predicted_editted.svg", ".svg")
    elif svg_filename.endswith("_pred_editted.svg"):
        svg_filename = svg_filename.replace("_pred_editted.svg", ".svg")
    elif svg_filename.endswith("_predicted.svg"):
        svg_filename = svg_filename.replace("_predicted.svg", ".svg")
    svg_out_path = os.path.join(output_dir, svg_filename)

    normalize_viewbox_with_padding_baked(
        svg_file,
        svg_out_path,
        target=target,
        pad_frac=pad_frac,
        scale_stroke=scale_stroke,
        constant_stroke_width=constant_stroke_width,
        remove_classes=remove_classes,
        use_union_bbox=use_union_bbox,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="Input directory containing SVG files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for processed SVG files",
    )
    parser.add_argument(
        "--target", type=int, default=140, help="Target size for viewBox"
    )
    parser.add_argument("--pad_frac", type=float, default=0.00, help="Padding fraction")
    parser.add_argument(
        "--scale_stroke", action="store_false", help="Scale stroke widths"
    )
    parser.add_argument(
        "--constant_stroke_width",
        type=str,
        default=0.1,
        help="Set constant stroke width",
    )
    parser.add_argument(
        "--nproc", type=int, default=64, help="Number of parallel processes"
    )
    parser.add_argument(
        "--remove_classes",
        action="store_true",
        help="Remove unnecessary classes (folding_door, bed, etc.)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of random samples to process (if not set, process all files)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible sampling"
    )
    parser.add_argument(
        "--use_union_bbox",
        action="store_true",
        help="Use union of content bboxes instead of viewBox for normalization",
    )

    args = parser.parse_args()

    if args.constant_stroke_width is not None:
        args.constant_stroke_width = str(args.constant_stroke_width)
    data_dir = args.input
    output_dir = args.output

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    svg_paths = sorted(glob.glob(os.path.join(data_dir, "*.svg")))

    # Random sampling if --n_samples is provided
    if args.n_samples is not None and args.n_samples > 0:
        if args.n_samples >= len(svg_paths):
            print(
                f"Warning: --n_samples ({args.n_samples}) >= total files ({len(svg_paths)}). Using all files."
            )
        else:
            random.seed(args.seed)
            svg_paths = random.sample(svg_paths, args.n_samples)
            print(f"Randomly sampled {args.n_samples} files (seed={args.seed})")

    if len(svg_paths) == 0:
        print(f"No SVG files found in {data_dir}")
    else:
        print(f"Found {len(svg_paths)} SVG files. Processing in parallel...")

        # Create a partial function with fixed arguments for parallel processing
        def process_wrapper(svg_file):
            return process(
                svg_file,
                output_dir,
                args.target,
                args.pad_frac,
                args.scale_stroke,
                args.constant_stroke_width,
                args.remove_classes,
                args.use_union_bbox,
            )

        mmcv.track_parallel_progress(process_wrapper, svg_paths, args.nproc)

        if args.remove_classes:
            print(f"Processing complete. Output saved to {output_dir}")
            print(
                f"Note: Unnecessary classes have been removed (see output for counts)"
            )
        else:
            print(f"Processing complete. Output saved to {output_dir}")
