# Analysis: Why `<g>` Tags Affect Model Performance

## Executive Summary

**Problem**: The model produces worse results on SVG files without `<g>` tags, even though it doesn't explicitly use layer ID information.

**Root Cause**: Element **ordering** differs between SVG files with and without `<g>` tags due to parsing logic in `parse_svg.py`. This creates a distribution shift that degrades model performance.

**Solution**: Fixed `parse_svg.py` to maintain consistent document ordering regardless of `<g>` tag presence.

---

## Detailed Analysis

### 1. Layer IDs are NOT Used by the Model ✓

Confirmed that `layerIds` do not appear anywhere in the model code:
```bash
grep -r "layerIds" svgnet/
# No matches found
```

The model only uses: `args`, `lengths`, `commands`, `semanticIds`, `instanceIds`

### 2. The Real Issue: Element Ordering

#### Original Parsing Behavior

**With `<g>` tags** (lines 200-218 in `parse_svg.py`):
```python
for g in root.iter(ns + 'g'):
    for path in g.iter(ns + 'path'):      # Paths in THIS group
        parse_element(...)
    for circle in g.iter(ns + 'circle'):  # Circles in THIS group
        parse_element(...)
    for ellipse in g.iter(ns + 'ellipse'): # Ellipses in THIS group
        parse_element(...)
```
**Result**: Elements grouped by layer → **maintains spatial coherence**

**Without `<g>` tags** (original lines 220-235):
```python
for path in root.iter(ns + 'path'):      # ALL paths
    parse_element(...)
for circle in root.iter(ns + 'circle'):  # ALL circles
    parse_element(...)
for ellipse in root.iter(ns + 'ellipse'): # ALL ellipses
    parse_element(...)
```
**Result**: Elements grouped by type → **breaks spatial coherence**

#### Example from Test File (0003-0010.svg)

| Format | Circle Positions | Pattern |
|--------|------------------|---------|
| **With `<g>` tags** | `[207, 208, 209, 225, 299, ...]` | Circles interspersed: `LLLL...CCC...LL...CC...` |
| **Without `<g>` tags** (BEFORE FIX) | `[407, 408, 409, 410, 411, ...]` | All circles at end: `LLLL...LLLCCCCCCCCC` |

### 3. Why This Affects Performance

1. **Training Distribution**: The model was trained on data with `<g>` tags (spatial coherence)
2. **Distribution Shift**: Testing without `<g>` tags presents data in unfamiliar ordering
3. **Spatial Context Loss**: Related elements (e.g., door and its frame) are far apart in sequence
4. **Augmentation Impact**: Cutmix assumes spatially coherent sequences (lines 177-197 in `svg.py`)

Even though the model:
- Processes data as a point cloud
- Shuffles during training (line 206-210 in `svg.py`)

The **test-time ordering** matters because:
- No shuffling at inference
- Model learned patterns from training distribution
- Spatial coherence aids instance segmentation

---

## The Fix

### Modified `parse_svg.py` (lines 219-233)

```python
else:
    # New format: paths are directly under root (no <g> tags)
    # To maintain consistent ordering, collect all elements with their document order
    id = 1
    all_elements = []
    
    # Collect all drawable elements with their position in the document
    for i, child in enumerate(root):
        if child.tag in [ns + 'path', ns + 'circle', ns + 'ellipse']:
            all_elements.append((i, child))
    
    # Process in document order (maintains spatial coherence like g-tag version)
    for _, element in all_elements:
        parse_element(element, ns, id, commands, args, lengths, semanticIds, 
                     instanceIds, strokes, layerIds, widths, inst_infos)
```

### Key Change

**BEFORE**: Iterate by element type (all paths → all circles → all ellipses)
**AFTER**: Iterate by document order (preserve original SVG sequence)

### Verification

```
✓ Commands identical: True
✓ Args identical: True
✓ Circle positions with g: [207, 208, 209, 225, 299, 300, 315, 335, 336]
✓ Circle positions without g: [207, 208, 209, 225, 299, 300, 315, 335, 336]
```

---

## Next Steps

### Option 1: Reprocess Your Test Data (Recommended)

```bash
# Activate environment
source /path/to/conda.sh
conda activate symp
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"

# Reprocess SVG files without g tags
python parse_svg.py --split floorplan_demo_wo_layer \
    --data_dir ./dataset/floorplan_demo_wo_layer/svg_gt/

# Now test with updated JSON files
python tools/inference.py configs/svg/svg_pointT.yaml \
    configs/svg/best.pth \
    --datadir dataset/json/floorplan_demo_wo_layer \
    --out results/floorplan_demo_wo_layer_fixed
```

### Option 2: Alternative Solutions

If you can't reprocess:

1. **Add wrapper `<g>` tags to SVG files**:
```python
# Quick fix: wrap all elements in a single <g> tag
import xml.etree.ElementTree as ET
tree = ET.parse('input.svg')
root = tree.getroot()
g = ET.SubElement(root, 'g')
for child in list(root):
    if child.tag.endswith(('path', 'circle', 'ellipse')):
        root.remove(child)
        g.append(child)
tree.write('output.svg')
```

2. **Train new model with mixed ordering** (requires retraining)

---

## Lessons Learned

1. **Implicit Assumptions**: Even when models don't explicitly use features (like `layerIds`), data format can affect performance through ordering
2. **Document Order Matters**: SVG parsing should preserve spatial relationships
3. **Test/Train Distribution**: Always verify test data follows same preprocessing as training data
4. **XML Iteration Semantics**: `root.iter()` vs `root` direct iteration produces different orders

---

## Files Modified

- `parse_svg.py`: Fixed element ordering logic (lines 219-233)

## Files Analyzed

- `parse_svg.py`: SVG parsing and JSON generation
- `svgnet/data/svg.py`: Dataset loader (confirmed `layerIds` not used)
- `svgnet/data/svg2.py`: Alternative dataset loader (confirmed `layerIds` not used)
- `tools/inference.py`: Inference script
- `tools/test.py`: Testing script

