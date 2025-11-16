
import math,re
import os,glob,json
import shutil
import xml.etree.ElementTree as ET
from svgpathtools import parse_path
from collections import defaultdict
import numpy as np
#from sklearn.metrics.pairwise import euclidean_distances

LABEL_NUM = 35
COMMANDS = ['Line', 'Arc','circle', 'ellipse']
import mmcv, argparse

def parse_args():
    '''
    Arguments
    '''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--split', type=str, default="test",
                        help='the split of dataset')
    parser.add_argument('--data_dir', type=str, default="./dataset/test/test/svg_gt",
                        help='save the downloaded data')
    args = parser.parse_args()
    return args



def parse_color_to_rgb(color_str):
    """
    Convert various color formats to RGB tuple.
    Handles: rgb(r,g,b), named colors (black, white, red, etc.), hex colors.
    """
    if color_str.startswith('rgb'):
        # Extract numbers from rgb(r,g,b) format
        rgb_values = re.findall(r'\d+', color_str)
        if len(rgb_values) >= 3:
            return list(map(int, rgb_values[:3]))
    
    # Named color mapping
    color_map = {
        'black': [0, 0, 0],
        'white': [255, 255, 255],
        'red': [255, 0, 0],
        'green': [0, 255, 0],
        'blue': [0, 0, 255],
        'yellow': [255, 255, 0],
        'cyan': [0, 255, 255],
        'magenta': [255, 0, 255],
        'gray': [128, 128, 128],
        'grey': [128, 128, 128],
    }
    
    if color_str.lower() in color_map:
        return color_map[color_str.lower()]
    
    # Try to extract any numbers as fallback
    rgb_values = re.findall(r'\d+', color_str)
    if len(rgb_values) >= 3:
        return list(map(int, rgb_values[:3]))
    
    # Default to black if parsing fails
    return [0, 0, 0]


def parse_element(element, ns, layer_id, commands, args, lengths, semanticIds, instanceIds, strokes, layerIds, widths, inst_infos, counts):
    """
    Parse individual SVG elements (path, circle, ellipse).
    Handles both elements with and without instanceId/semanticId attributes.
    """
    # Get or set default values
    semanticId = int(element.attrib.get('semanticId', 0)) - 1 if 'semanticId' in element.attrib else LABEL_NUM
    instanceId = int(element.attrib.get('instanceId', -1)) if 'instanceId' in element.attrib else -1
    
    # Get or default stroke color
    stroke_color = element.attrib.get('stroke', 'black')
    rgb = parse_color_to_rgb(stroke_color)
    strokes.append(rgb)
    
    # Get or default stroke width
    stroke_width = element.attrib.get("stroke-width", "0.5")
    widths.append(float(stroke_width))
    
    # Parse path elements
    if element.tag == ns + 'path':
        try:
            path_repre = parse_path(element.attrib['d'])
        except Exception as e:
            raise RuntimeError("Parse path failed! {}, {}".format(element.attrib.get('d', 'N/A'), e))
        
        path_type = path_repre[0].__class__.__name__
        commands.append(COMMANDS.index(path_type))
        length = path_repre.length()
        lengths.append(length)
        layerIds.append(layer_id)
        
        semanticIds.append(semanticId)
        instanceIds.append(instanceId)
        
        inds = [0, 1/3, 2/3, 1.0]
        arg = []


        # Handle degenerate paths (zero-length) by checking path length first
        try:
            path_length = path_repre.length()
            if path_length < 1e-10:
                counts['c2'] += 1
                # Degenerate path - use start point for all samples
                start_point = path_repre[0].start
                for _ in inds:
                    arg.extend([start_point.real, start_point.imag])
            else:
                counts['c1'] += 1
                # Normal path - sample points
                for ind in inds:
                    point = path_repre.point(ind)
                    arg.extend([point.real, point.imag])

        except (RuntimeError, ValueError) as e:
            # If point sampling fails, use the start point
            try:
                start_point = path_repre[0].start
                for _ in inds:
                    arg.extend([start_point.real, start_point.imag])
                counts['e1'] += 1
            except (AttributeError, IndexError):
                # Last resort: use origin
                for _ in inds:
                    arg.extend([0.0, 0.0])
                counts['e2'] += 1
        args.append(arg)
        inst_infos[(instanceId, semanticId)].extend(arg)
    
    # Parse circle elements
    elif element.tag == ns + 'circle':
        cx = float(element.attrib['cx'])
        cy = float(element.attrib['cy'])
        r = float(element.attrib['r'])
        circle_len = 2 * math.pi * r
        lengths.append(circle_len)
        semanticIds.append(semanticId)
        instanceIds.append(instanceId)
        commands.append(COMMANDS.index("circle"))
        layerIds.append(layer_id)
        
        thetas = [0, math.pi/2, math.pi, 3 * math.pi/2]
        arg = []
        for theta in thetas:
            x, y = cx + r * math.cos(theta), cy + r * math.sin(theta)
            arg.extend([x, y])
        args.append(arg)
        inst_infos[(instanceId, semanticId)].extend(arg)
    
    # Parse ellipse elements
    elif element.tag == ns + 'ellipse':
        cx = float(element.attrib['cx'])
        cy = float(element.attrib['cy'])
        rx = float(element.attrib['rx'])
        ry = float(element.attrib['ry'])
        
        if rx > ry:
            a, b = rx, ry
        else:
            a, b = ry, rx
        
        ellipse_len = 2 * math.pi * b + 4 * (a - b)
        lengths.append(ellipse_len)
        commands.append(COMMANDS.index("ellipse"))
        semanticIds.append(semanticId)
        instanceIds.append(instanceId)
        layerIds.append(layer_id)
        
        thetas = [0, math.pi/2, math.pi, 3 * math.pi/2]
        arg = []
        for theta in thetas:
            x, y = cx + a * math.cos(theta), cy + b * math.sin(theta)
            arg.extend([x, y])
        args.append(arg)
        inst_infos[(instanceId, semanticId)].extend(arg)
    
    return counts


def parse_svg(svg_file):
    tree = ET.parse(svg_file)
    root = tree.getroot()
    ns = root.tag[:-3]
    minx, miny, width, height = [int(float(x)) for x in root.attrib['viewBox'].split(' ')]
    
    commands = []
    args = []  # (x1,y1,x2,y2,x3,y3,x4,y4) 4points
    lengths = []
    semanticIds = []
    instanceIds = []
    strokes = []
    layerIds = []
    widths = []
    inst_infos = defaultdict(list)
    
    # Check if SVG has <g> tags
    groups = list(root.iter(ns + 'g'))

    counts = {'c1': 0, 'c2': 0, 'e1': 0, 'e2': 0}
    
    if len(groups) > 0:
        # Original format: paths are inside <g> tags
        id = 0
        for g in root.iter(ns + 'g'):
            id += 1
            # path
            for path in g.iter(ns + 'path'):
                counts = parse_element(path, ns, id, commands, args, lengths, semanticIds, 
                             instanceIds, strokes, layerIds, widths, inst_infos, counts)
            
            # circle
            for circle in g.iter(ns + 'circle'):
                counts = parse_element(circle, ns, id, commands, args, lengths, semanticIds, 
                             instanceIds, strokes, layerIds, widths, inst_infos, counts)
            
            # ellipse
            for ellipse in g.iter(ns + 'ellipse'):
                counts = parse_element(ellipse, ns, id, commands, args, lengths, semanticIds, 
                             instanceIds, strokes, layerIds, widths, inst_infos, counts)

    else:
        # paths are directly under root (no <g> tags)
        # To maintain consistent ordering, collect all elements with their document order
        id = 1
        all_elements = []
        
        # Collect all drawable elements with their position in the document
        for i, child in enumerate(root):
            if child.tag in [ns + 'path', ns + 'circle', ns + 'ellipse']:
                all_elements.append((i, child))
        
        # Process in document order (maintains spatial coherence like g-tag version)
        for _, element in all_elements:
            counts = parse_element(element, ns, id, commands, args, lengths, semanticIds, 
                         instanceIds, strokes, layerIds, widths, inst_infos, counts)

    # this approach is not good, because it will break the spatial coherence
    # else:
    #     id = 1
    #     # Parse all path elements directly from root
    #     for path in root.iter(ns + 'path'):
    #         parse_element(path, ns, id, commands, args, lengths, semanticIds, 
    #                      instanceIds, strokes, layerIds, widths, inst_infos)

    #     # Parse all circle elements directly from root
    #     for circle in root.iter(ns + 'circle'):
    #         parse_element(circle, ns, id, commands, args, lengths, semanticIds, 
    #                      instanceIds, strokes, layerIds, widths, inst_infos)

    #     # Parse all ellipse elements directly from root
    #     for ellipse in root.iter(ns + 'ellipse'):
    #         parse_element(ellipse, ns, id, commands, args, lengths, semanticIds, 
    #                      instanceIds, strokes, layerIds, widths, inst_infos)
    
    print(f"normal path: {counts['c1']}, degenerate path: {counts['c2']}, error1: {counts['e1']}, 0 used: {counts['e2']}")
    # Validate results
    if len(args) == 0:
        # Empty SVG - return minimal structure
        return {
            "commands": [],
            "args": [],
            "lengths": [],
            "semanticIds": [],
            "instanceIds": [],
            "width": width,
            "height": height,
            "obj_cts": [],
            "boxes": [],
            "rgb": [],
            "layerIds": [],
            "widths": []
        }
    
    assert len(args) == len(lengths), 'error'
    assert len(semanticIds) == len(instanceIds), 'error'
    
    obj_cts = []
    obj_boxes = []
    for (inst_id, sem_id), coords in inst_infos.items():
        if inst_id < 0:
            continue
        coords = np.array(coords).reshape(-1, 2)
        x1, y1 = np.min(coords[:, 0]), np.min(coords[:, 1])
        x2, y2 = np.max(coords[:, 0]), np.max(coords[:, 1])
        obj_cts.append([(x1+x2)/2, (y1+y2)/2, 0, inst_id])
        obj_boxes.append([x1, y1, x2, y2, sem_id])
    
    coords = np.array(args).reshape(-1, 4, 2)
    
    json_dicts = {
        "commands": commands,
        "args": args,
        "lengths": lengths,
        "semanticIds": semanticIds,
        "instanceIds": instanceIds,
        "width": width,
        "height": height,
        "obj_cts": obj_cts,  # (x,y,z)
        "boxes": obj_boxes,
        "rgb": strokes,
        "layerIds": layerIds,
        "widths": widths
    }
    return json_dicts

def save_json(json_dicts,out_json):
    json.dump(json_dicts, open(out_json, 'w'), indent=4)
    
def process(svg_file):
    
    json_dicts = parse_svg(svg_file)
    filename = svg_file.split('/')[-1].replace('.svg','.json').replace(" ","_")
    out_json = os.path.join(save_dir,filename)
    save_json(json_dicts,out_json)

    # Copy the original SVG file to dataset/svg/{split}/ directory
    svg_filename = svg_file.split("/")[-1].replace(" ","_")
    svg_out_path = os.path.join(svg_save_dir, svg_filename)
    shutil.copy2(svg_file, svg_out_path)


if __name__=="__main__":
    
    args = parse_args()
    data_dir = args.data_dir
    svg_paths = sorted(glob.glob(os.path.join(data_dir,'*.svg')))

    save_dir = os.path.join("./dataset/json/",args.split )
    svg_save_dir = os.path.join("./dataset/svg/",args.split)
    os.makedirs(save_dir,exist_ok=True)
    os.makedirs(svg_save_dir,exist_ok=True)

    mmcv.track_parallel_progress(process,svg_paths,64)


    


    
            
            
            

