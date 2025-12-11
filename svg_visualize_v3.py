import json,os,glob
import numpy as np
import xml.etree.ElementTree as ET
from collections import Counter
from svgpathtools import parse_path
import re, math
from svgnet.data.svg import SVG_CATEGORIES
from pathlib import Path
from svgnet.evaluation import InstanceEval
from svgnet.util  import get_root_logger

from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

UNNECESSARY_CLASSES = {
    35: "background",

    3: "folding_door",
    5: "rolling_door",

    7: "bay_window",
    8: "blind_window",
    9: "opening symbol",
    
    11: "bed",
    14: "TV_cabinet",
    15: "Wardrobe",
    17: "gas_stove",
    19: "refrigerator",
    20: "airconditioner",
    21: "bath",
    22: "bath_tub",
    23: "washing_machine",
}

def svg_reader(svg_path):
    svg_list = list()
    try:
        tree = ET.parse(svg_path)
    except Exception as e:
        print("Read{} failed!".format(svg_path))
        return svg_list
    root = tree.getroot()
    for elem in root.iter():
        line = elem.attrib
        line['tag'] = elem.tag
        svg_list.append(line)
    return svg_list



def svg_writer(svg_list, svg_path):
    root = None
    current_parent = None
    
    for idx, line in enumerate(svg_list):
        tag = line["tag"]
        line.pop("tag")
        
        if idx == 0:
            root = ET.Element(tag)
            root.attrib = line
            current_parent = root
        else:
            if "}g" in tag:
                group = ET.SubElement(root, tag)
                group.attrib = line
                current_parent = group
            else:
                # Support both grouped and ungrouped SVG formats
                if current_parent is None:
                    current_parent = root
                node = ET.SubElement(current_parent, tag)
                node.attrib = line
     
    from xml.dom import minidom
    reparsed = minidom.parseString(ET.tostring(root, 'utf-8')).toprettyxml(indent="\t")
    f = open(svg_path,'w',encoding='utf-8')
    f.write(reparsed)
    f.close()           
    #prettyxml = BeautifulSoup(ET.tostring(root, 'utf-8'), "xml").prettify()
    #with open(svg_path, "w") as f:
    #    f.write(prettyxml)


def visualSVG_with_ids(parsing_list, sem_labels, ins_labels, out_path, cvt_color=False):
    """
    Visualize SVG with model predictions for both semantic and instance IDs
    """
    ind = 0
    for line in parsing_list:
        tag = line["tag"].split("svg}")[-1]
        assert tag in ['svg', 'g', 'path', 'circle', 'ellipse', 'text'], tag+" is error!!"
        
        if tag in ["path", "circle", "ellipse"]:
            # Get model predictions
            sem_label = sem_labels[ind]
            ins_label = ins_labels[ind]
            
            # Update semantic and instance IDs in SVG attributes
            # remove unnecessary classes

            if sem_label in UNNECESSARY_CLASSES.keys():
                sem_label = 35  # Set to background class
                if "semanticId" in line:
                    line.pop("semanticId")
                if "instanceId" in line:
                    line.pop("instanceId")
            else:
                line["semanticId"] = str(int(sem_label) + 1)  # Convert to 1-based index
                if sem_label in [30, 31, 32, 33, 34]:  # Stuff classes
                    line["instanceId"] = "-1"
                else:
                    line["instanceId"] = str(int(ins_label))

            
            # Set color based on semantic prediction
            color = SVG_CATEGORIES[sem_label]["color"]
            line["stroke"] = "rgb({:d},{:d},{:d})".format(color[0], color[1], color[2]) 
            line["fill"] = "none"
            line["stroke-width"] = "0.2"
            ind += 1
     
        if tag == "svg":
            viewBox = line["viewBox"]
            viewBox = viewBox.split(" ")
            line["viewBox"] = " ".join(viewBox)
            if cvt_color:
                line["style"] = "background-color: #255255255;"

    svg_writer(parsing_list, out_path)
    return out_path
        


def process_dt_with_ids(input):
    parsing_list, sem_labels, ins_labels, out_path, png_out_path, generate_png, coords = input
    
    visualSVG_with_ids(parsing_list, sem_labels, ins_labels, out_path)

    if generate_png:
        # First convert SVG to PNG
        svg2png(out_path, png_out_path, background_color="white", scale=7)
        
        # Then draw bounding boxes and IDs on the PNG
        draw_bboxes_and_ids(png_out_path, coords, sem_labels, ins_labels, scale=7)
        

def svg2png(svg_path, png_path, background_color="white", scale=7.0):
    '''
    Convert svg to png
    '''
    # cairosvg.svg2png(url=svg_path, write_to=png_path, background_color="white")
    command = "cairosvg {} -o {} -b {} -s {}".format(svg_path, png_path, background_color, scale)
    os.system(command)

def draw_bboxes_and_ids(png_path, coords, sem_labels, ins_labels, scale=7.0):
    """
    Draw bounding boxes and instance IDs on PNG image
    """
    image = Image.open(png_path)
    draw = ImageDraw.Draw(image, 'RGBA')
    
    # Get original coords shape (N, 4, 2)
    if len(coords.shape) == 3:
        coords_2d = coords
        num_points = coords.shape[0]
    elif len(coords.shape) == 2 and coords.shape[1] == 8:
        num_points = coords.shape[0]
        coords_2d = coords.reshape(-1, 4, 2)
    else:
        return
    
    # Trim labels to match actual data points
    num_actual_points = min(num_points, len(sem_labels), len(ins_labels))
    sem_labels_trimmed = sem_labels[:num_actual_points]
    ins_labels_trimmed = ins_labels[:num_actual_points]
    
    # Get unique semantic-instance pairs
    labels = np.concatenate([sem_labels_trimmed[:, None], ins_labels_trimmed[:, None]], axis=1)
    uni_labels = np.unique(labels, axis=0)
    
    for ulabel in uni_labels:
        sem, ins = ulabel
        
        # Skip background or stuff classes
        if sem >= 30 or ins < 0:
            continue
        
        # Get mask for this instance
        mask = np.logical_and(labels[:, 0] == sem, labels[:, 1] == ins)
        
        # Get coordinates for this instance
        inst_coords = coords_2d[mask].reshape(-1, 2)
        
        if len(inst_coords) == 0:
            continue
        
        # Calculate bounding box
        x1, y1 = np.min(inst_coords[:, 0], axis=0), np.min(inst_coords[:, 1], axis=0)
        x2, y2 = np.max(inst_coords[:, 0], axis=0), np.max(inst_coords[:, 1], axis=0)
        
        # Get color for this semantic class
        color = tuple(SVG_CATEGORIES[int(sem)]["color"])
        
        # Draw bounding box with transparency
        draw.rectangle([x1 * scale, y1 * scale, x2 * scale, y2 * scale],
                       fill=color + (32,), outline=color, width=2)
        
        # Draw text with instance ID only
        text = f'{ins}'
        draw.text((x1 * scale, y1 * scale), text, fill=(0, 0, 0), align='left')
    
    # Save the modified image
    image.save(png_path)



def get_path(svg_lists):
    args, widths, gids, lengths, types = [], [], [], [], []
    COMMANDS = ['Line', 'Arc','circle', 'ellipse']
    for line in svg_lists:
       
        if "d" in line.keys():
            widths.append(line["stroke-width"])
            gid = int(line["gid"]) if "gid" in line.keys() else -1
            gids.append(gid)
            path_repre = parse_path(line['d'])
            inds = [0, 1/3, 2/3, 1.0]
            arg = []
            
            # Handle degenerate paths (zero-length) by checking path length first
            try:
                path_length = path_repre.length()
                if path_length < 1e-10:
                    # Degenerate path - use start point for all samples
                    start_point = path_repre[0].start
                    for _ in inds:
                        arg.extend([start_point.real, start_point.imag])
                else:
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
                except (AttributeError, IndexError):
                    # Last resort: use origin
                    for _ in inds:
                        arg.extend([0.0, 0.0])
            
            args.append(arg)
            length = path_repre.length()
            lengths.append(length)
            path_type = path_repre[0].__class__.__name__
            types.append(COMMANDS.index(path_type))
        elif "r" in line.keys():
            widths.append(line["stroke-width"])
            gid = int(line["gid"]) if "gid" in line.keys() else -1
            gids.append(gid)
            cx = float(line['cx'])
            cy = float(line['cy'])
            r = float(line['r'])
            arg = []
            thetas = [0,math.pi/2, math.pi, 3 * math.pi/2,]
            for theta in thetas:
                x, y = cx + r * math.cos(theta), cy + r * math.sin(theta)
                arg.extend([x,y])
            args.append(arg)
            circle_len = 2 * math.pi * r
            lengths.append(circle_len)
            types.append(COMMANDS.index("circle"))
    return widths, gids, args, lengths,types
            




    


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SVG Visualization with Model Predictions')
    parser.add_argument('--res_file', type=str, default="./results/floorplancad/",
                        help='Path to the results file')
    parser.add_argument('--generate_png', action='store_true', default=False,
                        help='Generate PNG files from SVG (requires cairosvg)')
    parser.add_argument('--semantic', action='store_true', default=False,
                        help='Use semantic scores from semantic predictions instead of instance scores')
    parser.add_argument('--out_dir', type=str, default="",
                        help='Path to the output directory')
    args = parser.parse_args()
    
    instance_eval = InstanceEval(num_classes=35,
                                 ignore_label=35,gpu_num=1)
    logger = get_root_logger()
    

    res_file = Path(args.res_file) / "model_output.npy"
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(args.res_file)
    generate_png = args.generate_png
    semantic = args.semantic
    os.makedirs(out_dir, exist_ok=True)
    
    detections = np.load(res_file,allow_pickle=True)
    import tqdm
    inputs = []
    coco_res = []
    for det in tqdm.tqdm(detections):
        
        svg_path = det["filepath"].replace("_s2.svg", ".svg")
        print(svg_path)
        #if "241f" not in svg_path: continue
        assert os.path.exists(svg_path) is True,"svg_file not exists!!!"
        parsing_list = svg_reader(svg_path)
        widths, gids, args, lengths, types = get_path(parsing_list)
        widths, gids, lengths, types = np.array(widths), np.array(gids), np.array(lengths), np.array(types)
        coords = np.array(args).reshape(-1, 4,2)
        det["instances"] = []
        ins_outs = det["ins"]
        semantic_bits = det["sem"]
        if not len(ins_outs): continue
        shape = ins_outs[0]["masks"].shape[0]
        sem_out = np.full_like(np.zeros(shape), 35)  # Default to background
        ins_out = np.full_like(np.zeros(shape), -1)  # Default to no instance
        
        if semantic:
            sem_out = np.argmax(semantic_bits, axis=1).astype(np.int64)
            if len(ins_outs) > 0:
                # Process each detected instance
                for instance in ins_outs:   
                    masks, labels = instance["masks"],instance["labels"]
                    scores = instance["scores"]
                    if scores<0.1: continue
                    sem_out[masks] = labels
                    ins_out[masks] = len(det["instances"])  # Assign instance ID
                    det["instances"].append({"masks":masks, "labels":sem_out[masks][0],"scores":scores})        
        else:

            if len(ins_outs) > 0:
                # Process each detected instance
                for instance in ins_outs:
                    masks, labels = instance["masks"],instance["labels"]
                    scores = instance["scores"]
                    if scores<0.1: continue
                    sem_out[masks] = labels
                    ins_out[masks] = len(det["instances"])  # Assign instance ID
                    det["instances"].append({"masks":masks, "labels":labels,"scores":scores}) 
        
        
        coco_res.append({'filepath': det['filepath'],'instances': det['instances']})
        instance_eval.update(det["instances"],det["targets"],det["lengths"])    
        
        # Create output directory structure based on data name
        data_name = os.path.splitext(os.path.basename(svg_path))[0]  # Get filename without extension
        os.makedirs( os.path.join(out_dir, "png"), exist_ok=True)
        os.makedirs( os.path.join(out_dir, "svg"), exist_ok=True)

        
        # Create separate paths for SVG and PNG
        svg_out_path = os.path.join(out_dir, "svg", f"{data_name}_predicted.svg")
        png_out_path = os.path.join(out_dir, "png", f"{data_name}_predicted.png")
        inputs.append([parsing_list, sem_out.astype(np.int64), ins_out.astype(np.int64), svg_out_path, png_out_path, generate_png, coords])
    instance_eval.get_eval(logger)
    np.save(os.path.join(out_dir, "coco_res_val.npy"), coco_res)
    import mmcv
    mmcv.track_parallel_progress(process_dt_with_ids, inputs, 16)
    
    print(f"Generated SVG files with predicted IDs in: {out_dir}/svg")
    print("Directory structure:")
    print(f"- SVG files saved as: {out_dir}/svg/<data_name>_predicted.svg")
    if generate_png:
        print(f"- PNG files saved as: {out_dir}/png/<data_name>_predicted.png")
    else:
        print("- PNG generation skipped (use --generate_png flag to enable)")
    print("Each SVG element now has:")
    print("- semanticId: Model's predicted semantic class (1-35)")
    print("- instanceId: Model's predicted instance ID")
    print("- stroke color: Color corresponding to predicted semantic class")
    
    

    
    
    
    
    
    
