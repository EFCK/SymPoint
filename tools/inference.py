
import argparse
import yaml
from munch import Munch
import glob,tqdm
import os.path as osp
import numpy as np
import os

import torch

from svgnet.model.svgnet import SVGNet as svgnet
from svgnet.data.svg import SVGDataset
from svgnet.util  import get_root_logger, load_checkpoint
from svgnet.evaluation import PointWiseEval,InstanceEval

import time

def get_args():
    parser = argparse.ArgumentParser("svgnet")
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("checkpoint", type=str, help="path to checkpoint")
    parser.add_argument("--datadir", type=str, help="the path to dataset")
    parser.add_argument("--out", type=str, help="the path to save results")
    args = parser.parse_args()
    return args


def has_ground_truth(data_json):
    """
    Check if JSON file has ground truth labels (semantic IDs and instance IDs).
    Returns True if file has non-default labels.
    """
    import json
    data = json.load(open(data_json))
    semantic_ids = data.get("semanticIds", [])
    instance_ids = data.get("instanceIds", [])
    
    # Check if all semantic IDs are background (LABEL_NUM = 35)
    if len(semantic_ids) == 0:
        return False
    if all(sid == 35 for sid in semantic_ids):
        return False
    
    # Check if any instance IDs are valid (>= 0)
    if any(iid >= 0 for iid in instance_ids):
        return True
    
    # If we have non-background semantic IDs, we have some ground truth
    return any(sid < 35 for sid in semantic_ids)


def main():
    args = get_args()
    cfg_txt = open(args.config, "r").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    logger = get_root_logger()

    model = svgnet(cfg.model).cuda()
   
    logger.info(f"Load state dict from {args.checkpoint}")
    load_checkpoint(args.checkpoint, logger, model)
    data_list = glob.glob(osp.join(args.datadir,"*.json"))
    logger.info(f"Load dataset: {len(data_list)} svg")
    
    # Only create evaluators if we have ground truth data
    has_gt_data = any(has_ground_truth(f) for f in data_list)
    
    if has_gt_data:
        sem_point_eval = PointWiseEval(num_classes=cfg.model.semantic_classes,ignore_label=cfg.model.ignore_labels,gpu_num=1)
        instance_eval = InstanceEval(
            num_classes=cfg.model.semantic_classes,
            ignore_label=cfg.model.ignore_labels,
            gpu_num=1,
            min_obj_score=cfg.model.get('test_object_score', 0.1),
            iou_threshold=cfg.model.get('eval_iou_threshold', 0.5)
        )
    else:
        logger.info("No ground truth labels found - skipping evaluation metrics")
    
    save_dicts = []
    total_times = []
    eval_count = 0
    
    with torch.no_grad():
        model.eval()
        for svg_file in tqdm.tqdm(data_list):
            coords, feats, labels,lengths = SVGDataset.load(svg_file,idx=1)
            coords -= np.mean(coords, 0)
            offset = [coords.shape[0]]
            offset = torch.IntTensor(offset)
            coords,feats,labels = torch.FloatTensor(coords), torch.FloatTensor(feats), torch.LongTensor(labels)
            batch = (coords,feats,labels,offset, torch.FloatTensor(lengths))
            
            torch.cuda.empty_cache()
            
            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                t1 = time.time()
                res = model(batch,return_loss=False)
                t2 = time.time()
                total_times.append(t2-t1)
                
                # Only evaluate if we have ground truth
                sem_preds = torch.argmax(res["semantic_scores"],dim=1).cpu().numpy()
                if has_gt_data and has_ground_truth(svg_file):
                    sem_gts = res["semantic_labels"].cpu().numpy()
                    sem_point_eval.update(sem_preds, sem_gts)
                    
                    instance_eval.update(
                        res["instances"],
                        res["targets"],
                        res["lengths"],
                    )
                    eval_count += 1
                
                save_dicts.append({
                    "filepath": svg_file.replace("dataset/json/", "dataset/svg/").replace(".json",".svg"),
                    "sem": res["semantic_scores"].cpu().numpy(),
                    "ins": res["instances"],
                    "targets":res["targets"],
                    "lengths":res["lengths"],
                })
                    
                    
    os.makedirs(args.out,exist_ok=True)
    np.save(osp.join(args.out, 'model_output.npy'), save_dicts)
    
    if has_gt_data and eval_count > 0:
        logger.info("Evaluate semantic segmentation")
        sem_point_eval.get_eval(logger)
        logger.info("Evaluate panoptic segmentation")
        instance_eval.get_eval(logger)
    else:
        logger.info(f"Saved predictions for {len(save_dicts)} files (no ground truth for evaluation)")
 
if __name__ == "__main__":
    main()       
        