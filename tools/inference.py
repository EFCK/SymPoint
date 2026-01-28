import argparse
import yaml
from munch import Munch
import glob, tqdm
import os.path as osp
import numpy as np
import os

import torch

from svgnet.model.svgnet import SVGNet as svgnet
from svgnet.data.svg import SVGDataset
from svgnet.util import get_root_logger, load_checkpoint
from svgnet.evaluation import PointWiseEval, InstanceEval

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
    data_list = glob.glob(osp.join(args.datadir, "*.json"))
    logger.info(f"Load dataset: {len(data_list)} svg")

    # Only create evaluators if we have ground truth data
    has_gt_data = any(has_ground_truth(f) for f in data_list)

    if has_gt_data:
        sem_point_eval = PointWiseEval(
            num_classes=cfg.model.semantic_classes,
            ignore_label=cfg.ignore_labels,
            gpu_num=1,
        )
        instance_eval = InstanceEval(
            num_classes=cfg.model.semantic_classes,
            ignore_label=cfg.ignore_labels,
            gpu_num=1,
            min_obj_score=cfg.model.get("test_object_score", 0.1),
            iou_threshold=cfg.model.get("eval_iou_threshold", 0.5),
        )
    else:
        logger.info("No ground truth labels found - skipping evaluation metrics")

    save_dicts = []
    total_times = []
    eval_count = 0
    skipped_files = []

    with torch.no_grad():
        model.eval()
        for svg_file in tqdm.tqdm(data_list):
            try:
                coords, feats, labels, lengths = SVGDataset.load(svg_file, idx=1)
                coords -= np.mean(coords, 0)
                offset = [coords.shape[0]]
                offset = torch.IntTensor(offset)
                coords, feats, labels = (
                    torch.FloatTensor(coords),
                    torch.FloatTensor(feats),
                    torch.LongTensor(labels),
                )
                batch = (coords, feats, labels, offset, torch.FloatTensor(lengths))

                torch.cuda.empty_cache()

                with torch.cuda.amp.autocast(enabled=cfg.fp16):
                    t1 = time.time()
                    res = model(batch, return_loss=False)
                    t2 = time.time()
                    total_times.append(t2 - t1)

                    # Only evaluate if we have ground truth
                    sem_preds = (
                        torch.argmax(res["semantic_scores"], dim=1).cpu().numpy()
                    )
                    if has_gt_data and has_ground_truth(svg_file):
                        sem_gts = res["semantic_labels"].cpu().numpy()
                        sem_point_eval.update(sem_preds, sem_gts)

                        instance_eval.update(
                            res["instances"],
                            res["targets"],
                            res["lengths"],
                        )
                        eval_count += 1

                    save_dicts.append(
                        {
                            "filepath": svg_file.replace(
                                "dataset/json/", "dataset/svg/"
                            ).replace(".json", ".svg"),
                            "sem": res["semantic_scores"].cpu().numpy(),
                            "ins": res["instances"],
                            "targets": res["targets"],
                            "lengths": res["lengths"],
                        }
                    )

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                    skipped_files.append(svg_file)
                    logger.warning(
                        f"[OOM] Skipped: {osp.basename(svg_file)} - CUDA out of memory"
                    )
                    continue
                else:
                    raise e

    # Log skipped files summary
    if skipped_files:
        logger.warning(f"Skipped {len(skipped_files)} files due to OOM:")
        for f in skipped_files:
            logger.warning(f"  - {osp.basename(f)}")

    os.makedirs(args.out, exist_ok=True)
    np.save(osp.join(args.out, "model_output.npy"), save_dicts)

    if has_gt_data and eval_count > 0:
        logger.info("Evaluate semantic segmentation")
        miou, pACC = sem_point_eval.get_eval(logger)
        logger.info("Evaluate panoptic segmentation")
        sPQ, sRQ, sSQ = instance_eval.get_eval(logger)

        # Calculate detailed metrics for saving
        # Semantic metrics
        conf_matrix = sem_point_eval._conf_matrix
        class_names = sem_point_eval._class_names
        num_classes = sem_point_eval._num_classes
        ignore_label = sem_point_eval.ignore_label

        class_ids = np.arange(num_classes)
        valid_mask = ~np.isin(class_ids, ignore_label)
        tp = conf_matrix.diagonal()[:-1].astype(np.float64)
        pos_gt = np.sum(conf_matrix[:-1, :-1], axis=0).astype(np.float64)
        pos_pred = np.sum(conf_matrix[:-1, :-1], axis=1).astype(np.float64)
        union = pos_gt + pos_pred - tp
        iou = np.full(num_classes, np.nan, dtype=np.float64)
        iou_valid = np.logical_and(pos_gt > 0, np.logical_and(valid_mask, union > 0))
        iou[iou_valid] = tp[iou_valid] / (union[iou_valid] + 1e-8)

        # Frequency weighted IoU
        class_weights = np.zeros_like(pos_gt, dtype=np.float64)
        denom_weights = np.sum(pos_gt[valid_mask]) + 1e-8
        class_weights[valid_mask] = pos_gt[valid_mask] / denom_weights
        fiou = 100 * np.sum(iou[iou_valid] * class_weights[iou_valid])

        # Panoptic metrics per class
        inst_eval = instance_eval
        RQ = inst_eval.tp_classes / (
            inst_eval.tp_classes
            + 0.5 * inst_eval.fp_classes
            + 0.5 * inst_eval.fn_classes
            + 1e-6
        )
        SQ = inst_eval.tp_classes_values / (inst_eval.tp_classes + 1e-6)
        PQ = RQ * SQ

        # Thing/Stuff metrics
        thing_class = inst_eval.thing_class
        stuff_class = inst_eval.stuff_class
        thing_RQ = sum(inst_eval.tp_classes[thing_class]) / (
            sum(inst_eval.tp_classes[thing_class])
            + 0.5 * sum(inst_eval.fp_classes[thing_class])
            + 0.5 * sum(inst_eval.fn_classes[thing_class])
            + 1e-6
        )
        thing_SQ = sum(inst_eval.tp_classes_values[thing_class]) / (
            sum(inst_eval.tp_classes[thing_class]) + 1e-6
        )
        thing_PQ = thing_RQ * thing_SQ
        stuff_RQ = sum(inst_eval.tp_classes[stuff_class]) / (
            sum(inst_eval.tp_classes[stuff_class])
            + 0.5 * sum(inst_eval.fp_classes[stuff_class])
            + 0.5 * sum(inst_eval.fn_classes[stuff_class])
            + 1e-6
        )
        stuff_SQ = sum(inst_eval.tp_classes_values[stuff_class]) / (
            sum(inst_eval.tp_classes[stuff_class]) + 1e-6
        )
        stuff_PQ = stuff_RQ * stuff_SQ

        # Counts
        total_tp = int(sum(inst_eval.tp_classes))
        total_fp = int(sum(inst_eval.fp_classes))
        total_fn = int(sum(inst_eval.fn_classes))

        # Timing statistics
        avg_time = np.mean(total_times) if total_times else 0
        total_time = np.sum(total_times) if total_times else 0

        # Save evaluation results to log file
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        results_file = osp.join(args.out, "inference_results.log")
        with open(results_file, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("INFERENCE EVALUATION RESULTS\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Config: {args.config}\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Dataset: {args.datadir}\n")
            f.write(f"Total files: {len(data_list)}\n")
            f.write(f"Evaluated files: {eval_count}\n")
            f.write(f"Skipped files (OOM): {len(skipped_files)}\n\n")

            f.write("-" * 60 + "\n")
            f.write("TIMING\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total inference time: {total_time:.2f}s\n")
            f.write(f"Average time per file: {avg_time:.4f}s\n\n")

            f.write("-" * 60 + "\n")
            f.write("SEMANTIC SEGMENTATION\n")
            f.write("-" * 60 + "\n")
            f.write(f"mIoU:  {miou:.3f}\n")
            f.write(f"fwIoU: {fiou:.3f}\n")
            f.write(f"pACC:  {pACC:.3f}\n\n")

            f.write("Per-class IoU:\n")
            for i, name in enumerate(class_names):
                if i in ignore_label:
                    continue
                if not np.isnan(iou[i]):
                    f.write(f"  {name:20s}: {iou[i] * 100:.3f}\n")
            f.write("\n")

            f.write("-" * 60 + "\n")
            f.write("PANOPTIC SEGMENTATION\n")
            f.write("-" * 60 + "\n")
            f.write(f"PQ: {sPQ:.3f}  |  RQ: {sRQ:.3f}  |  SQ: {sSQ:.3f}\n\n")

            f.write("Thing classes:\n")
            f.write(
                f"  PQ: {thing_PQ * 100:.3f}  |  RQ: {thing_RQ * 100:.3f}  |  SQ: {thing_SQ * 100:.3f}\n\n"
            )

            f.write("Stuff classes:\n")
            f.write(
                f"  PQ: {stuff_PQ * 100:.3f}  |  RQ: {stuff_RQ * 100:.3f}  |  SQ: {stuff_SQ * 100:.3f}\n\n"
            )

            f.write("Per-class PQ:\n")
            for i, name in enumerate(class_names):
                if i in ignore_label:
                    continue
                f.write(
                    f"  {name:20s}: PQ={PQ[i] * 100:6.3f}  RQ={RQ[i] * 100:6.3f}  SQ={SQ[i] * 100:6.3f}  (TP={int(inst_eval.tp_classes[i])}, FP={int(inst_eval.fp_classes[i])}, FN={int(inst_eval.fn_classes[i])})\n"
                )
            f.write("\n")

            f.write("-" * 60 + "\n")
            f.write("TOTAL COUNTS\n")
            f.write("-" * 60 + "\n")
            f.write(f"TP: {total_tp}\n")
            f.write(f"FP: {total_fp}\n")
            f.write(f"FN: {total_fn}\n")

            if skipped_files:
                f.write("\n")
                f.write("-" * 60 + "\n")
                f.write("SKIPPED FILES (OOM)\n")
                f.write("-" * 60 + "\n")
                for skipped in skipped_files:
                    f.write(f"  {osp.basename(skipped)}\n")

        logger.info(f"Saved evaluation results to {results_file}")
    else:
        logger.info(
            f"Saved predictions for {len(save_dicts)} files (no ground truth for evaluation)"
        )


if __name__ == "__main__":
    main()
