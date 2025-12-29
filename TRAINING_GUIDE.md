# SVGNet Training Process - Complete Guide

## Table of Contents
1. [Command-Line Arguments](#command-line-arguments)
2. [Configuration File Structure](#configuration-file-structure)
3. [Complete Dataflow](#complete-dataflow)
4. [Model Architecture Flow](#model-architecture-flow)
5. [Training Loop Details](#training-loop-details)

---

## Command-Line Arguments

### Required Arguments

**`config`** (positional, required)
- **Type**: `str`
- **Description**: Path to YAML configuration file
- **Example**: `configs/svg/svg_pointT.yaml`
- **Purpose**: Contains all hyperparameters, model settings, data paths, and training configuration

### Optional Arguments

**`--dist`** (flag)
- **Type**: `action="store_true"`
- **Description**: Enable distributed training across multiple GPUs
- **Default**: `False`
- **Usage**: When set, uses `DistributedDataParallel` for multi-GPU training
- **Effect**: 
  - Initializes distributed environment (`init_dist()`)
  - Uses `DistributedSampler` for data loading
  - Scales learning rate based on world size

**`--sync_bn`** (flag)
- **Type**: `action="store_true"`
- **Description**: Enable synchronized batch normalization across GPUs
- **Default**: `False`
- **Usage**: Converts all `BatchNorm` layers to `SyncBatchNorm`
- **When to use**: Helpful for distributed training with small batch sizes per GPU

**`--resume`** (optional)
- **Type**: `str`
- **Description**: Path to checkpoint file to resume training from
- **Example**: `--resume ./work_dirs/svg/svg_pointT/latest.pth`
- **Behavior**: 
  - Loads model weights AND optimizer state
  - Resumes from saved epoch number
  - Use regular checkpoints (`latest.pth`, `epoch_X.pth`) for full resume
  - `best.pth` can be used but optimizer starts fresh

**`--work_dir`** (optional)
- **Type**: `str`
- **Description**: Override working directory for saving checkpoints and logs
- **Default**: Auto-generated as `./work_dirs/{dataset_name}/{config_name}/{exp_name}`
- **Example**: `--work_dir ./my_experiment`

**`--skip_validate`** (flag)
- **Type**: `action="store_true"`
- **Description**: Skip validation during training (currently not fully implemented)
- **Default**: `False`

**`--local_rank`** (optional)
- **Type**: `int`
- **Default**: `0`
- **Description**: Local rank for distributed training (usually set automatically by `torchrun` or similar)

**`--seed`** (optional)
- **Type**: `int`
- **Default**: `2000`
- **Description**: Random seed for reproducibility
- **Effect**: Sets seed for NumPy, Python random, and PyTorch (CPU and CUDA)

**`--exp_name`** (optional)
- **Type**: `str`
- **Default**: `"default"`
- **Description**: Experiment name suffix for work directory
- **Example**: `--exp_name baseline_v1`

---

## Configuration File Structure

The YAML config file (`svg_pointT.yaml`) contains:

### Model Configuration (`model:`)
```yaml
model:
  in_channels: 9              # Input feature channels (3 coords + 6 features)
  semantic_classes: 35         # Number of semantic classes (including background)
  num_decoders: 3              # Number of decoder layers
  dropout: 0.0                 # Dropout rate
  pre_norm: False              # Pre-normalization in transformer layers
  num_heads: 8                 # Multi-head attention heads
  shared_decoder: True         # Share decoder weights across layers
  dim_feedforward: 512         # FFN dimension in transformer
  hidden_dim: 256             # Hidden dimension for queries
  num_queries: 500             # Number of object queries (DETR-style)
  gauss_scale: 1.0             # Gaussian scale for positional encoding
  normalize_pos_enc: False     # Normalize positional encoding
  scalar: 1                    # Denoising training scalar
  dn_mask_noise_scale: 0.0     # Denoising mask noise scale
  dn_label_noise_ratio: 0.2    # Denoising label noise ratio
```

### Matcher Configuration (`matcher:`)
```yaml
matcher:
  cost_class: 2.0              # Weight for classification cost in Hungarian matching
  cost_mask: 5.0               # Weight for mask cost in Hungarian matching
  cost_dice: 5.0               # Weight for dice loss cost
  num_points: -1                # Number of points for mask loss (-1 = all)
```

### Criterion Configuration (`criterion:`)
```yaml
criterion:
  num_classes: 35              # Number of classes
  eos_coef: 0.1                # End-of-sequence coefficient (for "no object")
  losses: ["labels", "masks"]  # Loss types to compute
  ignore_label: -1             # Label to ignore in loss
  class_weights: -1            # Class weights (-1 = uniform)
  num_points: -1                # Points sampled for mask loss
  contrast:                    # Contrastive learning config
    num_classes: 36
    stage: "Ua"
    num_layers: 5
    ftype: "f_out"
    dist: "l2"
    pos: "cnt"
    contrast_func: "softnn"
    sample: "label"
    temperature: 2.0
    weight: 8.0
```

### Data Configuration (`data:`)
```yaml
data:
  train:
    type: 'svg'                # Dataset type
    data_root: 'dataset/json/train'  # Path to training JSON files
    repeat: 5                  # Repeat dataset N times per epoch
    split: "train"            # Split name
    data_norm: "mean"         # Normalization: "mean" or "min"
    aug:                      # Augmentation config
      aug_prob: 0.5           # Probability of applying augmentation
      hflip: True             # Horizontal flip
      vflip: True             # Vertical flip
      rotate:                 # Rotation augmentation
        enable: False
        angle: [-180,180]
      rotate2: True           # Random 90-degree rotation
      scale:                  # Scale augmentation
        enable: True
        ratio: [0.5,1.5]
      shift:                  # Translation augmentation
        enable: True
        scale: [-0.5,0.5]
      cutmix:                 # CutMix augmentation
        enable: True
        queueK: 32            # Queue size for instance mixing
        relative_shift: [-0.5,0.5]
  test:
    type: 'svg'
    data_root: 'dataset/json/test'
    repeat: 1
    split: "test"
    data_norm: "mean"
    aug: False                # No augmentation for validation
```

### DataLoader Configuration (`dataloader:`)
```yaml
dataloader:
  train:
    batch_size: 1             # Batch size per GPU
    num_workers: 2            # Data loading workers
  test:
    batch_size: 1
    num_workers: 1
```

### Optimizer Configuration (`optimizer:`)
```yaml
optimizer:
  type: 'AdamW'               # Optimizer type: 'AdamW' or 'SGD'
  lr: 0.0001                  # Base learning rate (for batch_size=16)
  weight_decay: 0.0001        # Weight decay
  weight_decay_embed: 0.0     # Weight decay for embedding layers
  decoder_multiplier: 1.0     # LR multiplier for decoder layers
  clip_gradients_enabled: True
  clip_gradients_type: "full_model"
  clip_gradients_norm_type: 2.0
  clip_gradients_value: 0.01   # Gradient clipping value
```

### Training Configuration
```yaml
fp16: False                   # Mixed precision training
epochs: 50                    # Total training epochs
step_epoch: 30                # Epoch to start cosine decay
save_freq: 10                 # Save checkpoint every N epochs
pretrain: ''                  # Path to pretrained weights (model only, no optimizer)
work_dir: ''                  # Override work directory
```

---

## Complete Dataflow

### 1. Initialization Phase

```
main() function starts
    ↓
Parse command-line arguments (get_args())
    ↓
Load YAML config file → Convert to Munch dict
    ↓
Initialize distributed training (if --dist)
    ↓
Set random seed (args.seed + rank)
    ↓
Create work directory structure:
    ./work_dirs/{dataset}/{config_name}/{exp_name}/
    ↓
Initialize logger and TensorBoard writer
    ↓
Copy config file to work_dir
```

### 2. Model Setup Phase

```
Create HungarianMatcher (for bipartite matching)
    ↓
Create SetCriterion (loss computation)
    ↓
Initialize SVGNet model:
    - Backbone: PointTransformer (or PointNet2)
    - Decoder: Transformer decoder with queries
    ↓
Move model to CUDA
    ↓
Convert to SyncBatchNorm (if --sync_bn)
    ↓
Count parameters (total and trainable)
    ↓
Wrap with DistributedDataParallel (if --dist)
    ↓
Create GradScaler for mixed precision (if fp16)
```

### 3. Data Loading Phase

```
Build Training Dataset:
    SVGDataset(
        data_root='dataset/json/train',
        split='train',
        data_norm='mean',
        aug={...augmentation config...},
        repeat=5
    )
    ↓
    Loads all *.json files from data_root
    ↓
    Each JSON contains:
        - args: SVG command arguments (8 values per point)
        - lengths: Curve lengths
        - commands: SVG command types (one-hot encoded)
        - semanticIds: Semantic class labels
        - instanceIds: Instance labels
    ↓
Build Validation Dataset:
    SVGDataset(
        data_root='dataset/json/test',
        split='test',
        aug=False,
        repeat=1
    )
    ↓
Create DataLoaders:
    - train_loader: Shuffled, with augmentation
    - val_loader: Sequential, no augmentation
    - Uses DistributedSampler if dist=True
```

### 4. Data Processing Flow (Per Sample)

```
JSON File Loaded
    ↓
Extract coordinates:
    - args reshaped to (N, 8) / 140 (normalize)
    - coord_x = mean(args[:, 0::2])  # Average x coordinates
    - coord_y = mean(args[:, 1::2])  # Average y coordinates
    - coord_z = zeros (2D data)
    - Pad to min_points=2048 if needed
    ↓
Extract features (6 channels):
    - feat[0] = arctan(y/x) / π  # Angle feature
    - feat[1] = lengths clipped to [0,140] / 140  # Normalized length
    - feat[2:6] = one-hot encoding of command type (4 types)
    ↓
Extract labels:
    - semanticIds: Class labels (0-35, 35=background)
    - instanceIds: Instance labels (-1 for stuff, unique IDs for things)
    ↓
Training Augmentation (if split='train'):
    - Horizontal flip (50% prob)
    - Vertical flip (50% prob)
    - Random rotation (if enabled)
    - Random 90° rotation (50% prob)
    - Random translation (50% prob)
    - Random scaling (50% prob)
    - CutMix: Mix instances from queue (50% prob)
    ↓
Normalize coordinates:
    - Subtract mean (if data_norm='mean')
    - Or subtract min (if data_norm='min')
    ↓
Shuffle points (training only)
    ↓
Collate into batch:
    - Concatenate all samples
    - Create offset array for batch boundaries
    - Return: (coords, feats, labels, offsets, lengths)
```

### 5. Optimizer Setup Phase

```
Calculate scaled learning rate:
    default_lr = cfg.optimizer.lr  # Base LR for batch_size=16
    world_size = number of GPUs
    total_batch_size = batch_size * world_size
    scaled_lr = default_lr * (total_batch_size / 16)
    ↓
Build optimizer (build_new_optimizer):
    - Separate parameter groups:
      * Decoder layers: lr * decoder_multiplier
      * Normalization layers: weight_decay_norm
      * Embedding layers: weight_decay_embed (usually 0)
      * Position embeddings: weight_decay = 0
    - Optimizer type: AdamW or SGD
    - Gradient clipping enabled (if configured)
```

### 6. Checkpoint Loading Phase

```
If --resume:
    load_checkpoint(resume_path):
        - Load checkpoint file
        - Extract model weights (state_dict["net"])
        - Load into model (handle size mismatches)
        - Load optimizer state (if present)
        - Return epoch number + 1
    ↓
Else if cfg.pretrain:
    load_checkpoint(pretrain_path):
        - Load model weights only
        - No optimizer state
        - Start from epoch 1
```

---

## Model Architecture Flow

### Forward Pass (Training)

```
Input Batch:
    coords: (N, 3) - Point coordinates
    feats: (N, 9) - Point features (3 coords + 6 features)
    semantic_labels: (N, 2) - [semantic_id, instance_id]
    offsets: (B,) - Batch offsets
    lengths: (N,) - Curve lengths
    ↓
SVGNet.forward():
    ↓
Prepare targets:
    - Group points by (semantic_id, instance_id)
    - Create mask targets for each instance
    - Format for Hungarian matcher
    ↓
Backbone (PointTransformer):
    Input: coords, feats, offsets
    ↓
    Encoder (Downsampling):
        p0, x0, o0 → enc1 → p1, x1, o1  # 32 dims
        p1, x1, o1 → enc2 → p2, x2, o2  # 64 dims
        p2, x2, o2 → enc3 → p3, x3, o3  # 128 dims
        p3, x3, o3 → enc4 → p4, x4, o4  # 256 dims
        p4, x4, o4 → enc5 → p5, x5, o5  # 512 dims
    ↓
    Decoder (Upsampling):
        p5, x5, o5 → dec5 → x5_up
        p4, x4, o4 + x5_up → dec4 → x4_up
        p3, x3, o3 + x4_up → dec3 → x3_up
        p2, x2, o2 + x3_up → dec2 → x2_up
        p1, x1, o1 + x2_up → dec1 → x1_up
    ↓
    Output: Multi-scale features [p1-p5, x1_up]
    ↓
Decoder (Transformer):
    Input: Multi-scale features from backbone
    ↓
    Initialize queries:
        - query_feat: (500, 256) learnable embeddings
        - query_pos: (500, 256) learnable positional embeddings
    ↓
    For each decoder layer (3 layers):
        For each scale (p1-p4, bottom-up):
            - Cross-attention: queries attend to point features
            - Self-attention: queries attend to each other
            - FFN: Feed-forward network
            - Predict: class logits + mask logits
    ↓
    Final predictions:
        - pred_logits: (B, 500, 36) - Class predictions (35 classes + no-object)
        - pred_masks: (B, 500, N) - Mask predictions for each query
        - aux_outputs: Intermediate predictions for auxiliary loss
    ↓
Loss Computation (SetCriterion):
    - Hungarian matching: Match predictions to ground truth
    - Classification loss: Cross-entropy on matched pairs
    - Mask loss: Binary cross-entropy + Dice loss
    - Contrastive loss: (if enabled)
    ↓
Return: (model_outputs, total_loss, loss_dict)
```

### Forward Pass (Validation/Inference)

```
Same as training until decoder output
    ↓
Semantic Inference:
    - Apply softmax to pred_logits → mask_cls
    - Apply sigmoid to pred_masks → mask_pred
    - semseg = einsum("qc,qg->gc", mask_cls, mask_pred)
    - Result: (N, 35) semantic scores per point
    ↓
Instance Inference:
    - Filter queries by confidence
    - Apply NMS-like filtering
    - Extract instance masks
    ↓
Return: {
    "semantic_scores": (N, 35),
    "semantic_labels": (N,),
    "instances": instance predictions,
    "targets": ground truth targets,
    "lengths": curve lengths
}
```

---

## Training Loop Details

### Training Epoch (`train()` function)

```
For each epoch:
    model.train()
    Initialize meters (iter_time, data_time, loss meters)
    ↓
    Set epoch for DistributedSampler (if dist)
    ↓
    For each batch:
        ↓
        Data Loading:
            - Measure data loading time
            - Get batch: (coords, feats, labels, offsets, lengths)
        ↓
        Learning Rate Scheduling:
            - If scheduler=None: Use cosine annealing
            - cosine_lr_after_step():
                * Before step_epoch: constant LR
                * After step_epoch: cosine decay
        ↓
        Forward Pass:
            - Enable autocast (if fp16)
            - model(batch) → (outputs, loss, log_vars)
            - Disable autocast
        ↓
        Backward Pass:
            - optimizer.zero_grad()
            - scaler.scale(loss).backward()
            - scaler.step(optimizer)
            - scaler.update()
            - Check for parameters with None gradients (debug)
        ↓
        Logging:
            - Update loss meters
            - Calculate remaining time
            - Log every 50 iterations:
                * Epoch, iteration, LR, ETA
                * Memory usage, data time, iter time
                * All loss components
        ↓
    End of epoch:
        - Log average losses to TensorBoard
        - Save checkpoint (epoch_X.pth, latest.pth)
```

### Validation Epoch (`validate()` function)

```
model.eval()
torch.no_grad()
Initialize evaluators:
    - PointWiseEval: Semantic segmentation metrics
    - InstanceEval: Panoptic segmentation metrics
    ↓
For each batch:
    Forward pass (no gradient):
        - model(batch) → (res, loss, log_vars)
        - Extract predictions:
            * sem_preds = argmax(semantic_scores)
            * instances = instance predictions
    ↓
    Update evaluators:
        - sem_point_eval.update(sem_preds, sem_gts)
        - instance_eval.update(instances, targets, lengths)
    ↓
Compute metrics:
    - Semantic: mIoU, Accuracy
    - Panoptic: sPQ (segmentation Panoptic Quality)
                 sRQ (Recognition Quality)
                 sSQ (Segmentation Quality)
    ↓
Log to TensorBoard:
    - Validation losses
    - mIoU, Accuracy
    - sPQ, sRQ, sSQ
    ↓
Save best model:
    - If sPQ > best_metric:
        * Update best_metric
        * Save best.pth (model weights only, no optimizer)
```

### Checkpoint Saving

```
Regular Checkpoints (every save_freq epochs):
    checkpoint = {
        "net": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }
    Save as: epoch_{epoch}.pth
    Also save as: latest.pth (overwrites previous)
    ↓
Best Checkpoint (when validation improves):
    checkpoint = {
        "net": model.state_dict(),
        "epoch": epoch
    }
    Save as: best.pth (NO optimizer state)
    ↓
Checkpoint Cleanup:
    - Keep checkpoints at power-of-2 epochs
    - Keep checkpoints at save_freq intervals
    - Delete others to save space
```

---

## Key Design Decisions

1. **Point Cloud Representation**: SVG curves converted to point clouds with geometric features
2. **DETR-style Architecture**: Uses object queries for instance segmentation
3. **Multi-scale Features**: Backbone extracts features at multiple resolutions
4. **Hungarian Matching**: Bipartite matching for training stability
5. **Auxiliary Losses**: Intermediate decoder outputs contribute to loss
6. **CutMix Augmentation**: Mixes instances from previous batches for data augmentation
7. **Learning Rate Scaling**: Automatically scales LR based on total batch size
8. **Mixed Precision**: Optional FP16 training for memory efficiency

---

## Common Training Scenarios

### Starting Fresh Training
```bash
python tools/train.py configs/svg/svg_pointT.yaml --exp_name my_experiment
```

### Resuming Training
```bash
python tools/train.py configs/svg/svg_pointT.yaml \
    --resume ./work_dirs/svg/svg_pointT/my_experiment/latest.pth
```

### Distributed Training (4 GPUs)
```bash
torchrun --nproc_per_node=4 tools/train.py \
    configs/svg/svg_pointT.yaml \
    --dist --exp_name distributed_run
```

### Training with Pretrained Weights
```yaml
# In config file:
pretrain: './pretrained/checkpoint.pth'
```
```bash
python tools/train.py configs/svg/svg_pointT.yaml
```

---

## Output Files Structure

```
work_dirs/{dataset}/{config}/{exp_name}/
├── {timestamp}.log              # Training log file
├── svg_pointT.yaml              # Copied config file
├── epoch_10.pth                 # Regular checkpoint (epoch 10)
├── epoch_20.pth                 # Regular checkpoint (epoch 20)
├── latest.pth                   # Latest checkpoint (always updated)
├── best.pth                     # Best validation checkpoint
└── events.out.tfevents.*        # TensorBoard logs
```

---

## Monitoring Training

1. **Log File**: Check `{timestamp}.log` for detailed training logs
2. **TensorBoard**: `tensorboard --logdir work_dirs/`
3. **Metrics to Watch**:
   - Training losses (should decrease)
   - Validation mIoU (should increase)
   - Validation sPQ (panoptic quality, should increase)
   - Learning rate (follows cosine schedule)

---

This completes the comprehensive guide to the SVGNet training process!









