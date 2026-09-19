# Jersey Number Pipeline PlusPlus

This repository contains code for a **bag-of-tricks extension to a jersey number recognition pipeline for sports video**, building on **[A General Framework for Jersey Number Recognition in Sports Video](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Koshkina_A_General_Framework_for_Jersey_Number_Recognition_in_Sports_Video_CVPRW_2024_paper.pdf)** by Maria Koshkina and James H. Elder.

The original system recognizes jersey numbers at the tracklet level by filtering distractors, classifying legibility, estimating player pose, cropping the torso, running scene-text recognition, and consolidating predictions across frames. Our project keeps that overall structure and modifies several stages to improve robustness, reduce unnecessary computation, and make tracklet-level prediction more reliable.

![Pipeline](docs/soccer_pipeline.png)

## Pipeline

```text
Player Tracklets
      |
      v
Re-ID / Main-Subject Filtering
      |
      v
Legibility Classification
      |
      v
ViTPose Pose Estimation
      |
      v
Pose-Guided Torso Cropping
      |
      v
PARSeq Scene-Text Recognition
      |
      v
Tracklet Prediction Consolidation
      |
      v
Final Jersey Number
```

## What We Optimized

### Stage 1: More robust main-subject filtering

The first stage uses re-identification features to identify the main player in each tracklet and reject outliers based on their distance from the tracklet centroid. A failure mode we observed was that **background noise could distort feature generation**, making distractor frames appear closer to the centroid than they should.

We added **Non-Local Means denoising** before feature extraction. Compared with local smoothing, Non-Local Means was chosen to reduce background noise while better preserving image edges and player structure.

```python
cv2.fastNlMeansDenoisingColored(...)
```

We also experimented with a training-time augmentation that zeros pixels in a consistent noisy pattern so the feature extractor is encouraged to rely less on background information and more on player-specific visual features.

### Stage 2: Legibility-classifier augmentation

The legibility classifier determines which player frames contain a usable jersey region before the more expensive downstream stages.

We enhanced its training data with:

- **RandAugment**, applying randomized rotations, color shifts, and other image transformations to simulate variation in camera position, lighting, weather, occlusion, and noise.
- **MixUp**, blending pairs of training images and labels to increase data diversity and encourage smoother decision boundaries.

The project presentation reports that these optimizations improved robustness and training-data diversity while keeping overall accuracy consistent with the heuristic baseline.

### Stage 3: ViTPose inference and temporal stability

The pose stage identifies shoulders and hips so the pipeline can crop the torso region where jersey numbers are expected to appear. The model used is **ViTPose**, trained on MS COCO.

We explored three optimizations for this stage.

#### Reduced-precision inference

Pose inference was moved from FP32 to **FP16 where CUDA is available** using mixed-precision execution.

```python
with torch.autocast("cuda", dtype=torch.float16):
    ...
```

Using fewer bits per value reduces memory pressure and allows more data to be processed concurrently, with the trade-off that reduced precision can slightly affect numerical accuracy.

#### Temporal smoothing of keypoints

Keypoint predictions can jitter from frame to frame because of occlusion, motion, and pose variation. We added **exponential temporal smoothing** using the current and previous keypoint predictions:

```text
smoothed_keypoint =
    alpha * current_keypoint
    + (1 - alpha) * previous_keypoint
```

This makes shoulder and hip trajectories more stable before torso cropping.

#### Flash-attention-style pose optimization

We also explored replacing the standard attention computation in the ViTPose transformer with a **FlashAttention-style scaled dot-product attention path**. The motivation was to reduce attention-memory traffic by computing attention in tiled GPU-friendly blocks rather than materializing the full attention matrix.

This optimization has an important trade-off: efficient FlashAttention kernels are hardware-sensitive and require low-level implementation details to realize their full speedup.

### Stage 4: Task-specific PARSeq decoding

The scene-text recognition stage uses **PARSeq (Permuted Autoregressive Sequence Models)** to recognize jersey numbers from torso crops.

PARSeq normally trains and evaluates across multiple permutations of token positions. For jersey numbers, however, the useful reading direction is overwhelmingly **left-to-right and approximately horizontal**.

We therefore investigated restricting decoding to position permutations that correspond to near-horizontal left-to-right reading. This reduces unnecessary permutation work and memory usage for this task, with the acknowledged trade-off that aggressively reducing permutations can slightly reduce recognition accuracy.

A natural extension is to make the allowed reading-angle threshold learnable so the model can choose the amount of permutation flexibility appropriate for the data.

### Stage 5: Learned tracklet prediction consolidation

Individual frames within a player tracklet can disagree because of:

- partial jersey visibility,
- occlusion,
- pose variation,
- low-confidence STR predictions,
- ambiguity between one- and two-digit jersey numbers.

The baseline consolidation methods use confidence-weighted voting or probabilistic aggregation across frames.

We added a **bidirectional LSTM prediction consolidator** that learns to combine a sequence of frame-level jersey predictions and their confidence scores.

```text
Frame-level jersey predictions
        + confidence scores
                |
                v
         Embedding / Features
                |
                v
       Bidirectional LSTM
                |
                v
           Mean Pooling
                |
                v
       Fully Connected Layer
                |
                v
        100-way Prediction
```

This makes the final tracklet prediction a learned temporal aggregation problem rather than relying only on a fixed hand-designed rule.

## System-Level Optimizations

In addition to the stage-specific experiments, the code includes several integration optimizations needed to run the modified multi-model pipeline:

- dynamic CPU/GPU device selection for pose inference,
- FP16 autocast on CUDA,
- temporal keypoint state across frames,
- JSON-safe keypoint serialization,
- configurable pipeline stages,
- runtime instrumentation for end-to-end execution,
- compatibility updates for the bundled PARSeq code,
- integration of the learned consolidation stage into the SoccerNet pipeline.

## Results

The project evaluated the complete pipeline on the SoccerNet jersey-number recognition task. The presentation reports a leaderboard result of **87.45 accuracy** for the submitted system.

The project was designed as a collection of complementary modifications rather than a single model replacement, so individual stages have different trade-offs:

- denoising and augmentation target robustness,
- reduced precision and task-specific decoding target efficiency,
- temporal smoothing targets pose stability,
- learned consolidation targets consistency across frames.

## Requirements

- PyTorch
- OpenCV

The complete pipeline also relies on the external research implementations listed below.

## Setup

Clone the repository and run:

```bash
python3 setup.py
```

The setup process configures the external components used by the pipeline. They can also be installed manually.

### Centroid-ReID

Repository:

[https://github.com/mikwieczorek/centroids-reid](https://github.com/mikwieczorek/centroids-reid)

Download the [Centroid-ReID model weights](https://drive.google.com/file/d/1bSUNpvMfJkvCFOu-TK-o7iGY1p-9BxmO/view?usp=sharing) and place them under:

```text
reid/centroids-reid/models
```

### ViTPose

Repository:

[https://github.com/ViTAE-Transformer/ViTPose](https://github.com/ViTAE-Transformer/ViTPose)

Download the [ViTPose model weights](https://1drv.ms/u/s!AimBgYV7JjTlgShLMI-kkmvNfF_h?e=dEhGHe) and place them under:

```text
pose/ViTPose/checkpoints/
```

### PARSeq

The repository contains the PARSeq version used by the jersey-number recognition pipeline.

Original repository:

[https://github.com/baudm/parseq](https://github.com/baudm/parseq)

Model weights:

- [Original model weights](https://drive.google.com/file/d/1AK_GnM6pIYyfIf3tBYSKIyR3Fa3Z46Cx/view?usp=sharing)
- [Hockey fine-tuned](https://drive.google.com/file/d/1FyM31xvSXFRusN0sZH0EWXoHwDfB9WIE/view?usp=sharing)
- [SoccerNet fine-tuned](https://drive.google.com/file/d/1uRln22tlhneVt3P6MePmVxBWSLMsL3bm/view?usp=sharing)

### SAM

Repository:

[https://github.com/davda54/sam](https://github.com/davda54/sam)

## Data

### SoccerNet Jersey Number Recognition

Dataset:

[https://github.com/SoccerNet/sn-jersey](https://github.com/SoccerNet/sn-jersey)

Download and save it under the `data` directory.

Additional resources:

- [Weakly-labelled player images used to train the legibility classifier](https://drive.google.com/file/d/1CmJfUmS_ZudgEiCT14b2CbyMA3nEO_uy/view?usp=sharing)
- [Weakly-labelled jersey-number crops used to fine-tune STR](https://drive.google.com/file/d/1PX8XDF3nNMZAvcjL6M5hurwX78ePAhSs/view?usp=sharing)

### Hockey

The Hockey data contains the legibility and jersey-number datasets used by the original pipeline.

Request access from the original dataset authors and extract it under:

```text
data/Hockey
```

### Trained Legibility Classifier Weights

- [Hockey](https://drive.google.com/file/d/1RfxINtZ_wCNVF8iZsiMYuFOP7KMgqgDp/view?usp=sharing)
- [SoccerNet](https://drive.google.com/file/d/18HAuZbge3z8TSfRiX_FzsnKgiBs-RRNw/view?usp=sharing)

## Configuration

Update `configuration.py` to configure dataset paths, dependency paths, model checkpoints, and output directories.

Individual pipeline stages can be enabled or disabled in `main.py` for targeted experiments.

## Inference

### SoccerNet

```bash
python3 main.py SoccerNet test
```

### Hockey

```bash
python3 main.py Hockey test
```

## Training

### Hockey legibility classifier

```bash
python3 legibility_classifier.py \
  --train \
  --arch resnet34 \
  --sam \
  --data <new-dataset-directory> \
  --trained_model_path ./experiments/hockey_legibility.pth
```

### Hockey PARSeq fine-tuning

```bash
python3 main.py Hockey train --train_str
```

### SoccerNet legibility fine-tuning

SoccerNet training uses weak labels generated from models trained on Hockey data.

```bash
python3 legibility_classifier.py \
  --finetune \
  --arch resnet34 \
  --sam \
  --data <new-dataset-directory> \
  --full_val_dir <new-dataset-directory>/val \
  --trained_model_path ./experiments/hockey_legibility.pth \
  --new_trained_model_path ./experiments/sn_legibility.pth
```

### SoccerNet PARSeq fine-tuning

```bash
python3 main.py SoccerNet train --train_str
```

## Project Lineage

This project builds on:

1. **Koshkina, M. & Elder, J. H. (2024), A General Framework for Jersey Number Recognition in Sports Video**
2. Original implementation: [mkoshkina/jersey-number-pipeline](https://github.com/mkoshkina/jersey-number-pipeline)
3. Course-project extension: [MahmoudOsama97/jersey-number-pipeline_PlusPlus](https://github.com/MahmoudOsama97/jersey-number-pipeline_PlusPlus)

The modifications documented above were developed as a COSC 419/519B project by Chinmay Arvind, Mouhamed Jaber, Atharva Jagtap, Jayden Jayawardhena, and Mahmoud Soliman.

## References

1. Koshkina, M., & Elder, J. H. (2024). **A General Framework for Jersey Number Recognition in Sports Video.** Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 3235-3244.
2. Xu, Y., Zhang, J., Zhang, Q., & Tao, D. (2022). **ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation.** Advances in Neural Information Processing Systems, 35, 38571-38584.
3. Dao, T. (2023). **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.** arXiv:2307.08691.
4. Gordić, A. (2024). **ELI5: Flash Attention.** Medium.

## Citation

If you use the underlying jersey-number recognition framework, please cite the original work:

```bibtex
@InProceedings{Koshkina_2024_CVPR,
    author    = {Koshkina, Maria and Elder, James H.},
    title     = {A General Framework for Jersey Number Recognition in Sports Video},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {3235-3244}
}
```

## Acknowledgements

We would like to thank the authors of the following repositories:

- [PARSeq](https://github.com/baudm/parseq)
- [Centroid-ReID](https://github.com/mikwieczorek/centroids-reid)
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
- [SoccerNet](https://github.com/SoccerNet/sn-jersey)
- [McGill Hockey Player Tracking Dataset](https://github.com/grant81/hockeyTrackingDataset)
- [SAM](https://github.com/davda54/sam)

## License

[![License](https://i.creativecommons.org/l/by-nc/3.0/88x31.png)](http://creativecommons.org/licenses/by-nc/3.0/)

This work is licensed under a [Creative Commons Attribution-NonCommercial 3.0 Unported License](http://creativecommons.org/licenses/by-nc/3.0/).
