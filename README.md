# Object Detection in Drone Imagery (VisDrone)

CS6366 Neural Networks and Deep Learning, final project.
Example project: **Bounding Box**.

## Summary

We fine-tune a COCO-pretrained YOLO11 detector in PyTorch, using the Ultralytics library, to predict bounding boxes for pedestrians and vehicles in drone imagery from the VisDrone2019-DET dataset. Drone images are hard for detectors because most objects are tiny and crowded together: in VisDrone, most labelled objects fall in COCO's "small" size bin.

Ultralytics makes the training itself simple, so the core of our work is a **rigorous evaluation**:

- **VisDrone-aware scoring.** Ultralytics' VisDrone conversion deletes the dataset's *ignored regions*, so its built-in mAP counts correct detections inside unlabelled crowds as false positives. Our evaluator (`src/visdrone_eval.py`) rebuilds ground truth from the raw annotations, treats ignored regions as COCO crowd regions as the official VisDrone protocol does, and allows 500 detections per image.
- **Accuracy by object size and class.** COCO AP for small, medium and large objects, and for each of the 10 classes.
- **One controlled experiment.** Change a single design choice, with input size (640 against 1280) as the leading candidate, and keep the data splits, seed and schedule fixed.
- **Uncertainty and a held-out test set.** Bootstrap confidence intervals and repeated seeds where GPU time allows, with development on the validation split and one final evaluation on the held-out test-dev split.

## Team

| Name | GitHub |
|---|---|
| Quoc Nguyen | [@gizaquoc197](https://github.com/gizaquoc197) |
| Daniil Leonov | [@dleoleo](https://github.com/dleoleo) |
| Mukendi Patrick Tshiswaka | [@MukendiPatrick](https://github.com/MukendiPatrick) |

## Datasets

- **VisDrone2019-DET** (training and evaluation data): <https://github.com/VisDrone/VisDrone-Dataset>.
  P. Zhu, L. Wen, D. Du, X. Bian, H. Fan, Q. Hu and H. Ling, "Detection and Tracking Meet Drones Challenge," *IEEE TPAMI*, vol. 44, no. 11, pp. 7380–7399, 2021.
  We download it through Ultralytics' [VisDrone dataset config](https://docs.ultralytics.com/datasets/detect/visdrone/). The data is not stored in this repository.
- **COCO** (used only through the pretrained YOLO11 weights): <https://cocodataset.org/>

## Initial work

[`notebooks/01_data_and_baseline.ipynb`](notebooks/01_data_and_baseline.ipynb)
([open in Colab](https://colab.research.google.com/github/gizaquoc197/cs6366-drone-detection/blob/main/notebooks/01_data_and_baseline.ipynb))
covers the project's first stage:

1. Downloads VisDrone and audits its labels: objects per class, objects per image, object sizes, and how many objects shrink below YOLO11's finest 8-pixel stride at input sizes 640 and 1280.
2. Fine-tunes YOLO11n on a share of one epoch, to check the pipeline end to end and time an epoch on the available GPU.
3. Evaluates that model with both Ultralytics' built-in mAP and our VisDrone-aware evaluator.

The smoke-test model is not a result. The full fine-tuning runs and the controlled experiment are the next stage.

## Running it

Open the notebook in Colab with a GPU runtime and run all cells. It clones this repository and installs `requirements.txt`. To run it locally instead, install the requirements in a Python 3.10–3.12 environment and start Jupyter from the repository root.

## Repository layout

```
notebooks/01_data_and_baseline.ipynb   data audit, smoke test, evaluation harness
src/visdrone_eval.py                   VisDrone-aware COCO evaluation
requirements.txt                       pinned dependencies
```

## Related work

A previous CS6366 project ([erbayat/small-object-detection](https://github.com/erbayat/small-object-detection/)) compared YOLO11 model sizes on VisDrone, with and without sliced inference ([SAHI](https://github.com/obss/sahi)). We use a single small model and focus instead on evaluation rigor and one controlled training experiment.

## References

- J. Redmon, S. Divvala, R. Girshick and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," CVPR 2016. <https://arxiv.org/abs/1506.02640>
- Ultralytics YOLO: <https://github.com/ultralytics/ultralytics>
- PyTorch TorchVision Object Detection Finetuning Tutorial: <https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html>
