# QuantDIR Reproducibility

This repository contains the implementation and scripts used to reproduce the experiments in the QuantDIR paper.

QuantDIR is evaluated on image classification and object detection under two post-training quantization settings:

- **W4A8:** 4-bit weights and 8-bit activations
- **W4A4:** 4-bit weights and 4-bit activations

## 1. Environment Setup

### 1.1 Python Version

QuantDIR uses **Python 3.10**.

Check the installed Python version:

```bash
python3 --version
```

### 1.2 Create and Activate a Virtual Environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

---

## 2. Install Dependencies

### 2.1 Install MQBench

Clone MQBench:

```bash
git clone https://github.com/ModelTC/MQBench.git
```

Install MQBench in editable mode:

```bash
cd MQBench
pip install -e .
cd ..
```

### 2.2 Install Project Requirements

```bash
pip install -r requirements.txt
```

QuantDIR uses MQBench's Academic backend for post-training quantization.

Make sure that the installed PyTorch and TorchVision versions are compatible with MQBench.

---

## 3. Dataset Preparation

QuantDIR uses:

- **ImageNet-1K** for image classification
- **COCO 2017** for object detection
---

### 3.1 ImageNet

Download ImageNet from:

https://www.image-net.org/

The experiments require the ImageNet validation images. A typical archive is:

```text
ILSVRC2012_img_val.tar
```

Create the ImageNet directory:

```bash
mkdir -p data/test_dataset/imagenet
```

Extract the validation images:

```bash
tar -xf ILSVRC2012_img_val.tar -C data/test_dataset/imagenet
```

Organize the validation images into class-specific folders using the provided `valPrep.sh` script:

```bash
bash valPrep.sh data/test_dataset/imagenet
```
If ImageNet is stored elsewhere, update `IMAGENET_ROOT` in the corresponding experiment script.

---

### 3.2 COCO 2017

Download the COCO 2017 validation images and annotations from:

https://cocodataset.org/#download

The required archives are typically:

```text
val2017.zip
annotations_trainval2017.zip
```

Create the COCO directory:

```bash
mkdir -p data/test_dataset/coco
```

Extract the validation images:

```bash
unzip val2017.zip -d data/test_dataset/coco/
```

Extract the annotations:

```bash
unzip annotations_trainval2017.zip -d data/test_dataset/coco/
```

The object-detection experiments require:

```text
data/test_dataset/coco/val2017/
data/test_dataset/coco/annotations/instances_val2017.json
```

If COCO is stored elsewhere, update the COCO dataset path in the corresponding detection script.
---

## 4. Generate Instability Metrics 

### 4.1 Classification

To generate instability metrics for ResNet-18:

```bash
python3 instability_metrics.py --resnet18
```
Use the corresponding model option for the other supported classification models.

Supported classification models:

```text
resnet18
resnet34
resnet50
mobilenet_v2
regnet_x_800mf
vgg16
vgg19
```
### 4.2 Object Detection

```text
ssd300_vgg16
retinanet_resnet50_fpn
fasterrcnn_resnet50_fpn
```
---

## 5. RQ2: Effectiveness of Instability-Guided Repair

### 5.1 QuantDIR: Image Classification

Run ResNet-18 under W4A4:

```bash
python3 quantdir_classification.py --model resnet18 --quant w4a4
```
Run ResNet-18 under W4A8:

```bash
python3 quantdir_classification.py --model resnet18 --quant w4a8
```

Supported classification models:

```text
resnet18
resnet34
resnet50
mobilenet_v2
regnet_x_800mf
vgg16
vgg19
```

### 5.2 QuantDIR: Object Detection

Run SSD300/VGG16 under W4A4:

```bash
python3 quantdir_detection.py  --model ssd300_vgg16 --weight-bits 4  --activation-bits 4
```
Run SSD300/VGG16 under W4A8:

```bash
python3 quantdir_detection.py --model ssd300_vgg16  --weight-bits 4 --activation-bits 8
```

Supported object-detection models:

```text
ssd300_vgg16, retinanet_resnet50_fpn, fasterrcnn_resnet50_fpn
```

## 6. RQ3: Ablation Study

RQ3 evaluates two aspects of QuantDIR:

1. **Layer selection:** QuantDIR vs. Random, Early, and Late layer-selection strategies.
2. **Repair strategy:** QuantDIR repair vs. AdaRound applied only to the same selected locations.

### 6.1 Layer-Selection Ablation: Classification

Run the positional layer-selection strategies using:

```bash
python3 ablation_location_classification.py --location early --weight-bits 4  --activation-bits 8
```
Available layer-selection strategies are:

```text
early, late, random
```
Change `early` to `late` or `random` to reproduce the corresponding ablation.

### 6.2 Layer-Selection Ablation: Object Detection

Run the object-detection ablation using:

```bash
python3 ablation_location_object.py --model ssd300_vgg16 --location early --weight-bits 4  --activation-bits 8
```
Supported object-detection models:

```text
ssd300_vgg16, retinanet_resnet50_fpn, fasterrcnn_resnet50_fpn
```

### 6.3 Repair-Strategy Ablation
For Object Detection

```bash
python3 adaround_selected_classification.py --model resnet --weight-bits 4  --activation-bits 8
```

For Image Classification
```bash
python3 adaround_selected_detection.py --model ssd300_vgg16 --weight-bits 4  --activation-bits 8
```
