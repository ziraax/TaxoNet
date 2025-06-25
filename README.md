# Deep Learning Training & Inference Pipeline for Automatic Plankton classification

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![W&B Integration](https://img.shields.io/badge/Weights_&_Biases-Integrated-yellow)

An end-to-end deep learning solution supporting multiple model architectures with advanced features for training, evaluation, and production-ready inference.

## Features

### Training Pipeline
- **Multi-model Support**: YOLOv11, ResNet, DenseNet, EfficientNet
- **Advanced Training**:
  - Configurable hyperparameters
  - Early stopping with checkpointing
  - Class-weighted loss functions
  - Learning rate scheduling
- **Model Factory Pattern**: Dynamic model creation with variants
- **MC-Dropout Integration**: Bayesian uncertainty quantification
- **W&B Integration**: Real-time tracking of metrics, artifacts, and model versions

### Inference Pipeline
- **Batch Processing**: Efficient handling of image directories
- **Uncertainty Quantification**: Monte Carlo Dropout with configurable iterations
- **Flexible Output**:
  - Top-K predictions with confidence scores
  - CSV/JSON export capabilities
  - Interactive visualizations (to be implemented)
- **Production Ready**: Device-aware execution (CPU/GPU)

## Installation

1. **Clone Repository**:
```bash
git clone https://github.com/ziraax/TaxoNet.git
cd TaxoNet
``` 

2. **Install Dependancies**:
```bash
pip install -r requirements.txt
``` 

3. **Weights & Biases Setup**
```bash
wandb login
``` 

## Usage 
### Training
More to come but a basic example : 

```bash
python3 main.py \
  --model_name densenet \
  --densenet_variant 121 \
  --pretrained \
  --mc_dropout \
  --mc_p 0.3
``` 

Key Arguments:
TODO

### Inference

```bash
python -m inference.predict \
  --image_dir PATH/TO/UR/IMAGES \
  --model_name [resnet, densenet, efficientnet, yolov11] \
  --weights_path model_weights/[model_type]/[model_variant]/[run] \
  --densenet_variant 121 \
  --resnet_variant
  --efficientnet_variant 
  --mc_dropout \
  --mc_iterations 30 \
  --save_csv predictions.csv \
  --wandb_log
```

Key arguments: 
TODO

## Model Factory

Supported architectures and variants:

| Model        | Variants                  | Pretrained | MC-Dropout |
|--------------|---------------------------|------------|------------|
| YOLOv11      | Classification            | ✅         | ❌         |
| DenseNet     | 121, 169, 201, 161        | ✅         | ✅         |
| EfficientNet | b0-b7                     | ✅         | ✅         |
| ResNet       | 18, 34, 50, 101, 152      | ✅         | ✅         |


## Configuration

Modify `config.py` : 

```python
{
    "project_name": "my_experiment",
    "batch_size": 128,
    "learning_rate": 3e-4,
    "early_stopping_patience": 15,
    "img_size": 224,  # Auto-configured per model
    "yolo_dataset_path": "./datasets/yolo_data.yaml"
}
```

## Results and monitoring 

TODO

## Contributing 

To specify

## License

To specify





