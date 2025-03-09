# Sign Language Classification

This project provides a machine learning pipeline for sign language classification using Vision Transformers (ViT).

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Sign-Language-MLOps

# Install dependencies
pip install -e .
```

## Usage

### Training Mode

To train a new model:

```bash
python main.py --mode train
```

This will:
1. Extract frames from sign language videos
2. Prepare the dataset for training
3. Fine-tune a ViT model on the dataset
4. Save the trained model

### Inference Mode

To run real-time inference using your webcam:

```bash
python main.py --mode inference
```
or simply:
```bash
python main.py
```

This will:
1. Load the trained model
2. Activate your webcam
3. Perform real-time sign language classification

Press 'q' to quit the inference mode.

## Configuration

Model and training parameters can be modified in the `params.yaml` file.

## Requirements

See `requirements.txt` for the full list of dependencies.
