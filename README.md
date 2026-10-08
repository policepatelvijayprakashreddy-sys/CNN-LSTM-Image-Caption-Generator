# Image Caption Generator

An end-to-end deep learning model that automatically generates descriptive natural language captions for input images using a CNN-Transformer architecture trained on the Flickr8k dataset.

## Architecture Overview

- **Feature Extractor**: Pre-trained EfficientNetB0 extracts spatial feature maps from input images.
- **Sequence Modeling**: Multi-Head Attention and Positional Embeddings model dependencies between image regions and text tokens.
- **Decoder**: Autoregressive Transformer decoder predicts the caption token by token.
- **Training**: Custom training loop in TensorFlow/Keras with causal masking, label smoothing cross-entropy loss, and learning rate scheduling.

## Dataset

The model is trained and evaluated on the **Flickr8k Dataset**:
- 8,000 images with 5 paired natural language captions per image.
- Standard 80/20 train/validation split.

## Project Structure

```
.
├── project-code.ipynb       # Complete pipeline: data loading, model definition, training, and inference
├── Realtimeproject(ML).pdf  # Project documentation and report
├── README.md                # Project overview
└── .gitignore               # Ignored files and directories
```

## Setup and Usage

### Prerequisites
Install the required Python packages:

```bash
pip install tensorflow keras numpy matplotlib
```

### Running the Project
Open and execute the cells in `project-code.ipynb`:
1. The notebook automatically downloads and extracts the Flickr8k dataset.
2. Captions are tokenized and processed into input pipelines using `tf.data`.
3. The CNN-Transformer model trains with early stopping.
4. Run the caption generation cells at the end to generate captions on test images.
