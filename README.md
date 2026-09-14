# ♻️ Garbage Classifier — Image Classification with Deep Learning

A deep learning project that classifies images of waste items into recycling
categories (cardboard, glass, metal, paper, plastic, trash) using **transfer
learning** on a pretrained ResNet18. Includes a full training pipeline and an
interactive web demo built with Gradio.

## 🎯 Motivation

Incorrect waste sorting is a major bottleneck in recycling efficiency. This
project explores whether a lightweight, transfer-learned CNN can reliably
classify waste photos into the correct recycling stream — a simplified
prototype of the kind of vision system used in smart bins and automated
sorting facilities.

## 🧠 Approach

- **Model**: ResNet18 pretrained on ImageNet, with the backbone frozen and a
  new fully-connected classification head trained on the garbage dataset
  (transfer learning).
- **Data augmentation**: random crops, flips, rotation, and color jitter on
  the training set to reduce overfitting on a relatively small dataset.
- **Dataset**: [Garbage Classification dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
  on Kaggle (6 classes: cardboard, glass, metal, paper, plastic, trash).
- **Evaluation**: accuracy/loss curves, confusion matrix, and a full
  per-class classification report (precision/recall/F1) on a held-out
  validation set.

## 📊 Results

> _Fill this in after you run training — see "Train the model" below._

| Metric | Value |
|---|---|
| Validation accuracy | `TODO: e.g. 91.2%` |
| Training time | `TODO: e.g. ~4 min on Colab T4 GPU` |
| Best epoch | `TODO` |

![Training curves](models/training_curves.png)
![Confusion matrix](models/confusion_matrix.png)

## 🚀 Demo

Run the interactive web demo locally:

```bash
python app.py
```

Then open the local URL printed in your terminal (usually
`http://127.0.0.1:7860`), upload a photo of a waste item, and see the
model's prediction with confidence scores for each class.

_(Optional: add a GIF or screenshot of the demo here once you run it.)_

## 📁 Project Structure

```
garbage-classifier/
├── prepare_data.py      # Downloads dataset from Kaggle and splits into train/val/test
├── train.py              # Trains the model (transfer learning on ResNet18)
├── predict.py             # CLI script to classify a single image
├── app.py                 # Gradio web demo
├── requirements.txt       # Python dependencies
├── data/                   # (created after running prepare_data.py, gitignored)
└── models/                  # (created after training: weights, plots, class names)
```

## 🛠️ Setup & Usage

This project is designed to run easily on **Google Colab** (free GPU) or
locally if you have a GPU.

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/garbage-classifier.git
cd garbage-classifier
pip install -r requirements.txt
```

### 2. Get Kaggle API access

1. Create a free account at [kaggle.com](https://www.kaggle.com).
2. Go to your profile → **Settings** → **API** → **Create New Token**. This
   downloads `kaggle.json`.
3. Place it at `~/.kaggle/kaggle.json` (or upload it to your Colab session
   and run `mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/`).

### 3. Download and prepare the data

```bash
python prepare_data.py
```

This downloads the dataset and splits it into `data/split/{train,val,test}/`,
organized by class folder (ready for `torchvision.datasets.ImageFolder`).

### 4. Train the model

```bash
python train.py --epochs 10 --batch_size 32 --lr 0.001
```

This trains the model, saves the best-performing weights to
`models/best_model.pth`, and generates training curves and a confusion
matrix in `models/`. On a free Colab T4 GPU this takes roughly 10–20
minutes depending on dataset size.

### 5. Run predictions from the command line

```bash
python predict.py --image path/to/your/photo.jpg
```

### 6. Launch the web demo

```bash
python app.py
```

## 🔍 What I Learned / Key Techniques

- **Transfer learning**: rather than training a CNN from scratch (which
  needs huge datasets), freezing a pretrained backbone and training only a
  new classification head lets you get strong results from a small dataset
  in minutes.
- **Data augmentation** as a lightweight way to fight overfitting when
  training data is limited.
- **Evaluation beyond accuracy**: a confusion matrix reveals *which* classes
  get confused (e.g. glass vs. plastic often look visually similar), which
  is more actionable than a single accuracy number.
- **End-to-end deployment**: going from a trained `.pth` file to something
  a non-technical person can actually interact with, via Gradio.

## ⚠️ Limitations

- Trained on a relatively small, curated dataset — real-world waste photos
  (varied lighting, backgrounds, damaged/dirty items) would likely reduce
  accuracy.
- 6 broad categories only; doesn't handle mixed-material items (e.g. a
  plastic-coated paper cup) or contamination detection.
- Not tuned for deployment on edge devices (e.g. a physical smart bin) —
  the model isn't quantized or optimized for inference speed.

## 🔮 Future Improvements

- Fine-tune more layers of the backbone (not just the final layer) once a
  baseline is established.
- Expand to more granular classes or add an object-detection stage to
  handle multiple items in one photo.
- Collect a small "real-world" test set (phone photos in different
  lighting) to sanity-check generalization beyond the curated dataset.

## 📄 License

MIT — feel free to use or adapt this project.

## 🙏 Acknowledgments

- Dataset: [Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) on Kaggle.
- Base model: `torchvision.models.resnet18`, pretrained on ImageNet.
