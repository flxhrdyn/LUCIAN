---
language: en
license: mit
library_name: keras
pipeline_tag: image-classification
tags:
  - medical
  - histopathology
  - convnext
  - lung-cancer
  - keras
  - tensorflow
model_file: convnext_lung_82.keras
---

# ConvNeXt Lung Histopathology Classifier (LUCIAN)

This is a **model card** intended to be used as the `README.md` of the Hugging Face model repository:

- https://huggingface.co/felixhrdyn/convnextv1-lung-cancer

The model is a fine-tuned **ConvNeXt-Base** classifier for lung histopathology tissue type prediction.

## Model Details

- **Model type:** Image classifier (3 classes)
- **Base architecture:** ConvNeXt-Base (pretrained on ImageNet-1K)
- **Framework:** TensorFlow / Keras
- **Model file:** `convnext_lung_82.keras`
- **Input:** RGB image resized to **224×224**, scaled to **[0, 1]**
- **Output:** Softmax probabilities for 3 tissue classes

### Labels

The model predicts one of the following:

1. `Adenocarcinoma`
2. `Benign Tissue`
3. `Squamous Cell Carcinoma`

## Intended Use

- Educational and research use for **lung histopathology image classification**.
- Prototyping and benchmarking on data that is similar to the training distribution.

## Out-of-Scope Use

- Clinical diagnosis or patient-care decisions.
- Use on data distributions significantly different from the training dataset without validation.

## Training Data

- **Dataset:** LC25000 Lung and Colon Histopathology Dataset (lung subset)
- **Subset used:** 3 lung classes with **1,000 images per class** (3,000 total)

> This model card summarizes the training/evaluation reported by the LUCIAN project.

## Training Procedure (High-level)

- Transfer learning using ConvNeXt-Base pretrained weights.
- A small classification head is added and the model is fine-tuned.
- The “best checkpoint” is selected by **maximum `val_accuracy`**.

## Evaluation

Two split strategies were evaluated in the LUCIAN project:

| Metric | Split 80:10:10 (Final) | Split 70:15:15 |
|--------|-------------------------|----------------|
| Train Accuracy | 96.08% | 94.95% |
| Validation Accuracy | 96.67% | 94.00% |
| **Test Accuracy** | **93.67%** | **90.44%** |
| Precision (macro) | 93.63% | 90.47% |
| Recall (macro) | 93.67% | 90.44% |
| F1-Score (macro) | 93.64% | 90.39% |

> Metrics are reported from the best checkpoint epoch selected by `ModelCheckpoint(monitor='val_accuracy', mode='max')`.

## How to Use

### Load model from the Hub

```python
from huggingface_hub import hf_hub_download
import tensorflow as tf

path = hf_hub_download(
    repo_id="felixhrdyn/convnextv1-lung-cancer",
    filename="convnext_lung_82.keras",
)
model = tf.keras.models.load_model(path)
```

### Preprocess and run inference

This matches the preprocessing used in the LUCIAN app (`resize(224, 224)`, RGB, normalize to `[0, 1]`).

```python
import numpy as np
from PIL import Image
import tensorflow as tf

IMAGE_SIZE = (224, 224)
LABELS = ["Adenocarcinoma", "Benign Tissue", "Squamous Cell Carcinoma"]

img = Image.open("your_image.jpg").convert("RGB").resize(IMAGE_SIZE)
x = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

probs = model(tf.convert_to_tensor(x), training=False).numpy()[0]
pred = LABELS[int(np.argmax(probs))]
print(pred, probs)
```

## Limitations and Risks

- The model was trained on a specific dataset and preprocessing pipeline; performance may degrade on different scanners, staining protocols, magnifications, or patient populations.
- Like most CNNs, it can be sensitive to artifacts (blur, compression, pen marks) and distribution shift.
- This is **not** a medical device and has **not** undergone clinical validation.

## Ethical Considerations

- Do not use as a substitute for expert pathology assessment.
- If used in downstream workflows, ensure appropriate human oversight, dataset audits, and validation.

## Citation

If you use this model, cite the LUCIAN project (and the ConvNeXt paper):

- Liu et al., “A ConvNet for the 2020s” (ConvNeXt)

(You can add your thesis citation here once you have a stable reference.)
