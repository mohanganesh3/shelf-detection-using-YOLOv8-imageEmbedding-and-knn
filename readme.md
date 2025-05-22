# 🛒 Product Identification and Counting System

## 📌 Overview

The **Product Identification and Counting System** is a computer vision project that detects, classifies, and counts products (e.g., Coca-Cola bottles, Fanta cans, Lays packets) in retail shelf images. It automates inventory management tasks like stock-taking and shelf monitoring using deep learning and machine learning.

Key technologies used:
- **YOLO** for object detection
- **ResNet18** for feature extraction
- **k-NN** for classification
- **Streamlit** for user interaction and result refinement

---

## 🎯 Purpose

- Detect objects in retail shelf images  
- Classify detected objects using a dynamic knowledge base  
- Count high-confidence product instances  
- Enable user confirmation for uncertain predictions to improve future accuracy  

---

## ✨ Features

- **Object Detection**: Uses YOLO with configurable `yolo_conf=0.5` and `yolo_iou=0.4`
- **Feature Extraction**: 512-dimensional embeddings via ResNet18
- **Classification**: k-NN classifier (`knn_neighbors=5`)
- **Web Interface**: Built with Streamlit for easy interaction
- **Self-Learning**: Updateable knowledge base for continual learning

---

## 🏗️ System Architecture

```
product-identification-system/
├── data/
│   ├── img/                      # Input images
│   ├── knowledgebase/crops/object/  # Cropped images for knowledge base
│   ├── temp/                     # Preprocessed temp images
├── models/
│   ├── best.pt                   # YOLO model
├── src/
│   ├── img2vec_resnet18.py       # Feature extractor
├── notebooks/                    # t-SNE visualizations, testing
├── config.yaml                   # Paths, thresholds, colors
├── requirements.txt              # Python dependencies
├── env.yaml                      # Conda environment setup
├── app.py                        # Streamlit frontend
├── main.py                       # Backend logic
└── predictions.txt               # Output results
```

---

## ⚙️ Prerequisites

- Python 3.8+
- Conda
- GPU (optional for faster YOLO inference)
- OS: Windows/Linux/MacOS

---

## 🔧 Installation

1. **Clone the Repository**

```bash
git clone https://github.com/<your-username>/product-identification-system.git
cd product-identification-system
```

2. **Create Conda Environment**

```bash
conda env create -f env.yaml
```

3. **Activate the Environment**

```bash
conda activate facings-identifier
```

4. **Install PyTorch with GPU (optional)**

```bash
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> ⚠️ Skip `--index-url` if you're using CPU.

5. **Install Other Dependencies**

```bash
pip install -r requirements.txt
pip install streamlit pillow torch supervision tqdm pyyaml
```

---

## 🚀 Usage

1. **Run the Streamlit App**

```bash
streamlit run app.py
```

2. **Interact with the Interface**

- Upload an image (e.g., `data/img/cocacolabottle.jpeg`)
- View detections with bounding boxes and confidence (e.g., "Coca-Cola (85%)")
- Confirm uncertain predictions or add new ones
- Results saved to `predictions.txt`

---

## 🔄 Workflow

### 1. Image Upload & Preprocessing
- Uploaded via `app.py` → stored in `data/img/`
- Preprocessing via `main.py` → stored in `data/temp/`

### 2. Object Detection
- YOLO (`models/best.pt`) with thresholds
- Detected crops saved to `data/<image_stem>/crops/object/`

### 3. Classification
- Embeddings via `img2vec_resnet18.py`
- Labels via k-NN using knowledge base

### 4. Visualization
- Labels & boxes drawn using `config.yaml`
- Displayed in Streamlit

### 5. User Confirmation
- Confirm or add new classifications
- Saved to `data/knowledgebase/crops/object/`

### 6. Result Summary
- Counts shown (e.g., "Found 3 Coca-Cola(s)")
- Saved to `predictions.txt`

---

## ⚠️ Challenges & Observations

- 🔁 **Dependency Compatibility**: Match PyTorch and CUDA versions  
- 📚 **Knowledge Base Quality**: Critical for classification accuracy  
- 🧑‍💻 **Interface Limitations**: Batch confirmation can be improved  
- 🔧 **YOLO Thresholds**: Fine-tuning needed for robust detection  

---

## 🤝 Contributing

Contributions are welcome!

```bash
# Step 1
Fork the repository

# Step 2
Create a feature branch
git checkout -b feature-name

# Step 3
Make your changes and commit
git commit -m "Add feature"

# Step 4
Push your changes
git push origin feature-name

# Step 5
Open a pull request
```

---

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## Acknowledgments

- Built with [Ultralytics YOLO](https://github.com/ultralytics/yolov5), [PyTorch](https://pytorch.org/), and [Streamlit](https://streamlit.io/)
- Developed as part of an internship project on **May 20, 2025** by **Mohan Ganesh**
