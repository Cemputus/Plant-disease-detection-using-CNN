# 🌱 Plant Disease Detection using CNN

## 📌 Overview
This repository contains a **Convolutional Neural Network (CNN)**-based system for detecting plant diseases from leaf images.  
The model is trained on the **PlantVillage dataset** and deployed as a user-friendly web application:  
👉 [CenFarm Web App](https://cenfarm.vercel.app/)  ***(I have not fully integrated the model into the web App)***


Farmers, agronomists, and researchers can upload leaf images to receive **real-time disease predictions**, confidence scores, and practical remedy suggestions.  
This system aims to improve **early disease detection**, reduce **crop losses**, and support **sustainable agriculture**.

---

## ✨ Features
- 🔍 **Automated Detection**: Supports common plant diseases (e.g., bacterial spot, early blight, late blight, leaf mold, yellow leaf curl virus, mosaic virus).  
- 📊 **High Accuracy**: CNN model achieves **>95% accuracy** on validation sets.  
- 🌐 **Web Deployment**: Powered by **Streamlit + Vercel** for easy online access.  
- 🖼️ **Data Preprocessing**: Includes resizing, normalization, and augmentation (rotation, flips, zooms).  
- 📈 **Evaluation Metrics**: Accuracy, precision, recall, F1-score, confusion matrix.  
- 💡 **Remedy Suggestions**: Practical treatment guidelines for detected diseases.  

---

## 🛠️ Technologies Used
- **Language**: Python 3.x  
- **Deep Learning**: TensorFlow, Keras  
- **Data Processing**: NumPy, Pandas  
- **Image Processing**: OpenCV, Pillow  
- **Web App**: Streamlit  
- **Deployment**: Vercel  
- **Version Control**: Git  

---

## 📂 Dataset
- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)  
- ~54,000 leaf images from **14 crops** across **38 classes** (healthy + diseased).  
- Preprocessing steps:
  - Resize → 256×256 pixels  
  - Normalize pixel values to `[0, 1]`  
  - Augment with shear, zoom, flips, rotations  
- Data split: **80% train, 10% validation, 10% test**  

---

## 🧠 Model Architecture
CNN model design:
1. **Input Layer**: 256×256×3 RGB image  
2. **Conv Layers**: Three convolutional blocks (32 → 64 → 128 filters) + ReLU + MaxPooling  
3. **Dropout**: 0.25 after pooling to reduce overfitting  
4. **Flatten**: Convert feature maps → vector  
5. **Dense Layers**:  
   - Dense(512, ReLU)  
   - Dense(38, Softmax)  
6. **Output**: Predicted disease class  

**Optimizer**: Adam (lr=0.001)  
**Loss**: Categorical Crossentropy  
**Callbacks**: EarlyStopping + ModelCheckpoint  

---

## ⚙️ Installation

Clone and set up locally:

```bash
git clone https://github.com/Cemputus/Plant-disease-detection-using-CNN.git
cd Plant-disease-detection-using-CNN
```

- dataset - https://huggingface.co/datasets/AI-Lab-Makerere/beans
- https://www.kaggle.com/datasets/emmarex/plantdisease

- code for vision transformer - https://colab.research.google.com/drive/1uds_1IjueLNG7y2IhQt96OwUtGnK-6c6?usp=sharing
