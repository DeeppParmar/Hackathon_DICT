# Python Libraries for MediScan AI Project

## Complete List of Required Python Libraries

This document lists all Python libraries that should be installed in the WSL environment for the MediScan AI project, based on the PPT.txt specifications.

---

## 📦 Installation Command

```bash
# Activate your virtual environment first
source tensor/bin/activate

# Install all libraries
pip install -r requirements.txt

# OR install individually (see categories below)
```

---

## 🧠 Deep Learning Frameworks

### 1. **tensorflow** (>=2.13.0)
- **Purpose**: Primary deep learning framework for CNN model development
- **Usage**: Model architecture (ResNet-50), training, inference
- **Install**: `pip install tensorflow[and-cuda]` (for GPU support)

### 2. **keras** (>=2.13.0)
- **Purpose**: High-level neural network API (included with TensorFlow)
- **Usage**: Model building, layer definitions, training loops

---

## 🖼️ Computer Vision & Image Processing

### 3. **opencv-python** (>=4.8.0)
- **Purpose**: Image preprocessing, contour detection, visualization
- **Usage**: 
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Canny and Sobel edge detection
  - Contour mapping and morphological operations
  - Image resizing, normalization, denoising

### 4. **opencv-contrib-python** (>=4.8.0)
- **Purpose**: Extended OpenCV features
- **Usage**: Additional image processing algorithms

### 5. **Pillow** (>=10.0.0)
- **Purpose**: Image file handling and basic operations
- **Usage**: Reading, saving, converting image formats

---

## 🔢 Numerical Computing

### 6. **numpy** (>=1.24.0)
- **Purpose**: Numerical operations on arrays
- **Usage**: Image array manipulation, mathematical operations

### 7. **scipy** (>=1.11.0)
- **Purpose**: Scientific computing utilities
- **Usage**: Signal processing, optimization, statistical functions

---

## 📊 Data Processing

### 8. **pandas** (>=2.0.0)
- **Purpose**: Data manipulation and analysis
- **Usage**: Handling dataset metadata, CSV operations

### 9. **scikit-learn** (>=1.3.0)
- **Purpose**: Machine learning utilities
- **Usage**: Data preprocessing, metrics calculation, model evaluation

### 10. **scikit-image** (>=0.21.0)
- **Purpose**: Image processing algorithms
- **Usage**: Additional image transformation functions

---

## 🌐 Backend API

### 11. **flask** (>=2.3.0)
- **Purpose**: Web framework for backend API
- **Usage**: REST API endpoints for model inference

### 12. **flask-cors** (>=4.0.0)
- **Purpose**: Cross-Origin Resource Sharing support
- **Usage**: Enable frontend-backend communication

### 13. **fastapi** (>=0.103.0)
- **Purpose**: Alternative modern web framework (optional)
- **Usage**: High-performance API alternative to Flask

### 14. **uvicorn** (>=0.23.0)
- **Purpose**: ASGI server for FastAPI
- **Usage**: Running FastAPI applications

---

## 🔍 Explainable AI (XAI)

### 15. **tensorflow-explain** (>=0.4.0)
- **Purpose**: Explainable AI utilities for TensorFlow
- **Usage**: Grad-CAM implementation

### 16. **lime** (>=0.2.0)
- **Purpose**: Local Interpretable Model-Agnostic Explanations
- **Usage**: Feature importance visualization (mentioned in PPT.txt)

### 17. **shap** (>=0.42.0)
- **Purpose**: SHAP (SHapley Additive exPlanations) values
- **Usage**: Feature attribution and model interpretability

---

## 📈 Visualization

### 18. **matplotlib** (>=3.7.0)
- **Purpose**: Plotting and visualization
- **Usage**: Displaying images, heatmaps, graphs

### 19. **seaborn** (>=0.12.0)
- **Purpose**: Statistical data visualization
- **Usage**: Enhanced plotting capabilities

---

## 🎨 Image Augmentation

### 20. **albumentations** (>=1.3.0)
- **Purpose**: Fast image augmentation library
- **Usage**: Data augmentation (rotation, zoom, brightness, contrast)

### 21. **imgaug** (>=0.4.0)
- **Purpose**: Alternative image augmentation library
- **Usage**: Elastic deformation, geometric transformations

---

## 🛠️ Utilities

### 22. **tqdm** (>=4.66.0)
- **Purpose**: Progress bars for loops
- **Usage**: Training progress visualization

### 23. **requests** (>=2.31.0)
- **Purpose**: HTTP library
- **Usage**: API calls, downloading datasets

### 24. **python-dotenv** (>=1.0.0)
- **Purpose**: Environment variable management
- **Usage**: Configuration management

---

## 💾 Model Utilities

### 25. **h5py** (>=3.9.0)
- **Purpose**: HDF5 file format support
- **Usage**: Saving and loading model weights (.h5 files)

### 26. **protobuf** (>=4.23.0)
- **Purpose**: Protocol buffer support
- **Usage**: TensorFlow model serialization

---

## 📋 Quick Install Script

```bash
#!/bin/bash
# Install all libraries for MediScan AI

source tensor/bin/activate

# Core Deep Learning
pip install tensorflow[and-cuda] keras

# Computer Vision
pip install opencv-python opencv-contrib-python Pillow

# Numerical & Data
pip install numpy scipy pandas scikit-learn scikit-image

# Backend
pip install flask flask-cors fastapi uvicorn

# Explainable AI
pip install tensorflow-explain lime shap

# Visualization
pip install matplotlib seaborn

# Image Augmentation
pip install albumentations imgaug

# Utilities
pip install tqdm requests python-dotenv h5py protobuf
```

---

## 🎯 Library Usage by Feature

### **Disease Detection (TB, Pneumonia, Fractures)**
- `tensorflow`, `keras` - Model architecture
- `numpy` - Array operations
- `h5py` - Model weights

### **Image Preprocessing**
- `opencv-python` - CLAHE, normalization, denoising
- `Pillow` - Image I/O
- `numpy` - Array manipulation

### **Contour Mapping**
- `opencv-python` - Contour detection, edge detection
- `scikit-image` - Segmentation utilities

### **Explainable AI (Grad-CAM)**
- `tensorflow-explain` - Grad-CAM implementation
- `lime` - LIME explanations
- `shap` - SHAP values
- `matplotlib` - Visualization

### **Backend API**
- `flask` - REST API
- `flask-cors` - CORS support
- `numpy`, `Pillow` - Image processing in API

### **Data Augmentation**
- `albumentations` - Fast augmentation
- `imgaug` - Advanced transformations

---

## ⚠️ Important Notes

1. **GPU Support**: Use `tensorflow[and-cuda]` if you have NVIDIA GPU with CUDA support
2. **Python Version**: Requires Python 3.8+ (you're using Python 3.10, which is perfect)
3. **Virtual Environment**: Always use the virtual environment (`tensor`) you created
4. **Dependencies**: Some libraries may have conflicting dependencies - install in order listed

---

## 🔄 Alternative Libraries (Optional)

If you prefer PyTorch over TensorFlow:
- `torch` - PyTorch framework
- `torchvision` - Computer vision utilities for PyTorch

---

## 📝 Summary

**Total Libraries**: 26 core libraries

**Categories**:
- Deep Learning: 2
- Computer Vision: 3
- Numerical Computing: 2
- Data Processing: 3
- Backend API: 4
- Explainable AI: 3
- Visualization: 2
- Image Augmentation: 2
- Utilities: 3
- Model Utilities: 2

---

*Generated for MediScan AI - DA-IICT Hackathon 2026*

