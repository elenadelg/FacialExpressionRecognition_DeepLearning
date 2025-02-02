# Computer Vision: Facial Expression Analysis for EDA Prediction

## 📚 Introduction
This project investigates **ElectroDermal Activity (EDA)** prediction using video-based facial expression analysis. The study evaluates and compares several **deep learning architectures**, including **PhysNet**, **DeepPhys**, and **CNN-based baselines**, to explore their effectiveness in predicting EDA under real-world conditions using the **UBFC-Phys dataset**.

---

### 🎯 Objectives
- Evaluate deep learning models for predicting EDA from facial videos.
- Improve performance through **data augmentation** and **face cropping** techniques.
- Assess generalization capabilities and explore limitations in handling small, noisy datasets.

---

## 📦 Dataset: UBFC-Phys
- **Participants:** 56 individuals aged 19 to 38.
- **Data streams:** 
  - Synchronized **video frames (4 FPS)**
  - **EDA signals (4 Hz)**  
  - **Blood Volume Pulse (BVP) signals (64 Hz)**
- **Tasks:**  
  Each participant was recorded during three tasks simulating different physiological states:
  - **Resting**
  - **Speech**
  - **Arithmetic tests**

---

## 🛠️ Analysis 

### 🔍 Models Overview

#### 1. **Sequential CNNs**
- A **Sequential CNN** learns temporal patterns from facial video frames and integrates **LSTMs** to capture dependencies over time.
- **Sequential CNN + BVP:** Incorporates **BVP signals** alongside video data to enhance prediction accuracy.

#### 2. **PhysNet**
- Employs **spatio-temporal 3D convolutions** to capture **blood volume variations** across facial frames.
- Enhanced to handle **BVP signals** for better prediction of EDA.

#### 3. **DeepPhys**
- Incorporates **attention mechanisms** to focus on physiologically relevant regions, such as skin areas.
- Processes **appearance** and **motion tensors** via a dual-design **CNN architecture**.

---

### 📏 Evaluation Metrics
- **Mean Squared Error (MSE):** Measures absolute accuracy by quantifying the difference between actual and predicted EDA signals.
- **Negative Pearson Correlation (NPC):** Evaluates the linear correlation between actual and predicted EDA signals.  
- **Grad-CAM:** Visualizes model attention, highlighting the facial regions that contribute the most to EDA predictions.

---

## 🔬 Experiments

### Data Augmentation
- Random transformations (e.g., **rotations**, **color jitter**, **flips**) help models learn generalized patterns and reduce overfitting.

### Face Cropping
- **Face regions** are isolated using the **face_recognition library**, allowing the models to focus on meaningful features and reduce noise.

---

## 📊 Results

| **Model**                | **MSE (Test)** | **NPC (Test)** |
|-------------------------|----------------|----------------|
| Sequential CNN           | 0.0035         | 0.0040         |
| Sequential CNN + BVP     | 0.0534         | 0.0584         |
| PhysNet                  | 0.0751         | -0.0033        |
| DeepPhys                 | 0.0790         | -0.0762        |

### 🔍 Findings
- **Data Augmentation** and **face cropping** improved the performance of PhysNet and CNN models but were insufficient to fully resolve challenges related to generalization.
- **DeepPhys** demonstrated superior performance over other models, although **data limitations** hindered further improvements.

---

## ⚠️ Challenges and Limitations

### 1. **Data Quality and Diversity**
- The small dataset (56 participants) and **gender imbalance** limited model generalization.
- **Systematic bias** was introduced due to homogeneous recording environments.

### 2. **Limited Temporal Resolution**
- Downsampling video from **35 FPS to 4 FPS** reduced the available temporal context, limiting the ability to capture subtle variations in facial expressions.

---

## 📚 References
- **PhysNet:** Zitong Yu et al.  
- **DeepPhys:** W. Chen and D. McDuff  
- **UBFC-Phys Dataset:** R. M. Sabour et al.  

---

## 🛠️ Repository Structure
```plaintext
/FacialExpressionRecognition_DeepLearning
    ├── 01_DataExtraction/                 # Scripts for downloading, extracting, and normalizing raw data
    │   ├── utils_datadownloader.py        # Downloading raw UBFC-Phys data
    │   ├── utils_dataextraction.py        # Data extraction and organization scripts
    │   └── utils_normalization.py         # Data normalization functions
    │
    ├── 02_ExploratoryDataAnalysis/        # Jupyter notebooks for initial data analysis
    │   ├── exploratorydataanalysis.ipynb  # EDA, trends, and descriptive statistics
    │   ├── summary_patients.ipynb         # Summarizes patient-level insights
    │   └── utils.py                       # Utility functions for EDA
    │
    ├── 03_DataPreprocessing/              # Scripts for preparing data for model training
    │   ├── CNN_Physnet_tensor.py          # Tensor generation for CNN and PhysNet models
    │   └── DeepPhys_tensor.py             # Tensor creation specific to DeepPhys architecture
    │
    ├── 04_Models/                         # Model architectures and training scripts
    │   ├── dataloader/                    # Data loading utilities
    │   │   ├── DeepPhys_loader.py         # Data loader for DeepPhys
    │   │   ├── PhysNet_loader.py          # Data loader for PhysNet
    │   │   └── seqCNN_loader.py           # Data loader for Sequential CNN
    │   │
    │   ├── evaluation/                    # Evaluation scripts
    │   │   ├── gradcam/                   # Grad-CAM visualizations
    │   │   │   ├── DeepPhys_gradcam.py    # Grad-CAM for DeepPhys
    │   │   │   ├── PhysNet_gradcam.py     # Grad-CAM for PhysNet
    │   │   │   └── seqCNN_gradcam.py      # Grad-CAM for Sequential CNN
    │   │   ├── loss/                      # Loss computation and NPC evaluation
    │   │   │   ├── DeepPhys_npc.py        # Negative Pearson Correlation for DeepPhys
    │   │   │   ├── PhysNet_npc.py         # NPC for PhysNet
    │   │   │   └── seqCNN_npc.py          # NPC for Sequential CNN
    │   │
    │   ├── neural_models/                 # Core model architectures
    │   │   ├── DeepPhys.py                # DeepPhys model implementation
    │   │   ├── PhysNet.py                 # PhysNet model implementation
    │   │   ├── seqCNN_BVP.py              # Sequential CNN model with BVP signals
    │   │   └── seqCNN.py                  # Basic Sequential CNN model
    │   │
    │   ├── test.py                        # Testing and model evaluation script
    │   └── training.py                    # Training pipeline
    │
    └── .python-version                    # Python version specification
