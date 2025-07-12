# Face-Detection-And-Recognition
A Python project that performs **face detection** and **face recognition** using:
- OpenCV’s Haar Cascade for real-time face detection
- PCA (Eigenfaces) for facial feature extraction and recognition
- Optional SVM classifier for enhanced accuracy

---

## 🧭 Project Overview

1. **Dataset Structure**  
   Organize your data into:
```
dataset/
├─ s1/
│ ├─ img1.jpg
│ └─ ...
├─ s2/
└─ ...
```

3. **Face Detection & Cropping**  
Detects faces in each image using Haar cascades, draws bounding rectangles, crops and resizes faces for recognition.

4. **PCA / Eigenfaces**  
- Mean-centering and covariance matrix computation  
- Eigen decomposition (PCA) to find principal components (“eigenfaces”)  
- Projects faces into lower-dimensional PCA space

4. **Face Recognition**  
- Recognition via nearest neighbor in PCA space  
- (Optional) SVM classifier for trained-level classification  

5. **Evaluation**  
- Supports train/test splitting (e.g., 80/20, k-fold, leave-one-out)  
- Outputs accuracy metrics and confusion matrices  

---

## ⚙️ Installation & Dependencies

```
pip install numpy opencv-python scikit-learn matplotlib
```
🔧 Usage
```
python main.py \
  --dataset dataset/ \
  --mode train \
  --n_components 50 \
  --split 0.2
```
Modes:
train: prepare data, compute PCA & classifier, save model

test: load model and evaluate accuracy

recog: recognize faces in images or webcam

Parameters:
--n_components: number of PCA components (eigenfaces)

--split: train/test split ratio (e.g., 0.2 = 20% test)

📈 How It Works
Data Loading: Crops faces and flattens into vectors

PCA: Finds eigenfaces and projects training data

Recognition:

Nearest Neighbor based on Euclidean distance in PCA space

Evaluation:

Calculates accuracy and ROC curve

📋 Results
Accuracy: 93.5 %
