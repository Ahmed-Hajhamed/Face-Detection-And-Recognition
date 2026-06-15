# Face-Detection-And-Recognition
<img width="1400" height="800" alt="Python 3 11 7_13_2025 2_51_45 AM" src="https://github.com/user-attachments/assets/a1f27af5-1af4-4c17-b697-d1d93e9de107" />

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
├─ Train ├─ s1/
|        │ ├─ img1.pgm
|        │ └─ ...
|        ├─ s2/
|        └─ ...
├─ Test  ├─ ...
```

3. **Face Detection & Cropping**  
Detects faces in each image using Haar cascades, draws bounding rectangles, crops and resizes faces for recognition.

4. **PCA / Eigenfaces**  
- Mean-centering and covariance matrix computation 
- Eigen decomposition (PCA) to find principal components (“eigenfaces”)  
- Projects faces into lower-dimensional PCA space

4. **Face Recognition**  
- Recognition via nearest neighbor in PCA space  

---

## ⚙️ Installation & Dependencies

```
pip install numpy opencv-python scikit-learn matplotlib
```
Run ``` main.py ```

📈 How It Works
Data Loading: Crops faces and flattens into vectors

PCA: Finds eigenfaces and projects training data

Recognition:

Nearest Neighbor based on Euclidean distance in PCA space

Evaluation:

ROC curve:

<img width="1500" height="1500" alt="ROC_Curve" src="https://github.com/user-attachments/assets/1d27aa93-69e6-4b4a-b6f0-4f542cb23c34" />

📋 Results
Accuracy: 93.5 %

Snapshots:
<img width="1400" height="800" alt="Python 3 11 7_13_2025 2_51_23 AM" src="https://github.com/user-attachments/assets/9c5f3dba-8b29-4b7c-8f95-56fef49720f6" />


<img width="1400" height="800" alt="Python 3 11 7_13_2025 2_50_45 AM" src="https://github.com/user-attachments/assets/6b10d263-8649-40ec-95bc-a1acec5ea587" />

## 👨‍💻 Contributers

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Ahmed-Hajhamed">
        <img src="https://github.com/Ahmed-Hajhamed.png?size=100" width="100"><br>
        <b>Ahmed Hajhamed</b>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/AhmedEtma">
        <img src="https://github.com/AhmedEtma.png?size=100" width="100"><br>
        <b>Ahmed Etman</b>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/zeyad-wail">
        <img src="https://github.com/zeyad-wail.png?size=100" width="100"><br>
        <b>Zeyad Wail</b>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/MohamadAhmedAli">
        <img src="https://github.com/MohamadAhmedAli.png?size=100" width="100"><br>
        <b>Mohamed Ahmed</b>
      </a>
    </td>
  </tr>
</table>
