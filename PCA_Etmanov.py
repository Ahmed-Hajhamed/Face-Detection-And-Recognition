import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm


def load_images(dataset_path, image_shape=(100, 100)):
    X = []  # Image vectors
    y = []  # Labels
    for subject_folder in sorted(os.listdir(dataset_path)):
        subject_path = os.path.join(dataset_path, subject_folder)
        if os.path.isdir(subject_path):
            label = subject_folder  # e.g., "s1"
            for img_file in os.listdir(subject_path):
                img_path = os.path.join(subject_path, img_file)
                print(f"Loading {img_path}")
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, image_shape)
                X.append(img.flatten())
                y.append(label)
                print(f"Loaded {img_path} with label {label}")
    return np.array(X), np.array(y)

def compute_pca(X, num_components):
    mean_face = np.mean(X, axis=0)
    X_centered = X - mean_face
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvectors = eigenvectors[:, :num_components]
    return mean_face, eigenvectors, X_centered

def project_faces(X_centered, eigenvectors):
    return np.dot(X_centered, eigenvectors)

def recognize_face(test_face, mean_face, eigenvectors, projected_faces, labels):
    test_centered = test_face - mean_face
    test_proj = np.dot(test_centered, eigenvectors)
    distances = np.linalg.norm(projected_faces - test_proj, axis=1)
    return labels[np.argmin(distances)], distances[np.argmin(distances)]

# X, y = load_images("archive")
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
# X_train, y_train = load_images("dataset/train")
# X_test, y_test = load_images("dataset/test")


# ==== Load Data ====
X_train = np.load("PCA_DATA/X_train.npy")
X_test = np.load("PCA_DATA/X_test.npy")
y_train = np.load("PCA_DATA/y_train.npy")[0]
y_test = np.load("PCA_DATA/y_test.npy")[0]
mean_face = np.load("PCA_DATA/mean_face.npy")
eigenvectors = np.load("PCA_DATA/eigenvectors.npy")
projected_train = np.load("PCA_DATA/projected_train.npy")

# ==== Parameters ====
threshold = 0.1  # غيّر دا للتحكم في متى نرفض التصنيف

# ==== Prediction & Scoring ====
correct = 0
predicted_labels = []
distances_scores = []

for i, test_face in enumerate(X_test):
    label, score = recognize_face(test_face, mean_face, eigenvectors, projected_train, y_train)


    predicted_labels.append(label)
    distances_scores.append(-score)  # negate علشان نستخدمه في الـ ROC كـ "score"

    if label == y_test[i]:
        correct += 1

print(f"Accuracy: {correct / len(y_test) * 100:.2f}%")




# ==== ROC Curve - One-vs-Rest (Manual Threshold Version) ====
def manual_roc_for_class(y_true_bin, scores, thresholds):

    fpr = []
    tpr = []
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        TP = np.sum((predictions == 1) & (y_true_bin == 1))
        FP = np.sum((predictions == 1) & (y_true_bin == 0))
        TN = np.sum((predictions == 0) & (y_true_bin == 0))
        FN = np.sum((predictions == 0) & (y_true_bin == 1))

        TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
        fpr.append(FPR)
        tpr.append(TPR)


    return np.array(fpr), np.array(tpr)


lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)
# print("y_test_bin is:/n",y_test_bin)
# print("lb.classes_ is:/n",lb.classes_)
predicted_scores = np.zeros((len(y_test), len(lb.classes_)))

# ==== Score each test face against training set ====
for i, test_face in tqdm(enumerate(X_test), total=len(X_test), desc="Scoring for ROC"):
    test_centered = test_face - mean_face
    test_proj = np.dot(test_centered, eigenvectors)
    distances = np.linalg.norm(projected_train - test_proj, axis=1)

    # Compute scores for each class
    for j, cls in enumerate(lb.classes_):
        class_norm_dists = distances[y_train == cls]
        min_dist = np.min(class_norm_dists)
        predicted_scores[i, j] = min_dist  # Store minimum distance (not negated)

# ==== Normalize scores to [0, 1] for ROC computation ====
# Convert distances to similarity scores: smaller distance -> higher score
max_dist = np.max(predicted_scores)
min_dist = np.min(predicted_scores)
if max_dist != min_dist:
    predicted_scores = (max_dist - predicted_scores) / (max_dist - min_dist)
else:
    predicted_scores = np.ones_like(predicted_scores)  # Avoid division by zero

# ==== Compute ROC curves for multiple thresholds ====
thresholds = np.linspace(0, 15, 500)  # Evaluate 100 thresholds from 0 to 1
plt.figure(figsize=(15, 15))

for i, cls in enumerate(lb.classes_):
    y_true_bin = y_test_bin[:, i]
    y_score = predicted_scores[:, i]


    # Compute ROC curve for this class
    fpr, tpr = manual_roc_for_class(y_true_bin, y_score, thresholds)

    # Plot ROC curve
    plt.plot(fpr, tpr, label=f"Class {lb.classes_[i]}")
    plt.legend(loc="upper right")


# Plot random guess line
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.title("ROC Curves (One-vs-Rest)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


