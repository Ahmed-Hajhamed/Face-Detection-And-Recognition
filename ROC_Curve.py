import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

# ==== Load Data ====
X_train = np.load("PCA_DATA/X_train.npy")
X_test = np.load("PCA_DATA/X_test.npy")
y_train = np.load("PCA_DATA/y_train.npy")[0]
y_test = np.load("PCA_DATA/y_test.npy")[0]
mean_face = np.load("PCA_DATA/mean_face.npy")
eigenvectors = np.load("PCA_DATA/eigenvectors.npy")
projected_train = np.load("PCA_DATA/projected_train.npy")

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
        FPR = FP / (FP + TN) if  (FP + TN) != 0 else 0
        fpr.append(FPR)
        tpr.append(TPR)

    return np.array(fpr), np.array(tpr)


lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)
print(y_test_bin.shape)
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
thresholds = np.linspace(0, 1, 500)  # Evaluate 100 thresholds from 0 to 1
plt.figure(figsize=(15, 15))

for i, cls in enumerate(lb.classes_):
    y_true_bin = y_test_bin[:, i]     
    y_score = predicted_scores[:, i]
    if i == 0:
        print(y_true_bin)
        print(y_score)
   


    # Compute ROC curve for this class
    fpr, tpr = manual_roc_for_class(y_true_bin, y_score, thresholds)

    # Plot ROC curve
    plt.plot(fpr, tpr, label=f"Class {lb.classes_[i]}")
    plt.legend(loc="lower right")


# Plot random guess line
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.title("ROC Curves (One-vs-Rest)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
# plt.savefig("ROC_Curve.png")
plt.tight_layout()
plt.show()


