import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report

# Constants
IMG_SIZE = (100, 100)
N_COMPONENTS = 30

def load_images(folder, is_train=True):
    images, labels = [], []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".pgm"):
            print(f"Loading {filename}")
            parts = filename.split("_")
            if len(parts) < 2:
                continue  # skip malformed filenames

            subject_bin_from_file = parts[0]  # first part is subject ID (like 0000)
            print(f"Subject ID: {subject_bin_from_file}")
            angle = parts[1].split(".")[0]    # second part is random number/angle
            
            filepath = os.path.join(folder, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            images.append(img.flatten())
            labels.append(subject_bin_from_file)
    
    return np.array(images), np.array(labels)


# Compute PCA
def compute_pca(X_train):
    pca = PCA(n_components=N_COMPONENTS)
    X_pca = pca.fit_transform(X_train)
    return pca, X_pca

# Train KNN
def train_knn(X_pca, y_train):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_pca, y_train)
    return knn

# Evaluate and plot ROC
def evaluate_and_plot(y_true, y_pred, y_score):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    # For ROC: convert labels to integers
    y_true_int = np.array([int(y) for y in y_true])
    y_score_int = np.array([int(y) for y in y_score])

    fpr, tpr, _ = roc_curve(y_true_int, y_score_int, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

# Main pipeline
def main():
    train_dir = "MIT-CBCL-facerec-database\\training-synthetic"
    test_dir = "MIT-CBCL-facerec-database\\test"

    # Load datasets
    X_train, y_train = load_images(train_dir)
    X_test, y_test = load_images(test_dir, is_train=False)
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)

    print(f"Loaded {len(X_train)} training and {len(X_test)} testing samples.")

    # Compute PCA on training
    pca, X_train_pca = compute_pca(X_train)

    # Project test samples
    X_test_pca = pca.transform(X_test)

    # Train and predict
    knn = train_knn(X_train_pca, y_train)
    y_pred = knn.predict(X_test_pca)

    # Evaluate
    evaluate_and_plot(y_test, y_pred, y_pred)

if __name__ == "__main__":
    main()
