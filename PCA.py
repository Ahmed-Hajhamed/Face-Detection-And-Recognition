import os
import cv2
import numpy as np

def load_images(dataset_path, image_shape=(100, 100)):
    X = []  # Image vectors
    y = []  # Labels
    subjects = []
    file_paths = []
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
                subjects.append(label)
                file_paths.append(img_path)
                print(f"Loaded {img_path} with label {label}")
    y = [subjects, file_paths]
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

def recognize_face(test_face, mean_face, eigenvectors, projected_faces, labels, distance_threshold=2500):
    test_centered = test_face - mean_face
    test_proj = np.dot(test_centered, eigenvectors)
    distances = np.linalg.norm(projected_faces - test_proj, axis=1)
    subjects = labels[0]
    file_paths = labels[1]
    # if np.min(distances) > distance_threshold:
    #     return None
    return subjects[np.argmin(distances)], file_paths[np.argmin(distances)]

# X, y = load_images("archive")
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
# X_train, y_train = load_images("dataset/train")
# X_test, y_test = load_images("dataset/test")

# np.save("X_train.npy", X_train)
# np.save("X_test.npy", X_test)
# np.save("y_train.npy", y_train)
# np.save("y_test.npy", y_test)
X_train = np.load("PCA_DATA/X_train.npy")
X_test = np.load("PCA_DATA/X_test.npy")
y_train = np.load("PCA_DATA/y_train.npy")
y_test = np.load("PCA_DATA/y_test.npy")

# mean_face, eigenvectors, X_train_centered = compute_pca(X_train, num_components=50)
# projected_train = project_faces(X_train_centered, eigenvectors)
# np.save("mean_face.npy", mean_face)
# np.save("eigenvectors.npy", eigenvectors)
# np.save("projected_train.npy", projected_train)
mean_face = np.load("PCA_DATA/mean_face.npy")
eigenvectors = np.load("PCA_DATA/eigenvectors.npy")
projected_train = np.load("PCA_DATA/projected_train.npy")

# Test
correct = 0
for i, test_face in enumerate(X_test):
    label = recognize_face(test_face, mean_face, eigenvectors, projected_train, y_train)[0]
    if label == y_test[0][i]:
        correct += 1

print(f"Accuracy: {correct / len(y_test[0]) * 100:.2f}%")
# test_path = "dataset/test/s35/9.pgm"
# test_face = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
# test_face = cv2.resize(test_face, (100, 100))
# test_face1 = test_face.flatten()
# label, result_path = recognize_face(test_face1, mean_face, eigenvectors, projected_train, y_train)
# hhhh = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
# hhhh = cv2.resize(hhhh, (500, 500))
# cv2.imshow("Test Face", hhhh)
# result = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
# result = cv2.resize(result, (500, 500))
# cv2.imshow("Recognized Face", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
