import sys
import os
import cv2
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMessageBox
from GUI import FaceRecognitionApp
from PCA import load_images, compute_pca, project_faces, recognize_face
from sklearn.model_selection import train_test_split


def load_pca_parameters(folder="PCA_DATA"):
    """Load PCA parameters from CSV files"""
    try:
        if not os.path.exists(folder):
            return None
            
        mean_face = np.load(os.path.join(folder, "mean_face.npy"))
        eigenvectors = np.load(os.path.join(folder, "eigenvectors.npy"))
        projected_faces = np.load(os.path.join(folder, "projected_train.npy"))
        labels = np.load(os.path.join(folder, "y_train.npy"))
        # dataset_images = np.load(os.path.join(folder, "dataset_images.npy"))
        
        return {
            'mean_face': mean_face,
            'eigenvectors': eigenvectors,
            'projected_faces': projected_faces,
            'labels': labels
        }
   
    except Exception as e:
        print(f"Error loading PCA parameters: {str(e)}")
        return None


class MainApplication(FaceRecognitionApp):
    def __init__(self):
        super().__init__()
        
        self.mean_face = None
        self.eigenvectors = None
        self.projected_faces = None
        self.labels = None
        self.image_shape = (100, 100)
        
        # Load the dataset and compute PCA
        self.load_dataset_and_pca()
    
    def load_dataset_and_pca(self):
        """Load dataset and compute PCA (or load from cache if available)"""
        try:
            self.status_bar.setText("Loading dataset and computing PCA...")
            QApplication.processEvents()

            cached_data = load_pca_parameters()
            
            if cached_data:
                self.mean_face = cached_data['mean_face']
                self.eigenvectors = cached_data['eigenvectors']
                self.projected_faces = cached_data['projected_faces']
                self.labels = cached_data['labels']
                self.status_bar.setText("PCA parameters loaded from cache")
                return
            else:
                self.status_bar.setText("PCA computed but caching failed")
        except Exception as e:
            self.status_bar.setText("Error in PCA computation")
            QMessageBox.critical(self, "Error", f"Error in PCA computation: {str(e)}")
    
    def detect_face(self):
        """Detect faces in the loaded image"""
        if self.original_image is None:
            self.status_bar.setText("No image loaded to detect faces")
            return
            
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

            # Load Haar Cascade for face detection
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect face(s)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            if len(faces) == 0:
                self.status_bar.setText("No faces detected")
                QMessageBox.information(self, "No Faces", "No faces were detected in the image")
                return
                
            self.processed_image = self.original_image.copy()
            
            for (x, y, w, h) in faces:
                cv2.rectangle(self.processed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            self.display_image(self.processed_image, self.original_image_label)
            
            self.detected_face = gray[y:y+h, x:x+w]
            
            self.status_bar.setText(f"Detected 1 face - ready for PCA")
            
        except Exception as e:
            self.status_bar.setText("Face detection failed")
            QMessageBox.critical(self, "Error", f"Face detection error: {str(e)}")
    
    def load_neighbors(self, result_path):
        result = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
        result = cv2.resize(result, (500, 500))
        return result
    
    def apply_pca(self):
        """Apply PCA to the detected face and find similar faces"""
        try:
            self.status_bar.setText("Processing face with PCA...")
            QApplication.processEvents()  # Update UI
            self.neighbors = []
            # Prepare the detected face
            image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(image, self.image_shape)
            face_vector = face_resized.flatten()
            
            # Recognize the face and find similar faces
            result = recognize_face(
                                                face_vector, 
                                                self.mean_face, 
                                                self.eigenvectors, 
                                                self.projected_faces, 
                                                self.labels
                                            )
            for recognized_label, result_path in result:
                result_path = result_path.replace("\\", "/")
                self.neighbors.append(self.load_neighbors(result_path))
            
            # Reset neighbor navigation
            self.current_neighbor_index = 0
            self.prev_neighbor_btn.setEnabled(False)
            self.next_neighbor_btn.setEnabled(len(self.neighbors) > 1)
            
            # Display the first neighbor
            if self.neighbors:
                self.display_current_neighbor()
                self.neighbor_info.setText(
                    f"Recognized as {recognized_label} ({len(self.neighbors)} samples)")
                self.status_bar.setText(
                    f"PCA complete - recognized as {recognized_label}")
                self.pca_applied = True
            else:
                self.status_bar.setText("No matching faces found in dataset")
                
        except Exception as e:
            self.status_bar.setText("PCA processing failed")
            QMessageBox.critical(self, "Error", f"PCA error: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApplication()
    window.show()
    sys.exit(app.exec_())