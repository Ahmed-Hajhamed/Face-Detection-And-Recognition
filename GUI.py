import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QPushButton, QFileDialog, QScrollArea, QFrame, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPainter, QPen, QPolygon
from PyQt5.QtCore import Qt, QPoint

class NavigableArrowButton(QPushButton):
    """Custom button with arrow indicator for navigation"""
    def __init__(self, direction="right", parent=None):
        super().__init__(parent)
        self.direction = direction
        self.setFixedSize(40, 40)
        self.setStyleSheet("""
            QPushButton {
                background-color: #89B4FA;
                border-radius: 20px;
                border: 2px solid #74C7EC;
            }
            QPushButton:hover {
                background-color: #74C7EC;
            }
            QPushButton:pressed {
                background-color: #89DCEB;
            }
        """)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set arrow color
        painter.setPen(QPen(QColor("#1E1E2E"), 2))
        painter.setBrush(QColor("#1E1E2E"))
        
        # Calculate arrow points based on direction
        if self.direction == "right":
            points = QPolygon([
                QPoint(15, 20),
                QPoint(25, 13),
                QPoint(25, 27)
            ])
        elif self.direction == "left":
            points = QPolygon([
                QPoint(25, 20),
                QPoint(15, 13),
                QPoint(15, 27)
            ])
        
        painter.drawPolygon(points)

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition with PCA")
        self.setGeometry(100, 100, 1400, 800)
        
        # Initialize variables
        self.original_image = None
        self.processed_image = None
        self.pca_applied = False
        self.current_neighbor_index = 0
        self.neighbors = []
        
        # Initialize UI
        self.initUI()
    
    def initUI(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create image display area (taking up most of the window)
        image_display_widget = QWidget()
        image_display_layout = QHBoxLayout(image_display_widget)
        image_display_layout.setContentsMargins(0, 0, 0, 0)
        image_display_layout.setSpacing(10)
        
        # Original image panel
        original_panel = QFrame()
        original_panel.setStyleSheet("background-color: #1E1E2E; border-radius: 5px;")
        original_layout = QVBoxLayout(original_panel)
        
        original_title = QLabel("Original Image")
        original_title.setAlignment(Qt.AlignCenter)
        original_title.setFont(QFont("Arial", 14, QFont.Bold))
        original_title.setStyleSheet("color: #CDD6F4;")
        original_layout.addWidget(original_title)
        
        self.original_image_label = QLabel("No image loaded")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(500, 500)
        self.original_image_label.setStyleSheet("border: 1px solid #45475A; background-color: #181825; color: #CDD6F4;")
        
        original_scroll = QScrollArea()
        original_scroll.setWidgetResizable(True)
        original_scroll.setWidget(self.original_image_label)
        original_scroll.setStyleSheet("border: none;")
        original_layout.addWidget(original_scroll)
        
        image_display_layout.addWidget(original_panel)
        
        # PCA result panel with navigation arrows
        pca_container = QWidget()
        pca_container_layout = QHBoxLayout(pca_container)
        pca_container_layout.setContentsMargins(0, 0, 0, 0)
        pca_container_layout.setSpacing(5)
        
        # Left arrow
        self.prev_neighbor_btn = NavigableArrowButton("left")
        self.prev_neighbor_btn.clicked.connect(self.show_prev_neighbor)
        self.prev_neighbor_btn.setEnabled(False)
        pca_container_layout.addWidget(self.prev_neighbor_btn)
        
        # PCA panel
        pca_panel = QFrame()
        pca_panel.setStyleSheet("background-color: #1E1E2E; border-radius: 5px;")
        pca_layout = QVBoxLayout(pca_panel)
        
        pca_title = QLabel("PCA Result")
        pca_title.setAlignment(Qt.AlignCenter)
        pca_title.setFont(QFont("Arial", 14, QFont.Bold))
        pca_title.setStyleSheet("color: #CDD6F4;")
        pca_layout.addWidget(pca_title)
        
        self.pca_image_label = QLabel("PCA result will appear here")
        self.pca_image_label.setAlignment(Qt.AlignCenter)
        self.pca_image_label.setMinimumSize(500, 500)
        self.pca_image_label.setStyleSheet("border: 1px solid #45475A; background-color: #181825; color: #CDD6F4;")
        
        pca_scroll = QScrollArea()
        pca_scroll.setWidgetResizable(True)
        pca_scroll.setWidget(self.pca_image_label)
        pca_scroll.setStyleSheet("border: none;")
        pca_layout.addWidget(pca_scroll)
        
        # Neighbor information
        self.neighbor_info = QLabel("No neighbors found")
        self.neighbor_info.setAlignment(Qt.AlignCenter)
        self.neighbor_info.setStyleSheet("color: #CDD6F4;")
        pca_layout.addWidget(self.neighbor_info)
        
        pca_container_layout.addWidget(pca_panel)
        
        # Right arrow
        self.next_neighbor_btn = NavigableArrowButton("right")
        self.next_neighbor_btn.clicked.connect(self.show_next_neighbor)
        self.next_neighbor_btn.setEnabled(False)
        pca_container_layout.addWidget(self.next_neighbor_btn)
        
        image_display_layout.addWidget(pca_container)
        
        main_layout.addWidget(image_display_widget, 85)  # 85% of the space
        
        # Create buttons panel at the bottom
        buttons_panel = QWidget()
        buttons_layout = QHBoxLayout(buttons_panel)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(20)
        
        # Define button style
        button_style = """
            QPushButton {
                background-color: #89B4FA;
                color: #1E1E2E;
                border: none;
                border-radius: 5px;
                padding: 12px 25px;
                font-size: 16px;
                font-weight: bold;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #74C7EC;
            }
            QPushButton:pressed {
                background-color: #89DCEB;
            }
            QPushButton:disabled {
                background-color: #45475A;
                color: #7F849C;
            }
        """
        
        # Create buttons
        upload_btn = QPushButton("Upload Image")
        upload_btn.setStyleSheet(button_style)
        upload_btn.clicked.connect(self.load_image)
        buttons_layout.addWidget(upload_btn)
        
        detect_btn = QPushButton("Detect Face")
        detect_btn.setStyleSheet(button_style)
        detect_btn.clicked.connect(self.detect_face)
        buttons_layout.addWidget(detect_btn)
        
        pca_btn = QPushButton("Compute PCA")
        pca_btn.setStyleSheet(button_style)
        pca_btn.clicked.connect(self.apply_pca)
        buttons_layout.addWidget(pca_btn)
        
        save_btn = QPushButton("Save Result")
        save_btn.setStyleSheet(button_style)
        save_btn.clicked.connect(self.save_results)
        buttons_layout.addWidget(save_btn)
        
        main_layout.addWidget(buttons_panel, 15)  # 15% of the space
        
        # Status bar
        self.status_bar = QLabel("Ready")
        self.status_bar.setStyleSheet("background-color: #313244; color: #CDD6F4; padding: 5px; border-radius: 3px;")
        main_layout.addWidget(self.status_bar)
        
        # Apply final stylesheet to the entire window
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1E1E2E;
                color: #CDD6F4;
            }
            QScrollBar:vertical {
                border: none;
                background: #313244;
                width: 10px;
                margin: 0px 0px 0px 0px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #45475A;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        self.setCentralWidget(central_widget)
    
    def load_image(self):
        """Load an image file and display it in the original image panel"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", 
                                                "Image Files (*.png *.jpg *.jpeg *.bmp *.pgm)")
        if file_path:
            try:
                # Load and display the image
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    raise ValueError("Could not read image")
                
                # Reset states
                self.pca_applied = False
                self.processed_image = None
                self.neighbors = []
                self.current_neighbor_index = 0
                self.prev_neighbor_btn.setEnabled(False)
                self.next_neighbor_btn.setEnabled(False)
                
                # Display the image
                self.display_image(self.original_image, self.original_image_label)
                self.pca_image_label.setText("Detect face and compute PCA")
                self.neighbor_info.setText("No neighbors found")
                
                self.status_bar.setText(f"Loaded image: {os.path.basename(file_path)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading image: {str(e)}")
                self.status_bar.setText("Error loading image")

    def display_image(self, image, label):
        """Display an image on a QLabel with proper scaling"""
        if image is None:
            return
            
        # Convert to RGB for display
        if len(image.shape) == 2:  # Grayscale
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        h, w = rgb_image.shape[:2]
        bytes_per_line = 3 * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale to fit label while maintaining aspect ratio
        label.setPixmap(pixmap.scaled(label.width(), label.height(), 
                                    Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def detect_face(self):
        # This is a placeholder - implement your face detection logic here
        self.status_bar.setText("Face detected!")
    
    def apply_pca(self):
        # This is a placeholder - implement your PCA application logic here
        
        # Enable navigation buttons (for demonstration)
        self.prev_neighbor_btn.setEnabled(False)
        self.next_neighbor_btn.setEnabled(True)
        self.status_bar.setText("PCA applied - face processed")
    
    def show_next_neighbor(self):
        """Show the next neighbor in the PCA results"""
        if not self.neighbors or self.current_neighbor_index >= len(self.neighbors) - 1:
            return
        
        self.current_neighbor_index += 1
        self.display_current_neighbor()
        
        # Update navigation buttons
        self.prev_neighbor_btn.setEnabled(True)
        self.next_neighbor_btn.setEnabled(self.current_neighbor_index < len(self.neighbors) - 1)
        
        self.status_bar.setText(f"Showing neighbor {self.current_neighbor_index + 1}/{len(self.neighbors)}")

    def show_prev_neighbor(self):
        """Show the previous neighbor in the PCA results"""
        if not self.neighbors or self.current_neighbor_index <= 0:
            return
        
        self.current_neighbor_index -= 1
        self.display_current_neighbor()
        
        # Update navigation buttons
        self.prev_neighbor_btn.setEnabled(self.current_neighbor_index > 0)
        self.next_neighbor_btn.setEnabled(True)
        
        self.status_bar.setText(f"Showing neighbor {self.current_neighbor_index + 1}/{len(self.neighbors)}")

    def display_current_neighbor(self):
        """Helper function to display the current neighbor"""
        if not self.neighbors or self.current_neighbor_index >= len(self.neighbors):
            return
        
        neighbor_image = self.neighbors[self.current_neighbor_index]
        
        # Display the neighbor image
        self.display_image(neighbor_image, self.pca_image_label)
        
        # Update neighbor info
        self.neighbor_info.setText(f"Neighbor {self.current_neighbor_index + 1}/{len(self.neighbors)}")

    def save_results(self):
        """Save the currently displayed PCA result"""
        if not self.pca_applied or self.pca_image_label.pixmap() is None:
            QMessageBox.warning(self, "No Results", "No PCA results to save")
            return
            
        # Create directory for results if it doesn't exist
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        try:
            # Get default name based on current neighbor
            default_name = f"pca_result_{self.current_neighbor_index + 1}.png"
            
            # Show save dialog
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Save Result", 
                os.path.join(results_dir, default_name),
                "Image Files (*.png *.jpg *.jpeg)"
            )
            
            if file_path:
                # Get the current pixmap and save as image
                pixmap = self.pca_image_label.pixmap()
                pixmap.save(file_path)
                
                self.status_bar.setText(f"Result saved to: {file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Error saving result: {str(e)}")
            self.status_bar.setText("Error saving result")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())