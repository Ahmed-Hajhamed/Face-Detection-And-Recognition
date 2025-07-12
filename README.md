# Face-Detection-And-Recognition

1. Introduction
This report details the implementation of a face detection and recognition system using Python. The project utilizes a Haar cascade classifier for detecting faces within images and employs Principal Component Analysis (PCA) and Eigen analysis, implemented from scratch, for recognizing detected faces. The dataset comprises images of 40 distinct subjects, with 10 images per subject, split into a 60% training set and a 40% testing set.
2. Methodology
The core of this project involves two main components: face detection and face recognition.
2.1 Face Detection
Face detection is performed using OpenCV's implementation of the Haar Cascade Classifier. A pre-trained cascade specifically for frontal faces is utilized. This method involves applying a series of filters (cascades) to an image at multiple scales. Areas that pass all filter stages are likely to contain a face.
•	Pseudocode for Face Detection:
2.2 Face Recognition using PCA and Eigen Analysis
Face recognition is implemented from scratch using PCA. In Eigenface, these principal components represent the most significant variations in the training face images.
The process involves the following steps:
a. Prepare Data: All training images are flattened into 1D vectors and stacked to form a data matrix.
b. Calculate Mean Face: Compute the average vector of all training face vectors.
c. Subtract Mean: Subtract the mean face from each training face vector to obtain the mean-centered data matrix.
d. Calculate Covariance Matrix: Compute the covariance matrix of the mean-centered data matrix.
e. Calculate Eigenvectors and Eigenvalues: Compute the eigenvectors and eigenvalues of the covariance matrix. These eigenvectors represent the "Eigenfaces."
f. Select Principal Components: Sort eigenvectors by their corresponding eigenvalues in descending order and select the top `n` eigenvectors (where `n` is the number of PCA components, specified as 50). These form the basis of the lower-dimensional space.
g. Project Training Data: Project the mean-centered training face vectors onto the selected Eigenfaces to obtain the Eigenface coefficients (weights) for each training image.
h. Recognize New Face: For a new, unknown face image:
- Flatten and resize to 100x100.
- Subtract the mean face calculated during training.
- Project the mean-centered test face vector onto the selected Eigenfaces to obtain its Eigenface coefficients.
- Compare the coefficients of the test face to the coefficients of all training faces using Euclidean distance.
- The training face with the minimum distance is considered the closest match. The identity of the test face is assigned based on the identity of the closest training face.

2.3 Dataset Preparation and Split
The dataset consists of 40 subjects, each with 10 images in PGM format. Before processing, all images are resized to 100x100 pixels. The dataset is split into a training set (6 images per subject, 60%) and a testing set (4 images per subject, 40%).


3. Implementation Details
Primary libraries used include:
- OpenCV (`cv2`): For image loading, resizing, face detection (using Haar cascades), and drawing rectangles.
- NumPy: For numerical operations, matrix manipulations, and calculating eigenvalues and eigenvectors.
- PyQt5: Used to build the graphical user interface.
The GUI provides the following functionalities via four buttons:
1. Detect Faces: Loads an image, applies the Haar cascade classifier, and displays the image with rectangles drawn around detected faces.
2. Recognize Face: If faces are detected, it takes the detected face region(s), performs PCA-based recognition against the training set, and displays the closest 4 matching faces from the training data. Buttons are provided to scroll through these neighbors.
3. Load Image: Allows the user to select and load an image file.
4. Save Results: Saves the results of face detection to a PNG file.
GUI Snapshot:
 
5. Results
     
a)	Subject 39	   b) First 		                c) Second 		      d) Third 		  e) Fourth
Test Image	  Neighbor	  		Neighbor		       Neighbor		     Neighbor
     
a)	Subject 2		   b) First 		                c) Second 		      d) Third 		  e) Fourth
Test Image	  Neighbor	  		Neighbor		       Neighbor		     Neighbor
     
a)	Subject 36	   b) First 		                c) Second 		      d) Third 		  e) Fourth
Test Image	  Neighbor	  		Neighbor		       Neighbor		     Neighbor
     
a)	Subject 14	   b) First 		                c) Second 		      d) Third 		  e) Fourth
Test Image	  Neighbor	  		Neighbor		       Neighbor		     Neighbor


•	ROC Curve:
 
•	Analysis:	
- Face detection using OpenCV is instant and accurate for all cases.
- PCA-based faced recognition is fast and performed well on different subjects with variations in angles, emotions and with glasses on and off.

•	Accuracy:
	The number of PCA components used was 50, which produced 93.75% overall accuracy. When reducing components to 10, accuracy dropped slightly to 90%.

•	Team Members: (Team 17)
1.	Ahmed Adil
2.	Ahmed Etman
3.	Mohamed Ahmed
4.	Zeyad Wael
•	You are highly welcome to view the GitHub Repository.
