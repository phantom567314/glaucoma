import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from skimage.feature import greycomatrix, greycoprops

# Set the paths for the dataset
DATASET_PATH = 'path/to/your/dataset'
NORMAL_DIR = os.path.join(DATASET_PATH, 'normal')
GLAUCOMA_DIR = os.path.join(DATASET_PATH, 'glaucoma')

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def extract_features(image):
    # Resize image to a fixed size
    image = cv2.resize(image, (256, 256))
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 100, 200)
    
    # Calculate GLCM (Gray Level Co-occurrence Matrix) properties
    glcm = greycomatrix(image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]
    
    # Feature vector
    features = [contrast, dissimilarity, homogeneity, energy, correlation]
    
    return features

# Load images
normal_images = load_images_from_folder(NORMAL_DIR)
glaucoma_images = load_images_from_folder(GLAUCOMA_DIR)

# Extract features
normal_features = [extract_features(img) for img in normal_images]
glaucoma_features = [extract_features(img) for img in glaucoma_images]

# Create labels
normal_labels = [0] * len(normal_features)
glaucoma_labels = [1] * len(glaucoma_features)

# Combine features and labels
features = normal_features + glaucoma_features
labels = normal_labels + glaucoma_labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Evaluate the model
print('Classification Report')
print(classification_report(y_test, y_pred, target_names=['Normal', 'Glaucoma']))

print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))