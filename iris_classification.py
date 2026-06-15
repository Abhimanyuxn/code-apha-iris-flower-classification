# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# STEP 1: Load the Dataset
# =========================

# Read the Iris dataset from CSV file
df = pd.read_csv("Iris.csv")

# Display first 5 records
print("First 5 Records:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# =========================
# STEP 2: Select Features and Target
# =========================

# Input features (flower measurements)
X = df[['SepalLengthCm',
        'SepalWidthCm',
        'PetalLengthCm',
        'PetalWidthCm']]

# Target variable (flower species)
y = df['Species']

# =========================
# STEP 3: Split Dataset
# =========================

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =========================
# STEP 4: Create and Train Model
# =========================

# Create Decision Tree Classifier
model = DecisionTreeClassifier()

# Train the model using training data
model.fit(X_train, y_train)

# =========================
# STEP 5: Make Predictions
# =========================

# Predict species for test data
y_pred = model.predict(X_test)

# =========================
# STEP 6: Evaluate Model
# =========================

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

# Generate detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Generate confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# STEP 7: Predict New Flower
# =========================

# Example flower measurements
sample_flower = [[5.1, 3.5, 1.4, 0.2]]

# Predict species
prediction = model.predict(sample_flower)

print("\nPredicted Species:")
print(prediction[0])