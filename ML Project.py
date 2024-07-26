import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('data_human_activities.csv', header=None, names=['user', 'activity', 'timestamp', 'x', 'y', 'z'])

# Display the first few rows of the dataset
print(df.head())

# Check the shape of the dataset
print(f"Dataset Shape: {df.shape}")

# Check for missing values
print(df.isnull().sum())

# Display unique activities
print("Unique Activities:", df['activity'].unique())

# Get basic statistics
print(df.describe())

# Plot the distribution of activities
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='activity')
plt.title('Activity Distribution')
plt.xticks(rotation=45)
plt.show()

# Plot a sample of accelerometer data
sample_user = df[df['user'] == 1]
plt.figure(figsize=(12, 6))
plt.plot(sample_user['x'][:200], label='X-axis')
plt.plot(sample_user['y'][:200], label='Y-axis')
plt.plot(sample_user['z'][:200], label='Z-axis')
plt.title('Accelerometer Data for a Sample User')
plt.xlabel('Sample Number')
plt.ylabel('Acceleration (g)')
plt.legend()
plt.show()

# Convert activity labels to numerical values
activity_map = {
    'Walking': 0,
    'Jogging': 1,
    'Sitting': 2,
    'Standing': 3,
    'Upstairs': 4,
    'Downstairs': 5
}
df['activity'] = df['activity'].map(activity_map)

# Define features and target
X = df[['x', 'y', 'z']]
y = df['activity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=list(activity_map.keys()))

# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=list(activity_map.keys()), yticklabels=list(activity_map.keys()))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save the model
joblib.dump(model, 'activity_recognition_model.pkl')
