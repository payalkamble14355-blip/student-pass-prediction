import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("student_data.csv")

# Features and target
X = data[['study_hours', 'attendance', 'previous_scores']]
y = data['result']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Results
print("Predictions:", predictions)
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# Custom prediction
print("\n--- Custom Prediction ---")
new_student = [[6, 80, 72]]  # study_hours, attendance, previous_scores
result = model.predict(new_student)
print("Student Result:", result[0])
