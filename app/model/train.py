import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# Sample training data (you can replace with your dataset)
data = {
    "study_hours": [1, 2, 3, 4, 5],
    "attendance": [60, 70, 80, 90, 95],
    "previous_score": [40, 50, 60, 70, 80],
    "final_score": [45, 55, 65, 75, 85]
}

df = pd.DataFrame(data)

X = df[["study_hours", "attendance", "previous_score"]]
y = df["final_score"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

# Save model
joblib.dump(model, "model/student_model.pkl")

print("✅ Model recreated successfully!")
