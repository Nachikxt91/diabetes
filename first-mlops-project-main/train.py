import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# ✅ Load local dataset (file is in current directory)
file_path = "diabetesP.csv"
df = pd.read_csv(file_path)

print("✅ Columns:", df.columns.tolist())  # Debug print
print("✅ Sample data:\n", df.head())      # Optional: show first few rows

# Prepare data
x = df[["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]]
y = df["Outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save
joblib.dump(model, "diabetes_model.pkl")
print("✅ Model saved as diabetes_model.pkl")
