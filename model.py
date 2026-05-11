import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the data
data = pd.read_csv("crop.csv")

# Separate inputs and output
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the AI
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained AI
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")