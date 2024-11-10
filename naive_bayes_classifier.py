import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder

# Define the dataset
data = {
    'Outlook': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast', 'overcast', 'rainy'],
    'Temperature': ['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool', 'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild'],
    'Humidity': ['high', 'high', 'high', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'high'],
    'Windy': ['false', 'true', 'false', 'false', 'false', 'true', 'true', 'false', 'false', 'false', 'true', 'true', 'false', 'true'],
    'Play': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encode the categorical features
label_encoders = {col: LabelEncoder() for col in df.columns}
for col in df.columns:
    df[col] = label_encoders[col].fit_transform(df[col])

# Split features and labels
X = df.drop('Play', axis=1)
y = df['Play']

# Train the Naive Bayes classifier
model = CategoricalNB()
model.fit(X, y)

# Define the test data with feature names
test_data = pd.DataFrame([[2, 0, 0, 1]], columns=['Outlook', 'Temperature', 'Humidity', 'Windy'])

# Make the prediction
prediction = model.predict(test_data)
play_decision = label_encoders['Play'].inverse_transform(prediction)
print(f"Prediction: {'Play' if play_decision[0] == 'yes' else 'Don\'t Play'}")
