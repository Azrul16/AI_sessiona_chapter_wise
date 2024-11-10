import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

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

# Train the Decision Tree classifier
model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
model.fit(X, y)

# Plot the Decision Tree with improvements
plt.figure(figsize=(16, 10))
tree.plot_tree(
    model,
    feature_names=X.columns,
    class_names=['No', 'Yes'],
    filled=True,
    rounded=True,
    fontsize=12,
    impurity=True,
    precision=2
)

# Add a title and improve aesthetics
plt.title("Improved Decision Tree for Play Prediction", fontsize=20)
plt.show()
