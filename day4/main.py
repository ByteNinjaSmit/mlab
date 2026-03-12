"""
	Assignment on Decision Tree Classifier: A dataset collected in a Cloth shop showing details of customers and whether or not they responded to a special offer to buy a new Sarry is shown in table below. Use this dataset to build a decision tree, with Buys
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree # Added plot_tree here
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Create the dataset
data = {
    'Age': ['< 21', '< 21', '21-35', '> 35', '> 35', '> 35', '21-35', '< 21', '< 21', '> 35', '< 21', '21-35', '21-35', '> 35'],
    'Income': ['High', 'High', 'High', 'Medium', 'Low', 'Low', 'Low', 'Medium', 'Low', 'Medium', 'Medium', 'Medium', 'High', 'Medium'],
    'Gender': ['Male', 'Male', 'Male', 'Male', 'Female', 'Female', 'Female', 'Male', 'Female', 'Female', 'Female', 'Male', 'Female', 'Male'],
    'Marital Status': ['Single', 'Married', 'Single', 'Single', 'Single', 'Married', 'Married', 'Single', 'Married', 'Single', 'Married', 'Married', 'Single', 'Married'],
    'Buys': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}
df = pd.DataFrame(data)

# 2. Encode categorical variables
le_age = LabelEncoder()
le_income = LabelEncoder()
le_gender = LabelEncoder()
le_marital = LabelEncoder()
le_buys = LabelEncoder()

df['Age_n'] = le_age.fit_transform(df['Age'])
df['Income_n'] = le_income.fit_transform(df['Income'])
df['Gender_n'] = le_gender.fit_transform(df['Gender'])
df['Marital_n'] = le_marital.fit_transform(df['Marital Status'])
df['Buys_n'] = le_buys.fit_transform(df['Buys'])

# 3. Define Features (X) and Target (y)
X = df[['Age_n', 'Income_n', 'Gender_n', 'Marital_n']]
y = df['Buys_n']

# 4. Initialize and train the Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X, y)

# 5. Predict on the training data to evaluate the model
y_pred = clf.predict(X)

# --- EVALUATION METRICS ---

# Calculate Metrics
cm = confusion_matrix(y, y_pred)
acc = accuracy_score(y, y_pred)
report = classification_report(y, y_pred, target_names=le_buys.classes_)

# Print the Reports
print("--- Model Evaluation Metrics ---\n")
print(f"Overall Accuracy: {acc * 100:.2f}%\n")
print("Classification Report:")
print(report)

# 6. Predict the specific test data from the assignment
test_data = [[
    le_age.transform(['< 21'])[0],
    le_income.transform(['Low'])[0],
    le_gender.transform(['Female'])[0],
    le_marital.transform(['Married'])[0]
]]
prediction_encoded = clf.predict(test_data)
prediction_label = le_buys.inverse_transform(prediction_encoded)

print("\n--- Specific Test Data Prediction ---")
print(f"Test Data [Age < 21, Income = Low, Gender = Female, Marital Status = Married]")
print(f"Prediction: Buys = {prediction_label[0]}")

# 7. Visualize the Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_buys.classes_, 
            yticklabels=le_buys.classes_)
plt.title('Confusion Matrix for Customer Purchase Decision')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 8. Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(clf, 
          feature_names=['Age', 'Income', 'Gender', 'Marital Status'], 
          class_names=le_buys.classes_, 
          filled=True, 
          rounded=True,
          fontsize=10)
plt.title('Decision Tree Visualization')
plt.show()