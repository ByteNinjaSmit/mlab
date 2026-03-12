# Ass 3

# Decision Tree Classifier for Saree Purchase Prediction

# Import required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# -----------------------------------------
# Step 1: Create Saree Shop Dataset
# -----------------------------------------

data = {
    'Age': ['<=30','<=30','31-40','>40','>40','>40','31-40','<=30','<=30','>40','<=30','31-40','31-40','>40'],
    'Income': ['High','High','High','Medium','Low','Low','Low','Medium','Low','Medium','Medium','Medium','High','Medium'],
    'Student': ['No','No','No','No','Yes','Yes','Yes','No','Yes','Yes','Yes','No','Yes','No'],
    'Credit_Rating': ['Fair','Excellent','Fair','Fair','Fair','Excellent','Excellent','Fair','Fair','Fair','Excellent','Excellent','Fair','Excellent'],
    'Buys_Saree': ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
}

df = pd.DataFrame(data)

print("Original Dataset:\n")
print(df)

# -----------------------------------------
# Step 2: Convert Categorical Data
# -----------------------------------------

le = LabelEncoder()

for column in df.columns:
    df[column] = le.fit_transform(df[column])

print("\nEncoded Dataset:\n")
print(df)

# -----------------------------------------
# Step 3: Separate Features and Target
# -----------------------------------------

X = df.drop("Buys_Saree", axis=1)
y = df["Buys_Saree"]

# -----------------------------------------
# Step 4: Train Decision Tree Model
# -----------------------------------------

model = DecisionTreeClassifier(criterion='entropy')

model.fit(X, y)

# -----------------------------------------
# Step 5: Predict Example Customer
# -----------------------------------------

sample = [[0, 1, 1, 0]]  # Example encoded input

prediction = model.predict(sample)

print("\nPrediction for sample customer:", prediction)

# -----------------------------------------
# Step 6: Visualize Decision Tree
# -----------------------------------------

plt.figure(figsize=(12,8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No","Yes"],
    filled=True
)

plt.title("Decision Tree for Saree Purchase")
plt.show()