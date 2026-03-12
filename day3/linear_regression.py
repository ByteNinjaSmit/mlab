import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("../data/yellow_tripdata_2015-01.csv")

df = df[['trip_distance', 'fare_amount']]

df.dropna(inplace=True)

df = df[
    (df['trip_distance'] > 0.5) & (df['trip_distance'] < 50) &
    (df['fare_amount'] > 3) & (df['fare_amount'] < 200)
]

X = df[['trip_distance']]
y = df['fare_amount']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)

m = model.coef_[0]
c = model.intercept_

print("--------- Linear Regression Results ---------")
print(f"Slope (m): {m}")
print(f"Intercept (c): {c}")
print(f"Regression Equation: y = {m:.2f}x + {c:.2f}")
print(f"R² Score: {r2}")
print("---------------------------------------------")

plt.scatter(X_test, y_test, alpha=0.4, label="Actual Fare")
plt.scatter(X_test, y_pred, alpha=0.4, label="Predicted Fare")
plt.xlabel("Trip Distance (miles)")
plt.ylabel("Fare Amount")
plt.title("Linear Regression: Trip Distance vs Fare Amount")
plt.legend()
plt.show()
