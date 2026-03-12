import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, f1_score

np.random.seed(42)  # for reproducibility

# Hours spent driving (1 to 10 hours, 100 samples)
X = np.random.uniform(1, 10, 100).reshape(-1, 1)

Y = 2 * X.flatten() + 3 + np.random.normal(0, 1.5, 100)

model = LinearRegression()
model.fit(X, Y)

# Predictions
Y_pred = model.predict(X)

# R² Score
r2 = r2_score(Y, Y_pred)

# Convert regression → classification for F1 score
# Threshold: Risk >= 15 → High Risk
threshold = 15

Y_actual_class = (Y >= threshold).astype(int)
Y_pred_class = (Y_pred >= threshold).astype(int)

f1 = f1_score(Y_actual_class, Y_pred_class)

m = model.coef_[0]
c = model.intercept_

print("Slope (m):", m)
print("Intercept (c):", c)
print(f"Best Fit Line: y = {m:.2f}x + {c:.2f}")
print("R² Score:", r2)
print("F1 Score:", f1)

plt.scatter(X, Y, alpha=0.6, label="Actual Data")
plt.plot(X, Y_pred, label="Regression Line")
plt.xlabel("Hours Spent Driving")
plt.ylabel("Risk of Acute Backache")
plt.title("Linear Regression on Large Dataset")
plt.legend()
plt.show()
