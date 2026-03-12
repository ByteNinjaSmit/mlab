import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("../data/yellow_tripdata_2015-01.csv")

print("Total Records:", len(df))

# Sample for performance
df = df.sample(n=10000, random_state=5)

df = df.dropna()

df = df[df["trip_distance"] > 0]
df = df[df["passenger_count"] > 0]

df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
df["pickup_day"] = df["tpep_pickup_datetime"].dt.day
df["pickup_weekday"] = df["tpep_pickup_datetime"].dt.weekday

df["trip_duration"] = (
    (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"])
    .dt.total_seconds() / 60
)


y = df["total_amount"]


numeric_features = [
    "trip_distance",
    "passenger_count",
    "pickup_hour",
    "pickup_day",
    "pickup_weekday",
    "trip_duration",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude"
]

categorical_features = [
    "payment_type",
    "RateCodeID"
]

X = df[numeric_features + categorical_features]


preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01)
}

best_y_pred = None


for name, model in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n==============================")
    print(name)
    print("MSE:", round(mse, 3))
    print("R²:", round(r2, 3))

    if name == "Linear Regression":
        best_y_pred = y_pred


plt.figure()
plt.scatter(y_test, best_y_pred, alpha=0.4)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel("Actual Total Amount")
plt.ylabel("Predicted Total Amount")
plt.title("Actual vs Predicted (Linear Regression)")
plt.show()
