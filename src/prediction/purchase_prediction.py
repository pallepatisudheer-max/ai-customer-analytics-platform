import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from src.preprocessing.customer_preprocessing import (
    preprocess_customer_data
)


def train_purchase_prediction():

    df = preprocess_customer_data()

    features = [
        "Age",
        "Income",
        "Kidhome",
        "Teenhome",
        "Recency",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "NumWebVisitsMonth"
    ]

    X = df[features]

    y = df["Total_Spending"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nR2 Score:")
    print(r2_score(y_test, predictions))

    print("\nMean Absolute Error:")
    print(mean_absolute_error(y_test, predictions))

    joblib.dump(
        model,
        "models/purchase_model.pkl"
    )

    return model


if __name__ == "__main__":
    train_purchase_prediction()