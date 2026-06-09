import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report
)

from src.preprocessing.churn_preprocessing import (
    preprocess_churn_data
)


def train_churn_model():

    df = preprocess_churn_data()

    categorical_columns = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "Churn"
    ]

    encoder = LabelEncoder()

    for column in categorical_columns:
        df[column] = encoder.fit_transform(
            df[column]
        )

    X = df.drop(
        columns=["customerID", "Churn"]
    )

    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    print("\nAccuracy:")
    print(
        accuracy_score(
            y_test,
            predictions
        )
    )

    print("\nPrecision:")
    print(
        precision_score(
            y_test,
            predictions
        )
    )

    print("\nRecall:")
    print(
        recall_score(
            y_test,
            predictions
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions
        )
    )

    joblib.dump(
        model,
        "models/churn_model.pkl"
    )

    return model


if __name__ == "__main__":
    train_churn_model()