import pandas as pd
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.preprocessing.customer_preprocessing import (
    preprocess_customer_data
)


def train_segmentation_model():

    df = preprocess_customer_data()

    features = [
        "Income",
        "Age",
        "Total_Spending"
    ]

    X = df[features]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = model.fit_predict(X_scaled)

    joblib.dump(
        model,
        "models/segmentation_model.pkl"
    )

    joblib.dump(
        scaler,
        "models/segmentation_scaler.pkl"
    )

    print("\nCluster Counts:")
    print(df["Cluster"].value_counts())

    return df


if __name__ == "__main__":

    segmented_df = train_segmentation_model()

    print(segmented_df[
        ["Income",
         "Age",
         "Total_Spending",
         "Cluster"]
    ].head())