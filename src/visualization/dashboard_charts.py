import matplotlib.pyplot as plt
import os

from src.segmentation.kmeans_segmentation import (
    train_segmentation_model
)


def generate_customer_segments_chart():

    df = train_segmentation_model()

    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(
        df["Income"],
        df["Total_Spending"],
        c=df["Cluster"]
    )

    plt.xlabel("Income")
    plt.ylabel("Total Spending")
    plt.title("Customer Segmentation")

    plt.colorbar(scatter)
    os.makedirs("outputs/charts", exist_ok=True)

    plt.savefig(
        "outputs/charts/customer_segments.png"
    )

    plt.close()

    print(
        "Chart saved to outputs/charts/customer_segments.png"
    )


if __name__ == "__main__":
    generate_customer_segments_chart()