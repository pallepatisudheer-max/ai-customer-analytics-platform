import pandas as pd
import numpy as np

def preprocess_customer_data():

    df = pd.read_csv(
        "data/marketing_campaign.csv",
        sep="\t"
    )

    # Handle missing income values
    df["Income"] = df["Income"].fillna(
        df["Income"].median()
    )

    # Create Age column
    current_year = 2025
    df["Age"] = current_year - df["Year_Birth"]

    # Total Spending
    df["Total_Spending"] = (
        df["MntWines"] +
        df["MntFruits"] +
        df["MntMeatProducts"] +
        df["MntFishProducts"] +
        df["MntSweetProducts"] +
        df["MntGoldProds"]
    )

    return df


if __name__ == "__main__":

    customer_df = preprocess_customer_data()

    print(customer_df.head())

    print("\nShape:")
    print(customer_df.shape)