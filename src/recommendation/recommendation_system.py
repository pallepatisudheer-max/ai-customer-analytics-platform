import pandas as pd

from src.preprocessing.customer_preprocessing import (
    preprocess_customer_data
)

def recommend_products(customer_index=0):

    df = preprocess_customer_data()

    products = {
        "Wines": df.loc[customer_index, "MntWines"],
        "Fruits": df.loc[customer_index, "MntFruits"],
        "Meat": df.loc[customer_index, "MntMeatProducts"],
        "Fish": df.loc[customer_index, "MntFishProducts"],
        "Sweets": df.loc[customer_index, "MntSweetProducts"],
        "Gold": df.loc[customer_index, "MntGoldProds"]
    }

    recommendations = sorted(
        products.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nRecommended Product Categories")

    for category, amount in recommendations[:3]:
        print(f"{category} : {amount}")

if __name__ == "__main__":
    recommend_products()