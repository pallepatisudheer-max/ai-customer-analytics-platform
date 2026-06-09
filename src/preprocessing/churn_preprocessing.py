import pandas as pd

def preprocess_churn_data():

    df = pd.read_csv(
        "data/telco_churn.csv"
    )

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Fill missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    return df


if __name__ == "__main__":

    churn_df = preprocess_churn_data()

    print(churn_df.head())

    print("\nShape:")
    print(churn_df.shape)