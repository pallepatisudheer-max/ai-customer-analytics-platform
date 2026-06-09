import pandas as pd

customer_df = pd.read_csv(
    "data/marketing_campaign.csv",
    sep="\t"
)

churn_df = pd.read_csv(
    "data/telco_churn.csv"
)

print("\nCUSTOMER DATASET")
print(customer_df.shape)
print(customer_df.columns.tolist())

print("\nCHURN DATASET")
print(churn_df.shape)
print(churn_df.columns.tolist())