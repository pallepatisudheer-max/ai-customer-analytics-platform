from src.preprocessing.churn_preprocessing import (
    preprocess_churn_data
)

df = preprocess_churn_data()

print(df.dtypes)
print("\n")
print(df["Churn"].value_counts())