from flask import Flask, render_template
purchase_r2 = 0.8820
purchase_mae = 111.56
accuracy = 79.56
precision = 65.92
recall = 47.18

app = Flask(__name__)

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/segmentation")
def segmentation():

    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_csv("data/marketing_campaign.csv", sep="\t")
    top_income = df.nlargest(10, "Income")
    plt.figure(figsize=(8,5))

    df["Income"].hist(
    bins=20
)

    plt.title("Income Distribution")
    plt.xlabel("Income")
    plt.ylabel("Customers")

    plt.tight_layout()

    plt.savefig(
    "static/images/income_distribution.png"
)

    plt.close()

    plt.figure(figsize=(10,5))
    plt.bar(
    top_income["ID"].astype(str),
    top_income["Income"]
) 

    plt.title("Top 10 Customers By Income")
    plt.xlabel("Customer ID")
    plt.ylabel("Income")
    plt.tight_layout()

    plt.savefig(
    "static/images/top_income_chart.png"
)
    plt.close()
    customers = df[
        ["ID",
         "Income",
         "Kidhome",
         "Teenhome",
         "Recency"]
    ].head(20)

    top_customers = df[
    [
        "ID",
        "Income",
        "MntWines",
        "MntMeatProducts",
        "MntFishProducts"
    ]
].sort_values(
    by="MntWines",
    ascending=False
).head(10)

    customer_data = top_customers.to_dict(
    orient="records"
)
    total_customers = len(df)
    avg_income = round(
        df["Income"].mean(),
        2
    )
    max_wine = int(
        df["MntWines"].max()
    )

    return render_template(
        "segmentation.html",
        customers=customer_data,
        total_customers=total_customers,
        avg_income=avg_income,
        max_Wine=max_wine,
    )

    



@app.route("/purchase")
def purchase():

    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv("data/marketing_campaign.csv", sep="\t")

    customers = df[
        [
            "ID",
            "Income",
            "MntWines",
            "MntMeatProducts",
            "MntFishProducts"
        ]
    ].head(20)
    
    customer_data = customers.to_dict(
        orient="records"
    )
    

    plt.figure(figsize=(8,5))
    plt.bar(range(20), df["MntWines"].head(20))
    plt.title("Wine Purchases by Customer")
    plt.xlabel("Customer")
    plt.ylabel("Wine Purchases")
    plt.tight_layout()

    plt.savefig("static/images/purchase_chart.png")
    plt.close()
    # Fish Purchases Chart
    plt.figure(figsize=(8,5))
    df["MntFishProducts"].head(20).plot(kind="bar")
    plt.title("Fish Purchases by Customer")
    plt.xlabel("Customer")
    plt.ylabel("Fish Purchases")
    plt.tight_layout()
    plt.savefig("static/images/fish_chart.png")
    plt.close()

    return render_template(
        "purchase.html",
        r2=purchase_r2,
        mae=purchase_mae,
        customers=customer_data
    )


@app.route("/churn")
def churn():
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_csv(
        "data/telco_churn.csv"
    )

    plt.figure(figsize=(8,5))

    df["Churn"].value_counts().plot(
    kind="bar"
)

    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Customers")

    plt.tight_layout()

    plt.savefig(
    "static/images/churn_chart.png"
)
    plt.figure(figsize=(6,6))

    df["Churn"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

    plt.title("Customer Churn Ratio")
    plt.ylabel("")

    plt.savefig("static/images/churn_pie.png")
    plt.close()

    plt.close()

    plt.figure(figsize=(6,6))

    df["Churn"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

    plt.ylabel("")
    plt.title("Customer Churn Percentage")

    plt.savefig(
    "static/images/churn_pie.png"
)

    plt.close()
    churn_data = (
    df[["gender","SeniorCitizen","tenure","MonthlyCharges","Churn"]]
    .head(20)
    .to_dict(orient="records")
)
    
    return render_template(
        "churn.html",
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        customers=churn_data
    )

    
@app.route("/recommendations")
def recommendations():
    return render_template("recommendations.html")

@app.route("/reports")
def reports():
    return render_template("reports.html")


if __name__ == "__main__":
    app.run(debug=True)