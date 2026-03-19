import pandas as pd


def read_sales_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["customer_name"] = df["customer"].copy()
    df["customer"], _ = pd.factorize(df["customer"])
    return df
