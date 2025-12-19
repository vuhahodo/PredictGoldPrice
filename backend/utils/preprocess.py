import pandas as pd

def prepare_dataframe(df, date_col, price_col):
    df = df.copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    df[price_col] = (
        df[price_col].astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    df = df.dropna(subset=[date_col, price_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    return df
