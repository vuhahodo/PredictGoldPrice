import pandas as pd

def clean_column_names(df):
    """Remove quotes (single and double) and strip whitespace from column names"""
    df.columns = df.columns.str.replace('"', '').str.replace("'", '').str.strip()
    return df

def prepare_dataframe(df, date_col, price_col):
    df = clean_column_names(df.copy())

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
