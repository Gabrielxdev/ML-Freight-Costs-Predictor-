import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split

def load_vendor_invoice_data(df_path: str):
    conn = sqlite3.connect(df_path)
    query = "select * from vendor_invoice"
    df = pd.read_sql_query(query, conn)
    conn.close() # Close the connection after loading the data
    return df

def prepare_features(df: pd.DataFrame):
    X = df[['Dollars']]
    y = df['Freight']
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)