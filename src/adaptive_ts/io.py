'''
Module to read/write results

Results must be in, or coercible to a pandas Dataframe

'''

import pandas as pd
import sqlite3 as sql

def save_sql(df,name,conn_path,if_exists):
    with sql.connect(conn_path) as conn:
        df.to_sql(name,con=conn,if_exists=if_exists)

def read_sql(name,conn_path):
    with sql.connect(conn_path) as conn:
        df = pd.read_sql(f"SELECT * FROM {name}",con=conn)
    return df