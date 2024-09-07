import pandas as pd
from sqlalchemy import create_engine
import psycopg2

def data_connection():
    engine = create_engine('postgresql+psycopg2://postgres:Belayneh123@uog@localhost:5432/Telecom')
    return engine

def get_dataframe(engine, table_name):
    query = f'SELECT * FROM public.{table_name}'
    data = pd.read_sql_query(query, engine)
    return data

def connection():
    conn = psycopg2.connect(dbname = 'Telecom',
                            user = 'postgres',
                            password = 'Belayneh123@uog',
                            host = 'localhost',
                            port = '5432')
    return conn

def get_dataframe_sql(conn, table_name):
    query = f'SELECT * FROM public.{table_name}'
    data = pd.read_sql_query(query, conn)
    return data

def main():
    engine = data_extraction()
    data = get_dataframe(engine, 'data')
    print(data.head())


