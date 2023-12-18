import json
import pandas as pd
from sqlalchemy import create_engine
import urllib

class DatabaseETL:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = json.load(file)
        self.server = config['server']
        self.database = config['database']
        self.username = config['username']
        self.password = config['password']
        self.driver = config['driver']
        self.engine = self.create_engine()

    def create_engine(self):
        params = urllib.parse.quote_plus(
            f'DRIVER={{{self.driver}}};'
            f'SERVER={self.server};'
            f'DATABASE={self.database};'
            f'UID={self.username};'
            f'PWD={self.password}'
        )
        return create_engine(f'mssql+pyodbc:///?odbc_connect={params}')

    def read_sql(self, query):
        with self.engine.connect() as connection:
            return pd.read_sql_query(query, con=connection)

    def write_to_sql(self, df, table_name):
        with self.engine.connect() as connection:
            df.to_sql(table_name, con=connection, if_exists='append', index=False)

    def close(self):
        self.engine.dispose()