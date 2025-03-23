import pymssql
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np


# # Load environment variables
# load_dotenv()

# # Retrieve credentials from .env
# server = os.getenv('SERVER')
# database = os.getenv('DATABASE')
# username = os.getenv('UID')
# password = os.getenv('PWD')

# # # Check if environment variables are loaded correctly
# # if not all([server, database, username, password]):
# #     print("Error: One or more environment variables are missing.")
# # else:
# #     try:
# #         # Connect to the SQL Server database using pymssql
# #         conn = pymssql.connect(server=server, user=username, password=password, database=database)
# #         print("Connected to the database successfully!")

# #         # Define the CREATE TABLE query
# #         create_table_query = """
# #     CREATE TABLE ipbills_oct (
# #     sino INT PRIMARY KEY,
# #     bill_type VARCHAR(50),
# #     ipop VARCHAR(50),
# #     bill_no VARCHAR(50),
# #     bill_date DATE,
# #     bill_amt DECIMAL(18, 2),
# #     discount DECIMAL(18, 2),
# #     net_amount DECIMAL(18, 2),
# #     paid_amount DECIMAL(18, 2),
# #     balance_amt DECIMAL(18, 2),
# #     last_pay_mode VARCHAR(50),
# #     [user] VARCHAR(100),  -- Use square brackets for reserved keywords
# #     mrn VARCHAR(50),
# #     adm_id VARCHAR(50),
# #     category VARCHAR(50)
# # );
# # """

        
# #         # Try to execute the query to create the table
# #         try:
# #             cursor = conn.cursor()
# #             cursor.execute(create_table_query)
# #             conn.commit()
# #             print("Table 'ipbills_oct' created successfully.")
# #         except Exception as e:
# #             print(f"Error executing query: {e}")
# #         finally:
# #             # Close the connection once done
# #             conn.close()

# #     except Exception as e:
# #         print(f"Connection failed: {e}")


# query = 'SLEECT * from ipbills_oct'

# import pymssql
# import pandas as pd

# class DatabaseConnector:
#     def __init__(self, server, username, password, database):
#         self.server = server
#         self.username = username
#         self.password = password
#         self.database = database
#         self.conn = None
#         self.cursor = None

#     def connect(self):
#         """Establishes the database connection."""
#         try:
#             self.conn = pymssql.connect(server=self.server, user=self.username, password=self.password, database=self.database)
#             self.cursor = self.conn.cursor()
#             print("Connection successful.")
#         except Exception as e:
#             print("Error connecting to the database:", e)

#     def fetch_data(self, query):
#         try:
#             if not self.conn:
#                 raise Exception("Connection not established. Call connect() first.")
#             self.cursor.execute(query)            
#             columns = [col[0] for col in self.cursor.description]            
#             data = self.cursor.fetchall()
            
#             df = pd.DataFrame(data, columns=columns)
#             return df
        
#         except Exception as e:
#             print("Error fetching data:", e)
#             return None

#     def close_connection(self):
#         """Closes the database connection."""
#         if self.conn:
#             self.conn.close()
#             print("Connection closed.")

# if __name__ == "__main__":
#     server = os.getenv('SERVER')
#     username = os.getenv('UID')
#     password = os.getenv('PWD')
#     database = os.getenv('DATABASE')
    
    
#     db_connector = DatabaseConnector(server, username, password, database)
#     db_connector.connect()

#     query = 'SELECT * FROM ipbills_oct'
#     df = db_connector.fetch_data(query)
    
#     if df is not None:
#         print(df) 

#     db_connector.close_connection()






#THIS IS ENTIRELY LOCAL READING OF THE FILE WHICH ISNT SUITABLE FOR STAGING OR PRODUCTION
def read_file(file1, file2, file3):
    import pandas as pd
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    return pd.concat([df1, df2, df3], axis=0, ignore_index=True)


read_file('ip bills.xls Dec_aCSV1.csv', 'ip bills.xls nov_aCSV1.csv', 'ip bills.xls oct_aCSV1.csv')