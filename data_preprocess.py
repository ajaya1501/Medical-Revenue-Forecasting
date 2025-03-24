import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import boxcox
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_log_error



# import pandas as pd
# import numpy as np

# class DataPreprocessor:
#     def __init__(self, file1, file2, file3, output_file="concatenated_output.csv"):
#         self.file1 = file1
#         self.file2 = file2
#         self.file3 = file3
#         self.output_file = output_file
#         self.data = None

#     def read_and_concat_csv(self):
#         try:
#             df1 = pd.read_csv(self.file1)
#             df2 = pd.read_csv(self.file2)
#             df3 = pd.read_csv(self.file3)
            
#             self.data = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

#             self.data.to_csv(self.output_file, index=False)
#             print(f"File saved successfully: {self.output_file}")

#             self.check_nulls()
#             self.convert_datatype()
#             self.sort_and_consider_cols()

#         except FileNotFoundError as e:
#             print(f"Error: {e}")
#             return None

#     def check_nulls(self):
#         print("Checking for missing values...")
#         print(self.data.isnull().sum())

#         print("\nChecking for duplicates...")
#         print(self.data.duplicated().sum())

#         print("\nChecking for infinite values...")
#         print(self.data.replace([np.inf, -np.inf], np.nan).isnull().sum())

#     def convert_datatype(self):
#         self.data['Bill Date'] = pd.to_datetime(self.data['Bill Date'], format="%d-%m-%Y %H:%M")
#         self.data['day'] = self.data['Bill Date'].dt.day
#         self.data['month'] = self.data['Bill Date'].dt.month
#         self.data['year'] = self.data['Bill Date'].dt.year

#     def sort_and_consider_cols(self):
#         self.data = self.data.sort_values(ascending=False, by='Bill Date', inplace=False)
#         self.data = self.data[['Bill Date', 'Net Amount', 'day', 'month', 'year', 'Last Pay Mode']]
#         self.data.to_csv(self.output_file, index=False)
#         print(f"Final data saved after transformation: {self.output_file}")
#         self.read_saved_csv()

#     def read_saved_csv(self):
#         try:
#             df = pd.read_csv(self.output_file)
#             print(f"Successfully read {self.output_file}")
#             print(df.head(), df.shape, df.columns) 
#             return df
#         except FileNotFoundError:
#             print(f"Error: The file '{self.output_file}' was not found.")
#             return None

# if __name__ == "__main__":
#     processor = DataPreprocessor(
#         r"ip bills.xls Dec_aCSV1.csv", 
#         r"ip bills.xls nov_aCSV1.csv", 
#         r"ip bills.xls oct_aCSV1.csv"
#     )
#     processor.read_and_concat_csv()




# import pandas as pd
# import numpy as np

# class DataProcessor:
#     def __init__(self, csv_files, output_file, grouped_file):
#         self.csv_files = csv_files
#         self.output_file = output_file
#         self.grouped_file = grouped_file
#         self.data = None
    
#     def read_and_concat_csv(self):
#         # Read and concatenate the CSV files
#         print("Reading and concatenating CSV files...")
#         dfs = [pd.read_csv(file) for file in self.csv_files]
#         self.data = pd.concat(dfs, axis=0, ignore_index=True)
#         self.data.to_csv(self.output_file, index=False)
#         print(f"Data saved after concatenating: {self.output_file}")
        
#         # Proceed with other transformations
#         self.convert_datatype()
#         self.check_nulls()
#         self.sort_and_select_cols()
#         self.group_data()
#         self.resample_data()

#     def check_nulls(self):
#         # Check for missing values, duplicates, and infinite values
#         print("Checking for missing values...")
#         print(self.data.isnull().sum())

#         print("\nChecking for duplicates...")
#         print(self.data.duplicated().sum())

#         print("\nChecking for infinite values...")
#         print(self.data.replace([np.inf, -np.inf], np.nan).isnull().sum())
    
#     def convert_datatype(self):
#         # Strip time part if present and convert 'Bill Date' to datetime format
#         print("Converting 'Bill Date' to datetime format...")
#         self.data['Bill Date'] = self.data['Bill Date'].str.split(' ').str[0]  # Get only the date part
#         self.data['Bill Date'] = pd.to_datetime(self.data['Bill Date'], format="%d-%m-%Y")
#         self.data['day'] = self.data['Bill Date'].dt.day
#         self.data['month'] = self.data['Bill Date'].dt.month
#         self.data['year'] = self.data['Bill Date'].dt.year

#     def sort_and_select_cols(self):
#         # Sort by 'Bill Date' and select specific columns
#         print("Sorting by 'Bill Date'...")
#         self.data = self.data.sort_values(by='Bill Date', ascending=False, inplace=True)
#         self.data = self.data[['Bill Date', 'Net Amount', 'day', 'month', 'year', 'Last Pay Mode']]
#         self.data.to_csv(self.output_file, index=False)
#         print(f"Data saved after sorting and selecting columns: {self.output_file}")

#     def group_data(self):
#         # Group by 'Bill Date' and sum 'Net Amount' for each date
#         print("Grouping data by 'Bill Date' and summing 'Net Amount'...")
#         grouped_data = self.data.groupby(['Bill Date'], as_index=False)['Net Amount'].sum()

#         # Reset index to make 'Bill Date' a regular column
#         grouped_data = grouped_data.reset_index(drop=True)

#         # Save the grouped data to a new CSV file
#         grouped_data = grouped_data[['Bill Date', 'Net Amount']]
#         grouped_data.to_csv(self.grouped_file, index=False)
#         print(f"Grouped data saved to {self.grouped_file}")

#     def resample_data(self):
#         # Set 'Bill Date' as the index and resample to daily frequency
#         print("Resampling data to daily frequency...")
#         df = pd.read_csv(self.grouped_file)
#         df['Bill Date'] = pd.to_datetime(df['Bill Date'])
#         df.set_index('Bill Date', inplace=True)

#         # Resample by day and forward fill the missing values
#         y = df['Net Amount'].resample('D').mean().ffill()

#         # Save the resampled data to a new CSV file
#         resampled_file = 'resampled_data.csv'
#         y.to_csv(resampled_file, index=True)
#         print(f"Resampled data saved to {resampled_file}")

# # Usage example
# if __name__ == "__main__":
#     # List of CSV files to be read and concatenated
#     csv_files = [
#         'ip bills.xls Dec_aCSV1.csv',
#         'ip bills.xls nov_aCSV1.csv',
#         'ip bills.xls oct_aCSV1.csv'
#     ]
    
#     # Output files
#     output_file = 'concatenated_output.csv'
#     grouped_file = 'grouped_data.csv'

#     # Create DataProcessor instance and process the data
#     processor = DataProcessor(csv_files, output_file, grouped_file)
#     processor.read_and_concat_csv()











import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, csv_files, output_file, grouped_file):
        self.csv_files = csv_files
        self.output_file = output_file
        self.grouped_file = grouped_file
        self.data = None
    
    def read_and_concat_csv(self):
        # Read and concatenate the CSV files
        print("Reading and concatenating CSV files...")
        dfs = [pd.read_csv(file) for file in self.csv_files]
        self.data = pd.concat(dfs, axis=0, ignore_index=True)
        self.data.to_csv(self.output_file, index=False)
        print(f"Data saved after concatenating: {self.output_file}")
        
        # Proceed with other transformations
        self.convert_datatype()
        self.check_nulls()
        self.sort_and_select_cols()
        self.group_data()
        self.resample_data()

    def check_nulls(self):
        # Check for missing values, duplicates, and infinite values
        print("Checking for missing values...")
        print(self.data.isnull().sum())

        print("\nChecking for duplicates...")
        print(self.data.duplicated().sum())

        print("\nChecking for infinite values...")
        print(self.data.replace([np.inf, -np.inf], np.nan).isnull().sum())
    
    def convert_datatype(self):
        # Strip time part if present and convert 'Bill Date' to datetime format
        print("Converting 'Bill Date' to datetime format...")
        self.data['Bill Date'] = self.data['Bill Date'].str.split(' ').str[0]  # Get only the date part
        self.data['Bill Date'] = pd.to_datetime(self.data['Bill Date'], format="%d-%m-%Y")
        self.data['day'] = self.data['Bill Date'].dt.day
        self.data['month'] = self.data['Bill Date'].dt.month
        self.data['year'] = self.data['Bill Date'].dt.year

    def sort_and_select_cols(self):
        # Sort by 'Bill Date' and select specific columns
        print("Sorting by 'Bill Date'...")
        self.data = self.data.sort_values(by='Bill Date', ascending=False)
        self.data = self.data[['Bill Date', 'Net Amount', 'day', 'month', 'year', 'Last Pay Mode']]
        self.data.to_csv(self.output_file, index=False)
        print(f"Data saved after sorting and selecting columns: {self.output_file}")

    def group_data(self):
        # Group by 'Bill Date' and sum 'Net Amount' for each date
        print("Grouping data by 'Bill Date' and summing 'Net Amount'...")
        grouped_data = self.data.groupby(['Bill Date'], as_index=False)['Net Amount'].sum()

        # Reset index to make 'Bill Date' a regular column
        grouped_data = grouped_data.reset_index(drop=True)

        # Save the grouped data to a new CSV file
        grouped_data = grouped_data[['Bill Date', 'Net Amount']]
        grouped_data.to_csv(self.grouped_file, index=False)
        print(f"Grouped data saved to {self.grouped_file}")

    def resample_data(self):
        # Set 'Bill Date' as the index and resample to daily frequency
        print("Resampling data to daily frequency...")
        df = pd.read_csv(self.grouped_file)
        df['Bill Date'] = pd.to_datetime(df['Bill Date'])
        df.set_index('Bill Date', inplace=True)

        # Resample by day and forward fill the missing values
        y = df['Net Amount'].resample('D').mean().ffill()

        # Save the resampled data to a new CSV file
        resampled_file = 'resampled_data.csv'
        y.to_csv(resampled_file, index=True)
        print(f"Resampled data saved to {resampled_file}")

# Usage example
if __name__ == "__main__":
    # List of CSV files to be read and concatenated
    csv_files = [
        'ip bills.xls Dec_aCSV1.csv',
        'ip bills.xls nov_aCSV1.csv',
        'ip bills.xls oct_aCSV1.csv'
    ]
    
    # Output files
    output_file = 'concatenated_output.csv'
    grouped_file = 'grouped_data.csv'

    # Create DataProcessor instance and process the data
    processor = DataProcessor(csv_files, output_file, grouped_file)
    processor.read_and_concat_csv()
