import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler

# +
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler

class Cleaner:
    def __init__(self):
        pass
    
    def drop_columns(self, Dataset: pd.DataFrame, columns: list) -> pd.DataFrame:

        return Dataset.drop(columns=columns)
    
    def fill_missing_values(self, Dataset: pd.DataFrame, fill_value: str = 'missing') -> pd.DataFrame:

        return Dataset.fillna(fill_value)
    
    def drop_nan_column(self, Dataset: pd.DataFrame, col: str) -> pd.DataFrame:
        return Dataset.dropna(subset=[col])
    
    def drop_duplicates(self, Dataset: pd.DataFrame) -> pd.DataFrame:
        return Dataset.drop_duplicates()
    
    def convert_to_datetime(self, Dataset: pd.DataFrame, cols: list) -> pd.DataFrame:

        if isinstance(cols, str):

            if cols in Dataset.columns:
                Dataset[cols] = pd.to_datetime(Dataset[cols], errors='coerce')
            else:
                raise ValueError(f"Column '{cols}' does not exist in the DataFrame.")
        elif isinstance(cols, list):
            for col in cols:
                if col in Dataset.columns:
                    Dataset[col] = pd.to_datetime(Dataset[col], errors='coerce')
                else:
                    print(f"Column '{col}' does not exist in the DataFrame.")
        else:
            raise ValueError("Column name should be a string or a list of strings.")
        
        return Dataset
    
    def convert_to_string(self, Dataset: pd.DataFrame, cols: list) -> pd.DataFrame:
        if isinstance(cols, str):
            Dataset[cols] = Dataset[cols].astype(str)
        elif isinstance(cols, list):
            for col in cols:
                Dataset[col] = Dataset[col].astype(str)
        else:
            raise ValueError("Column name should be a string or a list of strings.")
        
        return Dataset
    
    def convert_to_integer(self, Dataset: pd.DataFrame, cols: list) -> pd.DataFrame:
        if isinstance(cols, str):
            Dataset[cols] = Dataset[cols].astype("int64")
        elif isinstance(cols, list):
            for col in cols:
                Dataset[col] = Dataset[col].astype("int64")
        else:
            raise ValueError("Column name should be a string or a list of strings.")
        
        return Dataset
    
    def remove_whitespace_column(self, Dataset: pd.DataFrame) -> pd.DataFrame:
        Dataset.columns = Dataset.columns.str.replace(' ', '_').str.lower()
        return Dataset
    
    def percent_missing(self, Dataset: pd.DataFrame) -> float:
        total_cells = np.product(Dataset.shape)
        missing_count = Dataset.isnull().sum().sum()
        return round(missing_count / total_cells * 100, 2)
    
    def get_numerical_columns(self, Dataset: pd.DataFrame) -> list:
        return Dataset.select_dtypes(include=['float64', 'int64']).columns.to_list()
    
    def get_categorical_columns(self, Dataset: pd.DataFrame) -> list:
        return Dataset.select_dtypes(include=['object', 'datetime64[ns]']).columns.to_list()
    
    def drop_missing_values(self, Dataset: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        missing_percentage = Dataset.isnull().mean()
        columns_to_drop = missing_percentage[missing_percentage >= threshold].index
        return Dataset.drop(columns=columns_to_drop)
    
    def fill_missing_values_categorical(self, Dataset: pd.DataFrame, method: str) -> pd.DataFrame:
        categorical_columns = Dataset.select_dtypes(include=['object', 'datetime64[ns]']).columns
        
        if method == "ffill":
            for col in categorical_columns:
                Dataset[col] = Dataset[col].fillna(method='ffill')
        elif method == "bfill":
            for col in categorical_columns:
                Dataset[col] = Dataset[col].fillna(method='bfill')
        elif method == "mode":
            for col in categorical_columns:
                Dataset[col] = Dataset[col].fillna(Dataset[col].mode()[0])
        else:
            print("Method unknown")
        
        return Dataset
    
    def fill_missing_values_numeric(self, Dataset: pd.DataFrame, method: str, columns: list = None) -> pd.DataFrame:
        if columns is None:
            numeric_columns = Dataset.select_dtypes(include=['float64', 'int64']).columns
        else:
            numeric_columns = columns
        
        if method == "mean":
            for col in numeric_columns:
                Dataset[col].fillna(Dataset[col].mean(), inplace=True)
        elif method == "median":
            for col in numeric_columns:
                Dataset[col].fillna(Dataset[col].median(), inplace=True)
        else:
            print("Method unknown")
        
        return Dataset
    
    def normalizer(self, Dataset: pd.DataFrame) -> pd.DataFrame:
        norm = Normalizer()
        return pd.DataFrame(norm.fit_transform(Dataset[self.get_numerical_columns(Dataset)]), columns=self.get_numerical_columns(Dataset))
    
    def min_max_scaler(self, Dataset: pd.DataFrame) -> pd.DataFrame:
        minmax_scaler = MinMaxScaler()
        return pd.DataFrame(minmax_scaler.fit_transform(Dataset[self.get_numerical_columns(Dataset)]), columns=self.get_numerical_columns(Dataset))
    
    def standard_scaler(self, Dataset: pd.DataFrame) -> pd.DataFrame:
        standard_scaler = StandardScaler()
        return pd.DataFrame(standard_scaler.fit_transform(Dataset[self.get_numerical_columns(Dataset)]), columns=self.get_numerical_columns(Dataset))
    
    def handle_outliers(self, Dataset: pd.DataFrame, col: str, method: str = 'IQR') -> pd.DataFrame:
        Dataset = Dataset.copy()
        q1 = Dataset[col].quantile(0.25)
        q3 = Dataset[col].quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        
        if method == 'mode':
            Dataset[col] = np.where(Dataset[col] < lower_bound, Dataset[col].mode()[0], Dataset[col])
            Dataset[col] = np.where(Dataset[col] > upper_bound, Dataset[col].mode()[0], Dataset[col])
        elif method == 'median':
            Dataset[col] = np.where(Dataset[col] < lower_bound, Dataset[col].median(), Dataset[col])
            Dataset[col] = np.where(Dataset[col] > upper_bound, Dataset[col].median(), Dataset[col])
        else:
            Dataset[col] = np.where(Dataset[col] < lower_bound, lower_bound, Dataset[col])
            Dataset[col] = np.where(Dataset[col] > upper_bound, upper_bound, Dataset[col])
        
        return Dataset
    
    def find_agg(self, Dataset: pd.DataFrame, agg_column: str, agg_metric: str, col_name: str, top: int, order=False) -> pd.DataFrame:
        new_data = Dataset.groupby(agg_column)[agg_column].agg(agg_metric).reset_index(name=col_name).\
                            sort_values(by=col_name, ascending=order)[:top]
        return new_data
    
    def convert_bytes_to_megabytes(self, Dataset: pd.DataFrame, bytes_columns: list) -> pd.DataFrame:
        megabyte = 1e+6  # 1 MB = 1e+6 bytes
        
        # Print existing columns
        print("Existing columns:", Dataset.columns.tolist())
        
        for col in bytes_columns:
            if col in Dataset.columns:
                Dataset[col] = Dataset[col] / megabyte
            else:
                print(f"Column '{col}' does not exist in the DataFrame.")
                
        return Dataset
    
    def missing_values_table(self, Dataset: pd.DataFrame) -> pd.DataFrame:
        mis_val = Dataset.isnull().sum()
        mis_val_percent = 100 * Dataset.isnull().sum() / len(Dataset)
        mis_val_dtype = Dataset.dtypes
        mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(columns={0: 'Missing Values', 1: '% of Total Values', 2: 'Data Type'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 0] != 0].sort_values('% of Total Values', ascending=False).reset_index()
        return mis_val_table_ren_columns
    
    def drop_missing_valuess(self, Dataset: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        missing_percentage = Dataset.isnull().mean()
        columns_to_drop = missing_percentage[missing_percentage >= threshold].index
        return Dataset.drop(columns=columns_to_drop)
    
    def calculate_data_volumes(self, Dataset: pd.DataFrame) -> pd.DataFrame:
        # Print existing columns
        print("Existing columns:", Dataset.columns.tolist())
        
        try:
            Dataset['Social Media Data Volume (Bytes)'] = Dataset['social_media_ul_(bytes)'] + Dataset['social_media_dl_(bytes)']
            Dataset['Google Data Volume (Bytes)'] = Dataset['google_ul_(bytes)'] + Dataset['google_dl_(bytes)']
            Dataset['Email Data Volume (Bytes)'] = Dataset['email_ul_(bytes)'] + Dataset['email_dl_(bytes)']
            Dataset['Youtube Data Volume (Bytes)'] = Dataset['youtube_ul_(bytes)'] + Dataset['youtube_dl_(bytes)']
            Dataset['Netflix Data Volume (Bytes)'] = Dataset['netflix_ul_(bytes)'] + Dataset['netflix_dl_(bytes)']
            Dataset['Gaming Data Volume (Bytes)'] = Dataset['gaming_ul_(bytes)'] + Dataset['gaming_dl_(bytes)']
            Dataset['Other Data Volume (Bytes)'] = Dataset['other_ul_(bytes)'] + Dataset['other_dl_(bytes)']
            Dataset['Total Data Volume (Bytes)'] = Dataset['total_ul_(bytes)'] + Dataset['total_dl_(bytes)']
        except KeyError as e:
            print(f"KeyError: {e}. Please ensure all required columns are present in the DataFrame.")
        
        return Dataset

# -




