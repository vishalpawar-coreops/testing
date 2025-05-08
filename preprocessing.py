import pandas as pd
from sqlalchemy import create_engine
import yaml

# Load the configuration file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
print(config['preprocessing']['input'][0]['mysql'])
# Extract MySQL connection details from the config file
db_user = config['preprocessing']['input'][0]['mysql']['user']
db_password = config['preprocessing']['input'][0]['mysql']['password']
db_host = config['preprocessing']['input'][0]['mysql']['host']
db_port = config['preprocessing']['input'][0]['mysql']['port']
db_name = config['preprocessing']['input'][0]['mysql']['database']
table_name = config['preprocessing']['input'][0]['mysql']['table_name']

# Create the MySQL connection string
connection_string = f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'

# Create an SQLAlchemy engine
engine = create_engine(connection_string)

# Load data from MySQL table
print("Loading data from MySQL table...")
df = pd.read_csv('raw_data.csv')

# Check for duplicates and NaN values
num_duplicates = df.duplicated().sum()  # Count duplicate rows
num_null_rows = df.isnull().any(axis=1).sum()  # Count rows with null values

# Remove duplicate rows and rows with NaN values
df_cleaned = df.drop_duplicates().dropna()

# Save the cleaned DataFrame to a CSV file
output_file = 'raw_data.csv'
df_cleaned.to_csv(output_file, index=False)
print(f"Cleaned data saved to {output_file}")

# Print the number of removed rows
print(f"Number of duplicate rows removed: {num_duplicates}")
print(f"Number of rows with null values removed: {num_null_rows}")
