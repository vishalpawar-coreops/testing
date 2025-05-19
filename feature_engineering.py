import pandas as pd
import yaml

# Load the configuration from the YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)
    
# Function for generalized feature engineering
def feature_engineering(df, config):
    # Step 1: Drop specified columns
    columns_to_drop = config['feature_engineering']['input']['feature_engineering']['columns_to_drop']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Step 2: Apply filter based on filter section
    filter_column = config['feature_engineering']['input']['feature_engineering']['filter']['filter_column']
    filter_value = config['feature_engineering']['input']['feature_engineering']['filter']['filter_value']
    df = df[df[filter_column] == filter_value]
    
    # Step 3: Aggregate sales (or other data) by group_by_column
    group_by_column = config['feature_engineering']['input']['feature_engineering']['aggregation']['group_by_column']
    target_column = config['feature_engineering']['input']['feature_engineering']['aggregation']['target_column']
    
    # Group by the specified column and count occurrences of the target column
    aggregated_df = df.groupby(group_by_column).size().reset_index(name=f'{target_column}_count')
    
    return aggregated_df

# Example DataFrame for testing
df = pd.read_csv('raw_data.csv')

# Apply the generalized feature engineering script
aggregated_df = feature_engineering(df, config)

# Save the result
aggregated_df.to_csv("data.csv", index=False)


