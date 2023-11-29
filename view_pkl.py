import pandas as pd

# Load the .pkl file into a DataFrame
# file_path = 'targets_fulldata_df.pkl'
file_path = 'targets_fulldata_df.pkl'
data = pd.read_pickle(file_path)

# Export the DataFrame to a .csv file
# csv_file_path = 'output_full.csv'
csv_file_path = 'output.csv'
data.to_csv(csv_file_path, index=False)  # Set index=False to exclude the DataFrame index in the CSV