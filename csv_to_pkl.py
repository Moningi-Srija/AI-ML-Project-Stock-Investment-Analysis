import pandas as pd

# Replace 'input.csv' with the path to your CSV file
csv_file_path = 'target_train_csv.csv'

# Read CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Replace 'output.pkl' with the desired path for the Pickle file
pkl_file_path = 'targets_train_df.pkl'

# Save the DataFrame to a Pickle file
df.to_pickle(pkl_file_path)