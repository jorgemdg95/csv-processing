import pandas as pd

df = pd.read_csv('test.csv')

# Multiply the 'test1' column by 2
df['test1'] = df['test1'] * 2

# Print the DataFrame to verify changes
print(df)

# Export results
df.to_csv('results.csv', index=False)

# Check the process
print("exported file")
