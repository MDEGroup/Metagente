import pandas as pd
df = pd.read_csv('Dataset_parsed.csv')
checkstyle_rows = df[df['workflows'].str.contains('checkstyle', na=False)]
checkstyle_rows.to_csv('Dataset_checkstyle.csv', index=False)