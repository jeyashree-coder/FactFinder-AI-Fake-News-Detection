import sqlite3
import pandas as pd
conn = sqlite3.connect('data/factfinder.db')
query = 'SELECT * FROM predictions ORDER BY id DESC LIMIT 5'
df = pd.read_sql(query, conn)
for _, row in df.iterrows():
    print(f"{row['input_type']}: {row['input_value'][:100]} | PREDICTED: {row['prediction']}")
