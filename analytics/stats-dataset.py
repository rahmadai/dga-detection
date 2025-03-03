import pandas as pd

df = pd.read_csv("dga_data-1.csv")

# df_grouped = df.groupby("subclass")

# df = df[df['subclass'] == 'nivdort']  # Filter for 'nivdort' subclass

print(df.groupby('subclass').size())  # Menjumlahkan nilai dalam tiap kelompok
# print(df.groupby('subclass').agg(list))
