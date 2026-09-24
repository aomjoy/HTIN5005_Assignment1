import pandas as pd

df = pd.read_csv("training_setA/p000001.psv", sep="|")
print(df.head())
print(df.columns.tolist())
print(df.shape)