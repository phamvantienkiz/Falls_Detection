import os
import pandas as pd

df = pd.read_csv('data/adl/UMAFall_Subject_02_ADL_Jogging_1_2016-06-13_20-40-29.csv', delimiter=';')

features = [' X-Axis', ' Y-Axis', ' Z-Axis']
X = df[features]

print(X.head())
# print(df.columns)