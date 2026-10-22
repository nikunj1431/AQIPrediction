import numpy as np
import pandas as pd


City = 'Chandigarh'

df = pd.read_csv("AQIData2023to25/aqi.csv")
print(df.columns.tolist())

for city, data in df.groupby('area'):
    if city == City:
        data.to_csv(f'{city}_23to25.csv', index= False)

