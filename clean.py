import pandas as pd
import numpy as np
import holidays

#add inplace = true to drop na, drop duplicates before final run so that it modifies the original dataset itself

# Policy: Takes in AQI data, standardises the field names in all the datasets. 
# We have around ~40k datapoints for each city of hourly data and around ~3k datapoints of daily data. For this reason, we fit the model to the hourly data and use the daily data strictly for testing
# We do not use the pollutant features in training since a) we can't provide pollutant as a feature during deployment 
# and b) it adds computational and cleaning overhead

df = pd.read_csv('AQIData2015to20/city_hour.csv')
totalrows = len(df)
emptyrows = df['AQI'].isnull().sum()

print(totalrows, ' number of rows with ', emptyrows, ' empty')
print('Removing all empty rows')
df = df.dropna(subset= ['AQI', 'City', 'Datetime'])
df = df.drop_duplicates()
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.drop(columns=['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI_Bucket'])
print('Dataset size after cleaning ', len(df))

#Customise features according to whatever you feel is relevant to the specific model
# By default adding features for festivals, day of the week, 

df['Weekday'] = df['Datetime'].dt.day_of_week
df['is_weekend'] = df['Datetime'].dt.day_of_week.isin([5,6]).astype(int)

years = df['Datetime'].dt.year.unique()
india_holidays = holidays.India(years = years)
df['date'] = df['Datetime'].dt.date

df['is_festival']= df['date'].apply(lambda x: 1 if x in india_holidays else 0)



