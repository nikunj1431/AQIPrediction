

#Learning
City = 'Chandigarh'

for city, data in df.groupby('City'):
    if city == City:
        df = data

features = [ 'Weekday', 'is_weekend', 'is_festival', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
target = ['AQI']

X = df[features]
y = df[target]

#X = pd.get_dummies(X, columns = ['City'], drop_first= True)

train_size = int(len(df) *0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print( "R square score: ", r2_score(y_test, y_pred))
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
for feature, coef in zip(features, model.coef_):
    print (f"{feature}: {coef}")



