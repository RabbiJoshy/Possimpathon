import pandas as pd
# import sklearn
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from Utils import IO

df = pd.read_csv('db.csv')
run = 10

def dispdaylight(df, run):
 daylightimage = np.array(df['out:DFdata'][run].split(';'))#.reshape(16,6)
 daylightimage = daylightimage[:96]
 daylightimage = daylightimage.astype(float)
 image = daylightimage.reshape(12,8)
 img = plt.imshow(image, cmap='Blues')
 plt.colorbar(img)
 plt.title('Daylight for run:' + str(run))
 plt.show()
 plt.clf()

inputs, outputs = IO(df)
scaler = MinMaxScaler()


data_notscaled = df.drop(inputs, axis=1)
df_scaled = scaler.fit_transform(df[inputs].to_numpy())
df_scaled = pd.DataFrame(df_scaled, columns= inputs)
df_norm = pd.concat((df_scaled, data_notscaled), 1)


train, test = train_test_split(df_norm, test_size= 0.8, random_state = 69)

model = xgb.XGBRegressor()
model.fit(train[inputs], train['out:Effectivedepth'])
y_pred = model.predict(test[inputs])
y_true = test['out:Effectivedepth']
MAE = mean_absolute_error(y_true, y_pred)

test['error'] = abs(y_pred-y_true)
test['pred'] = y_pred
view = pd.DataFrame(dic)
view.set_index()

imagefor = test.sort_values(by = 'error', ascending = False).iloc[:20, :]
imagefor_index = list(imagefor.index)
for run in imagefor_index:
 dispdaylight(df, run)