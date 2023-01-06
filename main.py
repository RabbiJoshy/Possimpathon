import pandas as pd
# import sklearn
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
# import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from Utils import IO
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.linear_model import LinearRegression
import seaborn as sns

#TODO - 2 values have effective depth
rawdf = pd.read_csv('db.csv')
df = rawdf.dropna()
df = df.reset_index().drop(['index'], axis = 1)
df['out:SVdata'] = df['out:SVdata'].apply(lambda x: np.array(x.split(';')[:-1]).astype(float))
df['out:DFdata'] = df['out:DFdata'].apply(lambda x: np.array(x.split(';')[:-1]).astype(float))
df['out:SVdata2D'] = df['out:SVdata'].apply(lambda x: x.reshape(4,4))
df['out:DFdata2D'] = df['out:DFdata'].apply(lambda x: x.reshape(12,8))
df['in:SVdataAVG'] = df['out:SVdata'].apply(lambda x: np.mean(x))
df['out:DFdataAVG'] = df['out:DFdata'].apply(lambda x: np.mean(x))
df = df.drop(['in:Blockage'], axis = 1)
def displaylight(df, run):
 daylight_matrix = np.array(df['out:DFdata2D'][run])#.reshape(16,6)
 img = plt.imshow(daylight_matrix, cmap='Blues')
 plt.colorbar(img)
 plt.title('Daylight for run:' + str(run))
 plt.show()
 plt.clf()
def IO(df):
 inputs = []
 outputs = []
 for column in df.columns:
  # if len(df[column].unique()) > 1:
  if column[:2] == 'in':
    # print(column)
   inputs.append(column)
  if column[:3] == 'out':
    # print(column)
   outputs.append(column)

 return inputs, outputs
inputs, outputs = IO(df)

def Scale(df, scale_columns = 'filllater'):
 scaler = MinMaxScaler()
 data_notscaled = df.drop(inputs, axis=1)
 df_scaled = scaler.fit_transform(df[inputs].to_numpy())
 df_scaled = pd.DataFrame(df_scaled, columns= inputs)
 df_norm = pd.concat((df_scaled, data_notscaled), axis = 1)

 return df_norm
scaled_df = Scale(df)

for test_size in np.arange(0.1,0.9,0.1):
 train, test = train_test_split(scaled_df, test_size= test_size, random_state = 69)

 # model = LinearRegression()#
 model = xgb.XGBRegressor()
 model.fit(train[inputs], train['out:Effectivedepth'])
 y_pred = model.predict(test[inputs])
 y_true = test['out:Effectivedepth']
 MAE = mean_absolute_error(y_true, y_pred)
 print('{} MAE:{}'.format(test_size, MAE))
 test['error'] = abs(y_pred-y_true)
 test['pred'] = y_pred

def view_errors(n = 1, col = 'error', smallest = False):
 biggest_errors = test.sort_values(by = col, ascending = smallest).iloc[:n, :]
 indices = list(biggest_errors.index)

 for idx in indices:
  sns.heatmap(df['out:DFdata2D'][idx], vmax= df['out:DFdataAVG'].max(), vmin=0)
  plt.show()
 return indices

# indices = view_errors(3, 'out:Effectivedepth')
# indices = view_errors(3, 'out:Effectivedepth', True)
#
# indices = view_errors(3)
# cormat = test.corr()
# errorcorr = cormat['error']