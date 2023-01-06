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
from sklearn.linear_model import LinearRegression
import seaborn as sns
warnings.simplefilter(action='ignore', category=FutureWarning)

sid = pd.read_csv('sid3_clean')
max = pd.read_csv('max_clean')

#TODO - 2 values have effective depth
rawdf = pd.read_csv('Merged Data.csv')
df = rawdf.rename(columns={'out:SVdata;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;': 'out:SVdata'})
# df = df.dropna()
# df = df.reset_index().drop(['index'], axis = 1)
df['out:SVdata'] = df['out:SVdata'].apply(lambda x: np.array(x.split(';')).astype(float))
df['out:DFdata'] = df['out:DFdata'].apply(lambda x: np.array(x[:-1].split(';')).astype(float))
df['out:SVdata2D'] = df['out:SVdata'].apply(lambda x: x.reshape(5,8))
df['out:DFdata2D'] = df['out:DFdata'].apply(lambda x: x.reshape(13,8))
df['in:SVdataAVG'] = df['out:SVdata'].apply(lambda x: np.mean(x))
df['out:DFdataAVG'] = df['out:DFdata'].apply(lambda x: np.mean(x))
df = df.drop(['in:Blockage'], axis = 1)
df['in:Split'] = 1
#
# for i in range(len(df['out:DFdata'])):
#  print(i)
#  np.array(df['out:DFdata'].iloc[i].split(';'))[:-1].astype(float)

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

# for test_size in np.arange(0.1,0.9,0.1):
# for test_size in [0.2]:
#  train, test = train_test_split(scaled_df, test_size= test_size)
full = pd.concat([max, sid, scaled_df], axis = 0)


train, test = train_test_split(full, test_size = 0.2)

 # model = LinearRegression()#
model = xgb.XGBRegressor()
model.fit(train[inputs], train['out:Effectivedepth'])
y_pred = model.predict(test[inputs])
y_true = test['out:Effectivedepth']
MAE = mean_absolute_error(y_true, y_pred)
print('MAE:{}'.format(MAE))
test['error'] = abs(y_pred-y_true)
test['pred'] = y_pred
test['pct_error'] = 100*(test['error']/test['out:Effectivedepth'])

for f, imp in zip(inputs,model.feature_importances_):
 print(f, imp)


display_df = test.copy()
display_df = display_df[['flyID', 'out:Effectivedepth', 'pred', 'error', 'pct_error']]

# def view_column_extremes(n = 1, col = 'error', top = True):
#  if top:
#   ascending = False
#  else:
#   ascending = True
#  col_highest = test.sort_values(by = col, ascending = ascending).iloc[:n, :]
#  indices = list(col_highest.index)
#
#  for idx in indices:
#   sns.heatmap(df['out:DFdata2D'][idx], vmax= df['out:DFdataAVG'].max() *2, vmin=0)
#   plt.title(str(idx))
#   plt.show()
#  return indices
#
# indices = view_column_extremes(3, 'out:Effectivedepth', top = False)
# indices = view_column_extremes(3, 'out:Effectivedepth', True)
# indices = view_column_extremes(20, 'pct_error', True)
#
# cormat = test.corr()
# errorcorr = cormat['pct_error'].sort_values()
#
# sns.heatmap(np.array(errorcorr).reshape(-1,1))
# plt.show()


# Scatterplot and Correlations
# Data
display_df = display_df.sample(n = 150)

x=display_df.index
y1= display_df['out:Effectivedepth']
y2= display_df['pred']

# Plot
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.scatter(x, y1, label='prediction', s = 9)
plt.scatter(x, y2, label='true depth', s = 9)

# Plot
plt.title('Scatterplot and Correlations')
plt.legend()
plt.show()

diff = display_df['error']
diff.hist(bins = 40)
plt.title('Histogram of prediction errors')
plt.xlabel('Error (m^2)')
plt.ylabel('Frequency')
plt.show()