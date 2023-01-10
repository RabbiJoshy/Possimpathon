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
sid['origin'] = 'sid'
ketel = pd.read_csv('max_clean')
ketel['origin'] = 'ketel'
orig = pd.read_csv('Day1_run_clean')
orig['origin'] = 'orig'
inputs, outputs = IO(orig)

full = pd.concat([ketel, sid, orig.sample(n = 100)], axis = 0)
# full = pd.concat([max, sid], axis = 0)

def Scale(df, scale_columns = 'filllater'):
 scaler = MinMaxScaler()
 data_notscaled = df.drop(inputs, axis=1).reset_index()
 df_scaled = scaler.fit_transform(df[inputs].to_numpy())
 df_scaled = pd.DataFrame(df_scaled, columns= inputs).reset_index()
 df_norm = pd.concat((df_scaled, data_notscaled), axis = 1)

 return df_norm
scaled_df = Scale(full)


# for test_size in np.arange(0.1,0.9,0.1):
for test_size in [0.4]:
 print(test_size)
 train, test = train_test_split(scaled_df, test_size = 0.2)

 # model = LinearRegression()#
 model = xgb.XGBRegressor()
 model.fit(train[inputs], train['out:Effectivedepth'])
 for name in ['orig', 'ketel', 'sid']:
  to_pred_df = test[test['origin'] == name]
  y_pred = model.predict(to_pred_df[inputs])
  y_true = to_pred_df['out:Effectivedepth']
  MAE = mean_absolute_error(y_true, y_pred)
  print('MAE:{}'.format(MAE))
  to_pred_df['error'] = abs(y_pred-y_true)
  to_pred_df['pred'] = y_pred
  to_pred_df['pct_error'] = 100*(to_pred_df['error']/to_pred_df['out:Effectivedepth'])

  # for f, imp in zip(inputs,model.feature_importances_):
  #  print(f, imp)

display_df = to_pred_df.copy()
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

diff = display_df['pct_error']
diff.hist(bins = 40)
plt.title('Histogram of prediction errors')
plt.xlabel('Error (m^2)')
plt.ylabel('Frequency')
plt.show()