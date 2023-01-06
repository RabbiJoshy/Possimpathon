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

max = pd.read_csv('db MAX.csv')
max['out:SVdata'] = max['out:SVdata'].apply(lambda x: np.array(x[:-1].split(';')).astype(float))
# max['out:DFdata'] = max['out:DFdata'].apply(lambda x: np.array(x.split(';')).astype(float))
#max['out:SVdata2D'] = max['out:SVdata'].apply(lambda x: x.reshape(4,17))
# sid2['out:DFdata2D'] = sid2['out:DFdata'].apply(lambda x: x.reshape(8,17)) #10m long
max['in:SVdataAVG'] = max['out:SVdata'].apply(lambda x: np.mean(x))
max.columns
max = max.drop(['in:WWRWest', 'in:X', 'in:Y'], axis = 1)
max['in:Split'] = 1

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
inputs, outputs = IO(max)
def Scale(df, scale_columns = 'filllater'):
 scaler = MinMaxScaler()
 data_notscaled = df.drop(inputs, axis=1)
 df_scaled = scaler.fit_transform(df[inputs].to_numpy())
 df_scaled = pd.DataFrame(df_scaled, columns= inputs)
 df_norm = pd.concat((df_scaled, data_notscaled), axis = 1)

 return df_norm
scaled_df = Scale(max)

max_clean = scaled_df[scaled_df['out:Effectivedepth'] > 0 ]
max_clean.to_csv('max_clean')