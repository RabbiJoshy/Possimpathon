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
#Converted to SV data
df = df.drop(['in:Blockage'], axis = 1)
#Include information about split, they were all split
df['in:Split'] = 1
#Exclude Failures
df = df[df['out:Effectivedepth'] > 0 ]
df['in:X']= 1
df['in:Y']= 1
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

# def Scale(df, scale_columns = 'filllater'):
#  scaler = MinMaxScaler()
#  data_notscaled = df.drop(inputs, axis=1)
#  df_scaled = scaler.fit_transform(df[inputs].to_numpy())
#  df_scaled = pd.DataFrame(df_scaled, columns= inputs)
#  df_norm = pd.concat((df_scaled, data_notscaled), axis = 1)
#
#  return df_norm
# scaled_df = Scale(df)

df.to_csv('Day1_run_clean')