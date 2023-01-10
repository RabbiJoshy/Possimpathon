import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import xgboost as xgb
from collections import Counter
pd.options.mode.chained_assignment = None  # default='warn'

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

# inputs, outputs = IO(df)

def Scale(df, inputs, scale_columns = 'filllater'):
 scaler = MinMaxScaler()
 data_notscaled = df.drop(inputs, axis=1)
 df_scaled = scaler.fit_transform(df[inputs].to_numpy())
 df_scaled = pd.DataFrame(df_scaled, columns= inputs)
 df_norm = pd.concat((df_scaled, data_notscaled), axis = 1)

 return df_norm

def displaylight(df, run):
 daylight_matrix = np.array(df['out:DFdata2D'][run])#.reshape(16,6)
 img = plt.imshow(daylight_matrix, cmap='Blues')
 plt.colorbar(img)
 plt.title('Daylight for run:' + str(run))
 plt.show()
 plt.clf()


