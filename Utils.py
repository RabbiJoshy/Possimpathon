import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import xgboost as xgb
from collections import Counter
pd.options.mode.chained_assignment = None  # default='warn'

def IO(df):
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


def ScaleOld(df, inputs, outputs):
    scaler = MinMaxScaler()
    df_scaled = df[inputs + outputs]
    df_scaled[list(df_scaled.columns)] = scaler.fit_transform(df_scaled[list(df_scaled.columns)])

    return df_scaled

def displaylight(df, run):
 daylight_matrix = np.array(df['out:DFdata2D'][run])#.reshape(16,6)
 img = plt.imshow(daylight_matrix, cmap='Blues')
 plt.colorbar(img)
 plt.title('Daylight for run:' + str(run))
 plt.show()
 plt.clf()


