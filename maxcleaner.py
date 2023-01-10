import pandas as pd
import numpy as np
from Utils import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


max = pd.read_csv('db MAX.csv')
max = max.dropna()
max['out:SVdata'] = max['out:SVdata'].apply(lambda x: np.array(x[:-1].split(';')).astype(float))
max['out:DFdata'] = max['out:DFdata'].apply(lambda x: np.array(x[:-1].split(';')).astype(float))
#max['out:SVdata2D'] = max['out:SVdata'].apply(lambda x: x.reshape(4,17))
# sid2['out:DFdata2D'] = sid2['out:DFdata'].apply(lambda x: x.reshape(8,17)) #10m long
max['in:SVdataAVG'] = max['out:SVdata'].apply(lambda x: np.mean(x))
# max.columns
max = max.drop(['in:WWRWest'], axis = 1)
# max = max.drop(['in:WWRWest', 'in:X', 'in:Y'], axis = 1)
max['in:Split'] = 1
##weird 10x column
max = max.drop([100])

inputs, outputs = IO(max)
# scaled_df = Scale(max, inputs)

max_clean = max[max['out:Effectivedepth'] > 0 ]
max_clean.to_csv('max_clean')
