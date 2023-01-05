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
        if len(df[column].unique()) > 1:
            if column[:2] == 'in':
                #print(column)
                inputs.append(column)
            if column[:3] == 'out':
                #print(column)
                outputs.append(column)

    return inputs, outputs

def Scale(df, inputs, outputs):
    scaler = MinMaxScaler()
    df_scaled = df[inputs + outputs]
    df_scaled[list(df_scaled.columns)] = scaler.fit_transform(df_scaled[list(df_scaled.columns)])

    return df_scaled

def perform_PCA(df, output_parameters, frac = 0.1, dimensionality = 2, random_state = 25):
    """First Round of Clustering using inputs to predict clusters in 2D"""
    #do PCA
    PCA_df_train = df.sample(frac=frac, random_state=random_state)
    PCA_df_train = PCA_df_train
    pca = PCA(n_components = 2)
    PCA_array = pca.fit_transform(PCA_df_train[output_parameters])

    #First compute
    # global total_computed
    # total_computed += len(PCA_df_train)

    PCA_df_train['PCA1'] = PCA_array[:, 0]
    PCA_df_train['PCA2'] = PCA_array[:, 1]

    return PCA_df_train, pca

"""Use inputs to predict certain criteria"""
def cluster_stage_1(ReducedDF, scaled_df, input_parameters, n_clusters = 3, random_state = 25):

    kmeans_train = ReducedDF
    kmeans = KMeans(n_clusters= n_clusters, random_state= random_state, n_init = 'auto').fit(kmeans_train[['PCA1', 'PCA2']])
    kmeans_train['ClusterS1'] = kmeans.labels_ #kmeans.predict(kmeans_train[outputs])
    print('Stage 1 Clusters:', Counter(kmeans_train['ClusterS1']))

    classifier = xgb.XGBClassifier()
    classifier.fit(kmeans_train[input_parameters], kmeans_train['ClusterS1'])
    scaled_df_clustered = scaled_df.copy()
    scaled_df_clustered['Pred_Cluster_S1'] = classifier.predict(scaled_df[input_parameters])

    kmeans_train['Pred_Cluster_S1'] = classifier.predict(kmeans_train[input_parameters])

    # print('Clustering Accuracy: ',
    #         len(scaled_df_clustered[scaled_df_clustered['Pred_Cluster1'] == scaled_df_clustered['Cluster1']])
    #       / len(scaled_df_clustered)
    #       )

    # ClusterDict = dict()
    # for label in subdf['Cluster1'].unique():
    #     ClusterDict[label] = subdf[subdf['Cluster1'] == label]
    #     print(label, len(ClusterDict[label]))
    # ClusterDict = dict(sorted(ClusterDict.items()))

    return kmeans_train, scaled_df_clustered

def predict_criteria(df, compute_train, criteria, inputs, priority_dictionary):
    crit_df = df.copy()
    crit_train = compute_train.copy()

    output_model_dict = {}
    multiplier = priority_dictionary.copy()

    for criterion in criteria:

        xgc_crit = xgb.XGBRegressor()
        xgc_crit.fit(compute_train[inputs], compute_train[criterion])

        crit_train['pred_' + str(criterion)] = xgc_crit.predict(crit_train[inputs])
        crit_df['pred_' + str(criterion)] = xgc_crit.predict(crit_df[inputs])

        crit_train['weighted_pred_' + str(criterion)] = crit_train['pred_' + str(criterion)] * int(multiplier[criterion])
        crit_df['weighted_pred_' + str(criterion)] = crit_df['pred_' + str(criterion)] * int(multiplier[criterion])

        output_model_dict[criterion] = xgc_crit

    pred_crit_col_list = ['pred_' + str(criterion) for criterion in criteria]
    weighted_pred_crit_col_list = ['weighted_pred_' + str(criterion) for criterion in criteria]

    crit_df['pred_out:Total'] = crit_df[pred_crit_col_list].sum(axis=1)
    crit_df['weighted_pred_out:Total'] = crit_df[weighted_pred_crit_col_list].sum(axis=1)

    return crit_df, output_model_dict

def createdict_stage_1(critdf, cluster_col, criteria, threshold, quartiles = 10):
    subdfdict = {}
    for label in critdf[cluster_col].unique():
        print('Cluster: ', label)
        ### for criterion in criteria:
        ###     subdf = critdf[critdf[criterion] > threshold]  #All that meet a certain criteria

        subdf = critdf[critdf[cluster_col] == label]

        #### subdf['percentile_rank_vals'] = pd.qcut(subdf['pred_out:Total'], 100)
        subdf['S1_percentile_rank'] = pd.qcut(subdf['pred_out:Total'], quartiles, labels = False)
        #Threshold is currently a percentage
        subdf2 = subdf[subdf['S1_percentile_rank'] > (threshold * quartiles)]

        subdfdict[label] = subdf2
        print('Meet Threshold:', len(subdf2))
    subdfdict = dict(sorted(subdfdict.items()))

    return subdfdict
'''Stage 1 over, output: subdfdict'''
def cluster_stage_x(ReducedDF, scaled_df, x, input_parameters, n_clusters = 3, random_state = 25):
    kmeans_train = ReducedDF
    kmeans = KMeans(n_clusters= n_clusters, random_state= random_state, n_init = 'auto').fit(kmeans_train[['PCA1', 'PCA2']])
    kmeans_train['ClusterS' + x] = kmeans.labels_ #kmeans.predict(kmeans_train[outputs])
    print(Counter(kmeans_train['ClusterS' + x]))

    classifier = xgb.XGBClassifier()
    classifier.fit(kmeans_train[input_parameters], kmeans_train['ClusterS' + x])
    scaled_df_clustered = scaled_df.copy()
    scaled_df_clustered['Pred_Cluster_S'+ x] = classifier.predict(scaled_df[input_parameters])

    kmeans_train['Pred_Cluster_S' + x] = classifier.predict(kmeans_train[input_parameters])

    # print('Clustering Accuracy: ',
    #         len(scaled_df_clustered[scaled_df_clustered['Pred_Cluster1'] == scaled_df_clustered['Cluster1']])
    #       / len(scaled_df_clustered)
    #       )

    # ClusterDict = dict()
    # for label in subdf['Cluster1'].unique():
    #     ClusterDict[label] = subdf[subdf['Cluster1'] == label]
    #     print(label, len(ClusterDict[label]))
    # ClusterDict = dict(sorted(ClusterDict.items()))

    return scaled_df_clustered

def cluster_computed_stage_x(subdf, inputs, outputs, x, n_clusters =4, frac = 0.5):

    Reduced_DF_Sx, pca_model = perform_PCA(subdf, outputs, frac)
    clustered_Sx = cluster_stage_x(Reduced_DF_Sx, subdf, x, inputs)
    # kmeans = KMeans(n_clusters=n_clusters, random_state=25)
    # k = kmeans.fit(Reduced_DF_S2[['PCA1', 'PCA2']])
    # subdf['Pred_Cluster2'] = k.labels_

    return clustered_Sx
def create_subdfdict_stage_x(subdfdict, inputs, outputs, x, threshold = 0.5, quartiles = 10):
    #x-1
    subdfdict_Sx = {}
    for key in subdfdict.keys():
        print(f'Stage {x}, Cluster {key}')
        subdf = subdfdict[key]
        if len(subdf) > 0:
            clustered_df_Sx = cluster_computed_stage_x(subdf, inputs, outputs, x)
            for label in clustered_df_Sx['Pred_Cluster_S' + x].unique():

                subsubdf = clustered_df_Sx[clustered_df_Sx["Pred_Cluster_S" + x] == label]

                #TODO make a new set of predictions based on a newly fitted model - [out total] has to change or be new at this point
                if len(subsubdf) > quartiles:
                    subsubdf['S' + str(x)+ '_percentile_rank'] = (pd.qcut(subsubdf['pred_out:Total'], quartiles, labels=False))
                    subsubdf = subsubdf[subsubdf['S' + str(x)+ '_percentile_rank'] > (threshold * quartiles)]
                else:
                    subsubdf['S' + str(x) + '_percentile_rank'] = ['N/A'] * len(subsubdf)

                subdfdict_Sx[str(key) + '.' + str(label)] = subsubdf


    return subdfdict_Sx

def add_crit_col(df, criterion_columns, top_x):
    """ top_x: how many of the best performing variants past to next stage"""
    #TODO add criterion weighting?
    critdf = df.copy()
    critdf['final_criterion'] = critdf[criterion_columns].sum(axis=1)
    critdf = critdf.sort_values(by = 'weighted_pred_out:Total', ascending = False)
    finaldf = critdf[:top_x]

    return finaldf

def add_weighting(df, priority_dictionary):
    for key in priority_dictionary:
        df['weighted_' + key] = int(priority_dictionary[key]) * df[key]
    return df

def create_solutionsdf(subdfdict_x, crit_cols, topx):
    SolutionsDFlist = []
    for key in subdfdict_x.keys():
        subsubdf = subdfdict_x[key]
        critsss = add_crit_col(subsubdf, crit_cols, topx)
        SolutionsDFlist.append(critsss)
    SolutionsDF = pd.concat(SolutionsDFlist, axis = 0)

    return SolutionsDF
