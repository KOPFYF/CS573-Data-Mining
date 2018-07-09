#    predict whether or not a molecule is poisonous or not   #
#################       Yifan Fei            #################
#################       03/20/2018           #################
# networkx igraph
# data analysis and wrangling
import pandas as pd
import numpy as np
import igraph
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import *
from xgboost import XGBRegressor

import re

from vecstack import stacking

train_df = pd.read_csv('molecule_training.csv')
test_df = pd.read_csv('molecule_TestFeatures.csv')
# https://iwatobipen.wordpress.com/2016/12/30/convert-rdkit-molecule-object-to-igraph-graph-object/
# print('Before dropping dup:',train_df.shape)
# train_df = train_df.drop_duplicates(subset=['Graph', 'smiles','target'], keep='first')
# train_df = train_df.drop_duplicates(subset=['Graph','smiles'], keep='first')
# print('After dropping dup:',train_df.shape)

print(train_df.head(10))

# print('Before dropping dup:',test_df.shape)
# test_df = test_df.drop_duplicates(subset=['Graph', 'smiles'], keep='first')
# print('After dropping dup:',test_df.shape)

# mol2graph function can convert molecule to graph. And add some node and edge attribute.

def mol2graph(mol):
    admatrix = rdmolops.GetAdjacencyMatrix(mol)
    bondidxs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                for b in mol.GetBonds()]
    adlist = np.ndarray.tolist(admatrix)
    graph = igraph.Graph()
    g = graph.Adjacency(adlist).as_undirected()
    for idx in g.vs.indices:
        g.vs[idx]["AtomicNum"] = mol.GetAtomWithIdx(idx).GetAtomicNum()
        g.vs[idx]["AtomicSymbole"] = mol.GetAtomWithIdx(idx).GetSymbol()
    for bd in bondidxs:
        btype = mol.GetBondBetweenAtoms(bd[0], bd[1]).GetBondTypeAsDouble()
        g.es[g.get_eid(bd[0], bd[1])]["BondType"] = btype
        # print( bd, mol.GetBondBetweenAtoms( bd[0], bd[1] ).GetBondTypeAsDouble() )
    return g

def get_Bond_type(graph):
    num_info = re.findall(r"[-+]?[0-9]*\.?[0-9]+",graph)
    Bond_info = []
    # print(type(len(num_info)),len(num_info))
    for i in range(int(len(num_info)/3)):
        Bond_info.append(float(num_info[3*i+2]))
    return Bond_info

def Bond1_percentage(Bond_info):
    Bond1_num = 0
    if len(Bond_info) == 0:
        return 0
    else:
        for i in range(len(Bond_info)):
            if Bond_info[i] == 1:
                Bond1_num += 1
        # return Bond1_num/len(Bond_info)
        return Bond1_num

def Bond2_percentage(Bond_info):
    Bond2_num = 0
    if len(Bond_info) == 0:
        return 0
    else:
        for i in range(len(Bond_info)):
            if Bond_info[i] == 1.5:
                Bond2_num += 1
        # return Bond2_num/len(Bond_info)
        return Bond2_num

def Bond3_percentage(Bond_info):
    Bond3_num = 0
    if len(Bond_info) == 0:
        return 0
    else:
        for i in range(len(Bond_info)):
            if Bond_info[i] == 2:
                Bond3_num += 1
        # return Bond3_num/len(Bond_info)
        return Bond3_num

def fingerprint(fps1,fps2_list):
    similarity_list = []
    for i in fps2_list:
        similarity = DataStructs.FingerprintSimilarity(fps1,i)
        similarity_list.append(similarity)   
        max_sim = np.max(np.asarray(similarity_list))
        # mean_sim = np.mean(np.asarray(similarity_list))
    return max_sim

# train_df['fps'] = train_df['mol'].apply(FingerprintMols.FingerprintMol)
# print('Fingerprints')
# toxic_list = train_df.groupby(['target']).get_group(1)['fps'].tolist()
#     # print('toxic_list:',toxic_list)
# train_df['Max_similarity'] = train_df['fps'].map(lambda x: fingerprint(x,toxic_list))

def stringlist2intarray(A):
    return np.squeeze(np.array([list(s) for s in A], dtype = int))

def get_entropy(lst):
    H = 0
    for i in lst:
        if i == 0:
            pass
        else:
            H = H - i*np.log(i)
    return H


# Feature Engineering


graph_df = train_df[["Graph", "smiles"]]
print('---------------- graph info  ------------------')

graph_ex = graph_df.loc[6, 'Graph']
smile_ex = graph_df.loc[6, 'smiles']
print(graph_ex)
print(type(graph_ex))
print(smile_ex)
print(type(smile_ex))


mol2 = Chem.MolFromSmiles(smile_ex)
g2 = mol2graph(mol2)
print(type(g2), g2)
print('---------------- Igraph info  ------------------')


def Cal_density(x):
    return x[0] / x[1]

def np_max(x):
    if x.size == 0:
        y = 0
    else:
        y = np.max(x)
    return y


def np_min(x):
    if x.size == 0:
        y = 0
    else:
        y = np.min(x)
    return y

def np_mean(x):
    if x.size == 0:
        y = 0
    else:
        y = np.mean(x)
    return y

def features(train_df,test_df):
    train_df['Bond_info'] = train_df['Graph'].apply(get_Bond_type)
    train_df['Bond1_percentage'] = train_df['Bond_info'].apply(Bond1_percentage)
    train_df['Bond2_percentage'] = train_df['Bond_info'].apply(Bond2_percentage)
    train_df['Bond3_percentage'] = train_df['Bond_info'].apply(Bond3_percentage)

    train_df['mol'] = train_df['smiles'].apply(Chem.MolFromSmiles)
    train_df['igraph'] = train_df['mol'].apply(mol2graph)

    # train_df['fps'] = train_df['mol'].apply(FingerprintMols.FingerprintMol)
    # print('Fingerprints')
    # toxic_list = train_df.groupby(['target']).get_group(1)['fps'].tolist()
    #     # print('toxic_list:',toxic_list)
    # train_df['Max_similarity'] = train_df['fps'].map(lambda x: fingerprint(x,toxic_list))

    train_df['average_path_length'] = train_df[
        'igraph'].map(lambda x: x.average_path_length())
    train_df['density'] = train_df[
        'igraph'].map(lambda x: x.density())
    train_df['diameter'] = train_df[
        'igraph'].map(lambda x: x.diameter())

    train_df['number of vertex'] = train_df[
        'igraph'].map(lambda x: x.vcount())
    train_df['number of edges'] = train_df[
        'igraph'].map(lambda x: x.ecount())

    train_df['Average area'] = train_df[[
        'Polar Surface Area','number of vertex']].apply(Cal_density,axis=1)
    train_df['Average weight'] = train_df[[
        'Molecular Weight','number of vertex']].apply(Cal_density,axis=1)

    train_df['Vertex_betweenness'] = train_df[
        'igraph'].map(lambda x: x.betweenness())
    train_df['Edge_betweenness'] = train_df[
        'igraph'].map(lambda x: x.edge_betweenness())

    train_df['Vertex_betweenness'] = train_df[
        'Vertex_betweenness'].apply(np.asarray)
    train_df['Max_Vertex_betweenness'] = train_df[
        'Vertex_betweenness'].apply(np.max)
    train_df['Min_Vertex_betweenness'] = train_df[
        'Vertex_betweenness'].apply(np.min)
    train_df['Mean_Vertex_betweenness'] = train_df['Vertex_betweenness'].apply(np.mean)

    train_df['Edge_betweenness'] = train_df['Edge_betweenness'].apply(np.asarray)
    train_df['Max_Edge_betweenness'] = train_df['Edge_betweenness'].apply(np_max)
    train_df['Min_Edge_betweenness'] = train_df['Edge_betweenness'].apply(np_min)
    train_df['Mean_Edge_betweenness'] = train_df['Edge_betweenness'].apply(np_mean)

    fp_len = 2048   
    fp_feature_list = []
    for i in range(fp_len):
        fp_feature_list.append('m_fp_'+ str(i))
    # print(fp_feature_list)

    print('-' * 25, 'train_data', '-' * 25)
    print(train_df.head(7))

    train_df['m_fps'] = train_df['mol'].map(lambda x: (AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=fp_len)).ToBitString())
    train_df['m_fps_intarray'] = train_df['m_fps'].apply(stringlist2intarray)
    train_df[fp_feature_list] = pd.DataFrame(train_df['m_fps_intarray'].values.tolist(), columns=fp_feature_list)

    # train_df['degree'] = train_df[
    #     'igraph'].map(lambda x: x.degree())
    # train_df['degree'] = train_df['degree'].apply(np.asarray) 
    # train_df['degree_entropy'] = train_df['degree'].apply(get_entropy)



    print('-' * 25, 'train_data', '-' * 25)
    print(train_df.head(7))

    print('-' * 25, 'test_data', '-' * 25)
    test_df['Bond_info'] = test_df['Graph'].apply(get_Bond_type)
    test_df['Bond1_percentage'] = test_df['Bond_info'].apply(Bond1_percentage)
    test_df['Bond2_percentage'] = test_df['Bond_info'].apply(Bond2_percentage)
    test_df['Bond3_percentage'] = test_df['Bond_info'].apply(Bond3_percentage)

    test_df['mol'] = test_df['smiles'].apply(Chem.MolFromSmiles)
    test_df['igraph'] = test_df['mol'].apply(mol2graph)
    # test_df['fps'] = test_df['mol'].apply(FingerprintMols.FingerprintMol)
    # test_df['Max_similarity'] = test_df['fps'].map(lambda x: 
    #         fingerprint(x,toxic_list))

    test_df['average_path_length'] = test_df[
        'igraph'].map(lambda x: x.average_path_length())
    test_df['density'] = test_df[
        'igraph'].map(lambda x: x.density())
    test_df['diameter'] = test_df[
        'igraph'].map(lambda x: x.diameter())

    test_df['number of vertex'] = test_df[
        'igraph'].map(lambda x: x.vcount())
    test_df['number of edges'] = test_df[
        'igraph'].map(lambda x: x.ecount())

    test_df['Average area'] = test_df[[
        'Polar Surface Area','number of vertex']].apply(Cal_density,axis=1)
    test_df['Average weight'] = test_df[[
        'Molecular Weight','number of vertex']].apply(Cal_density,axis=1)

    test_df['Vertex_betweenness'] = test_df[
        'igraph'].map(lambda x: x.betweenness())
    test_df['Edge_betweenness'] = test_df[
        'igraph'].map(lambda x: x.edge_betweenness())


    test_df['Vertex_betweenness'] = test_df['Vertex_betweenness'].apply(np.asarray)
    test_df['Max_Vertex_betweenness'] = test_df['Vertex_betweenness'].apply(np.max)
    test_df['Min_Vertex_betweenness'] = test_df['Vertex_betweenness'].apply(np.min)
    test_df['Mean_Vertex_betweenness'] = test_df['Vertex_betweenness'].apply(np.mean)

    test_df['Edge_betweenness'] = test_df['Edge_betweenness'].apply(np.asarray)
    test_df['Max_Edge_betweenness'] = test_df['Edge_betweenness'].apply(np_max)
    test_df['Min_Edge_betweenness'] = test_df['Edge_betweenness'].apply(np_min)
    test_df['Mean_Edge_betweenness'] = test_df['Edge_betweenness'].apply(np_mean)

    test_df['m_fps'] = test_df['mol'].map(lambda x: (AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=fp_len)).ToBitString())
    test_df['m_fps_intarray'] = test_df['m_fps'].apply(stringlist2intarray)
    test_df[fp_feature_list] = pd.DataFrame(test_df['m_fps_intarray'].values.tolist(), columns=fp_feature_list)

    # test_df['degree'] = test_df[
    #     'igraph'].map(lambda x: x.degree())
    # test_df['degree'] = test_df['degree'].apply(np.asarray) 
    # test_df['degree_entropy'] = test_df['degree'].apply(get_entropy)


    print(test_df.head(7))

    print('-' * 50)

    drop_list = ['inchi_key', 'smiles', 'Graph', 'mol', 'igraph', 'Bond_info', 'Vertex_betweenness',
                    'Edge_betweenness','Min_Vertex_betweenness','m_fps','m_fps_intarray',
                'Min_Edge_betweenness', 
                 # 'Maximum Degree',
                 # 'Minimum Degree', 
                 'number of vertex',
                 'number of edges', 
                 'diameter', #  after drop perofrmance increase
                 'density',
                 # 'Number of Rings',
                 'average_path_length',
                 'Average area', 
                 # 'Polar Surface Area',
                 'Average weight',
                 # 'Molecular Weight',
                 'Mean_Vertex_betweenness',
                 'Mean_Edge_betweenness',
                 'Bond1_percentage',
                 'Bond2_percentage',
                 'Bond3_percentage'
                 ]

    train_df = train_df.drop(drop_list, axis=1)
    test_df = test_df.drop(drop_list, axis=1)
    # train_df = train_df.drop(['index', 'inchi_key'], axis=1)
    # test_df = test_df.drop(['inchi_key'], axis=1)
    # combine = [train_df, test_df]
    print('shape', train_df.shape)
    print('-' * 25, 'train_data', '-' * 25)
    print(train_df.head(20))
    print('-' * 25, 'test_data', '-' * 25)
    print(test_df.head(20))

    return train_df, test_df

train_df, test_df = features(train_df,test_df)


print('----------- after cutting slices ------------')
# print(train_df.head())

print('----------- train info ------------')

# train, valid = train_test_split(train_df, test_size=0.2)

# X_train = train.drop(["target",'index'], axis=1)
# Y_train = train["target"]

# X_valid = valid.drop(["target",'index'], axis=1)
# Y_valid = valid["target"]

# print("shape of X train set : ", X_train.shape)
# print("shape of Y train set : ", Y_train.shape)
# print("shape of X test set : ", X_valid.shape)
# print("shape of Y test set : : ", Y_valid.shape)
# print(X_train.head())
# print(Y_train.head())

# X_train = train_df.drop(["target", 'index'], axis=1)
# Y_train = train_df["target"]

def hyperparameter_tunning(train_df, test_df):
    # tunning

    X_train = train_df.drop(["target", 'index'], axis=1)
    Y_train = train_df["target"]

    # param_test1 = {
    #  'max_depth':[1,2,3],
    #  'min_child_weight':[4,5,6],
    # }
    # gsearch1 = GridSearchCV(estimator = XGBRegressor(learning_rate =0.1, n_estimators=100, max_depth=5,
    #  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
    #  objective= 'binary:logistic', nthread=4, max_delta_step = 1, seed=27),
    #  param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    # print('in test1')
    # gsearch1.fit(X_train,Y_train)
    # print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

    param_test3 = {
        # 'gamma':[i/10.0 for i in range(0,5)]
        # 'subsample':[i/10.0 for i in range(6,10)],
        # 'colsample_bytree':[i/10.0 for i in range(6,10)]
        # 'reg_alpha':[0,0.1,1,10,100]
        # 'reg_lambda':[0.5,1,2,5],
        # 'max_delta_step': [1,3,5,7,9],
        # 'learning_rate':[0.1,0.05,0.02,0.01,0.005],
        # 'n_estimators':[100,200,500,1000,2000]
    }

    gsearch3 = GridSearchCV(estimator = XGBRegressor(learning_rate =0.1, n_estimators=100, max_depth=2,
     min_child_weight=5, gamma=0.4, subsample=0.8, colsample_bytree=0.8, reg_lambda = 1,
     objective= 'binary:logistic', nthread=4, max_delta_step = 1, seed=27),
     param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    print('test3')
    gsearch3.fit(X_train, Y_train)
    print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)

# hyperparameter_tunning(train_df, test_df)


def kfold_CV(train_df):
    # k-fold CV
    AUC_xgb_train_result = []
    AUC_xgb_test_result = []

    kfold = 10
    skf = StratifiedKFold(n_splits=kfold, random_state=42)

    train_x = train_df.drop(["target",'index'], axis=1)
    train_y = train_df["target"]

    print('target: 0 and 1 numbers', train_y.value_counts().tolist())
    weight_balance = train_y.value_counts().tolist(
        )[0] / train_y.value_counts().tolist()[1]
    print('scale_pos_weight:', weight_balance)

    train_x = train_x.values
    train_y = train_y.values
    # X_test = test_df.drop(['index'], axis=1).copy()

    for i, (train_index, test_index) in enumerate(skf.split(train_x,train_y)):

        train_X, test_X = train_x[train_index], train_x[test_index]
        train_Y, test_Y = train_y[train_index], train_y[test_index]

        xgb = XGBRegressor(learning_rate=0.023, n_estimators=5000, max_depth=6,
                        min_child_weight=6, gamma=0, subsample=0.6, colsample_bytree=0.9,
                        objective='binary:logistic', nthread=4, scale_pos_weight=weight_balance, 
                        seed=27, reg_alpha=118)
        xgb.fit(train_X, train_Y)

        Y_pred_onTrain = xgb.predict(train_X)
        AUC_train = roc_auc_score(train_Y, Y_pred_onTrain)
        print("Performance on train: ", AUC_train)
        AUC_xgb_train_result.append(AUC_train)

        Y_pred_onValid = xgb.predict(test_X)
        AUC_test = roc_auc_score(test_Y, Y_pred_onValid)
        print("Performance on test: ", AUC_test)
        AUC_xgb_test_result.append(AUC_test)

        print('[Fold %d/%d Prediciton, AUC score(train, test): %s %s]' % (i + 1, kfold, AUC_train,AUC_test))
    print('Mean and std AUC score for train set:',np.asarray(AUC_xgb_train_result).mean(), np.asarray(AUC_xgb_train_result).std())
    print('Mean and std AUC score for test set:',np.asarray(AUC_xgb_test_result).mean(),np.asarray(AUC_xgb_test_result).std())

# kfold_CV(train_df)

# real predict
X_train = train_df.drop(["target",'index'], axis=1)
Y_train = train_df["target"]

X_test = test_df.drop("index", axis=1).copy()

# print('target: 0 and 1 numbers', Y_train.value_counts().tolist())
# weight_balance = Y_train.value_counts().tolist(
# )[0] / Y_train.value_counts().tolist()[1]
# print('scale_pos_weight:', weight_balance)


xgb = XGBRegressor(learning_rate=0.052338, n_estimators=200, max_depth=4, reg_lambda = 1,
            min_child_weight=8, gamma=9, subsample=0.8523367, colsample_bytree=0.8,
            objective='binary:logistic', nthread=4, max_delta_step = 1, scale_pos_weight=11,
            seed=27)
xgb.fit(X_train, Y_train)
Y_pred = xgb.predict(X_test)

# submission
submission = pd.DataFrame({
        "index": test_df["index"],
        "target": Y_pred
    })
submission.to_csv('Submission_430_n250.csv', index=False) 
print('----------- submitted ------------')
''' 415 with many new features '''
''' 420 with 2048 bit mfps '''
''' 421 with max_delta_step '''
''' 428 with grid search '''

