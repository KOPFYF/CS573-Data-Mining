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

from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.ensemble import ExtraTreesRegressor

import re

from vecstack import stacking

train_df = pd.read_csv('molecule_training.csv')
test_df = pd.read_csv('molecule_TestFeatures.csv')
# https://iwatobipen.wordpress.com/2016/12/30/convert-rdkit-molecule-object-to-igraph-graph-object/
# http://igraph.org/python/doc/igraph.Graph-class.html
print('Before dropping dup:',train_df.shape)
# drop_set = ['Graph','smiles','target','Molecular Weight','Polar Surface Area',
#             'Number of Rings','Number of Rotatable Bonds','Number of H-Bond Donors',
#             'Maximum Degree', 'Minimum Degree']
# train_df = train_df.drop_duplicates(subset=drop_set, keep='first')
train_df = train_df.drop_duplicates(subset=['Graph','smiles','target'], keep='first')
# train_df = train_df.drop_duplicates(subset=['Graph','smiles'], keep='first')
print('After dropping dup:',train_df.shape)
print(train_df.head(10))

# mol2graph function can convert molecule to graph. And add some node and
# edge attribute.


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



# Feature Engineering



graph_df = train_df[["Graph", "smiles"]]
print('---------------- graph info  ------------------')

graph_ex = graph_df.loc[0, 'Graph']
smile_ex = graph_df.loc[0, 'smiles']
print(graph_ex)
print(type(graph_ex))
print(smile_ex)
print(type(smile_ex))

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

Bond_info = get_Bond_type(graph_ex)
print(Bond1_percentage(Bond_info))

mol2 = Chem.MolFromSmiles(smile_ex)
print('mol2',mol2)
g2 = mol2graph(mol2)
print(type(g2), g2)
m_fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2,2,nBits=32)
print(type(m_fp2), m_fp2)
print('---------------- Igraph info  ------------------')
# print('component:',g2.components())
# print('degree',g2.degree())

# vb = g2.betweenness()
# eb = g2.edge_betweenness()
# ceb = g2.community_edge_betweenness()
# hist = g2.path_length_hist(directed=False)
# print(hist)
# print('vertex_betweenness',g2.betweenness(),type(vb))
# print('edge_betweenness',g2.edge_betweenness())
# # print('community_edge_betweenness',g2.community_edge_betweenness())

# max_eb = max(g2.edge_betweenness())
# max_b = max(g2.betweenness())
# # max_ceb = max(g2.community_edge_betweenness())
# print('max vertex betweenness',max_b)
# print('max edge betweenness',max_eb,type(max_eb))
# # print('max community_edge_betweenness',max_ceb)


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

def np_median(x):
    if x.size == 0:
        y = 0
    else:
        y = np.median(x)
    return y

def fingerprint(fps1,fps2_list):
    similarity_list = []
    for i in fps2_list:
        similarity = DataStructs.FingerprintSimilarity(fps1,i)
        similarity_list.append(similarity)   
        # max_sim = np.max(np.asarray(similarity_list))
        mean_sim = np.mean(np.asarray(similarity_list))
    return mean_sim

def stringlist2intarray(A):
    return np.squeeze(np.array([list(s) for s in A], dtype = int))



def features(train_df,test_df):
    # train_df['density'] = train_df[['Molecular Weight','Polar Surface Area']].apply(Cal_density,axis=1)
    # train_df['Bond_info'] = train_df['Graph'].apply(get_Bond_type)
    # train_df['Bond1_percentage'] = train_df['Bond_info'].apply(Bond1_percentage)
    # train_df['Bond2_percentage'] = train_df['Bond_info'].apply(Bond2_percentage)
    # train_df['Bond3_percentage'] = train_df['Bond_info'].apply(Bond3_percentage)

    train_df['mol'] = train_df['smiles'].apply(Chem.MolFromSmiles)
    train_df['igraph'] = train_df['mol'].apply(mol2graph)
    # train_df['fps'] = train_df['mol'].apply(FingerprintMols.FingerprintMol)
    # print('Fingerprints')
    # toxic_list = train_df.groupby(['target']).get_group(1)['fps'].tolist()
    # train_df['Max_similarity'] = train_df['fps'].map(lambda x: 
    #     fingerprint(x,toxic_list))
    
    
    # train_df['degree'] = train_df[
    #     'igraph'].map(lambda x: x.degree())
    # train_df['degree'] = train_df['degree'].apply(np.asarray) 

    # train_df['Average degree'] = train_df['degree'].apply(np.mean) 

    # train_df['average_path_length'] = train_df[
    # 'igraph'].map(lambda x: x.average_path_length())
    # train_df['density'] = train_df[
    # 'igraph'].map(lambda x: x.density())
    # train_df['diameter'] = train_df[
    # 'igraph'].map(lambda x: x.diameter())

    # sns.boxplot(x="target", y="diameter", data=train_df)
    # plt.show()

    # train_df['number of vertex'] = train_df[
    # 'igraph'].map(lambda x: x.vcount())
    # train_df['number of edges'] = train_df[
    # 'igraph'].map(lambda x: x.ecount())

    # train_df['Average area'] = train_df[[
    # 'Polar Surface Area','number of vertex']].apply(Cal_density,axis=1)
    # train_df['Average weight'] = train_df[[
    # 'Molecular Weight','number of vertex']].apply(Cal_density,axis=1)

    # train_df['automorphisms'] = train_df[
    #     'igraph'].map(lambda x: x.count_automorphisms_vf2())

    train_df['Vertex_betweenness'] = train_df['igraph'].map(lambda x: x.betweenness())
    train_df['Edge_betweenness'] = train_df['igraph'].map(lambda x: x.edge_betweenness())

    train_df['Vertex_betweenness'] = train_df['Vertex_betweenness'].apply(np.asarray)
    train_df['Max_Vertex_betweenness'] = train_df['Vertex_betweenness'].apply(np.max)
    train_df['Min_Vertex_betweenness'] = train_df['Vertex_betweenness'].apply(np.min)
    train_df['Mean_Vertex_betweenness'] = train_df['Vertex_betweenness'].apply(np.mean)

    train_df['Edge_betweenness'] = train_df['Edge_betweenness'].apply(np.asarray)
    train_df['Max_Edge_betweenness'] = train_df['Edge_betweenness'].apply(np_max)
    train_df['Min_Edge_betweenness'] = train_df['Edge_betweenness'].apply(np_min)
    train_df['Mean_Edge_betweenness'] = train_df['Edge_betweenness'].apply(np_mean)
    # train_df['Median_Edge_betweenness'] = train_df['Edge_betweenness'].apply(np_median)

    fp_len = 2048   
    fp_feature_list = []
    for i in range(fp_len):
        fp_feature_list.append('m_fp_'+ str(i))
    # print(fp_feature_list)

   
    train_df['m_fps'] = train_df['mol'].map(lambda x: (AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=fp_len)).ToBitString())
    train_df['m_fps_intarray'] = train_df['m_fps'].apply(stringlist2intarray)
    train_df[fp_feature_list] = pd.DataFrame(train_df['m_fps_intarray'].values.tolist(), columns=fp_feature_list)
    # print(df0)
    # train_df[fp_feature_list] = df0

    print('-' * 25, 'train_data', '-' * 25)
    print(train_df.head(7))


    print('-' * 25, 'test_data', '-' * 25)


    # test_df['density'] = test_df[['Molecular Weight','Polar Surface Area']].apply(Cal_density,axis=1)
    # test_df['Bond_info'] = test_df['Graph'].apply(get_Bond_type)
    # test_df['Bond1_percentage'] = test_df['Bond_info'].apply(Bond1_percentage)
    # test_df['Bond2_percentage'] = test_df['Bond_info'].apply(Bond2_percentage)
    # test_df['Bond3_percentage'] = test_df['Bond_info'].apply(Bond3_percentage)

    test_df['mol'] = test_df['smiles'].apply(Chem.MolFromSmiles)
    test_df['igraph'] = test_df['mol'].apply(mol2graph)
    # test_df['fps'] = test_df['mol'].apply(FingerprintMols.FingerprintMol)
    # test_df['Max_similarity'] = test_df['fps'].map(lambda x: 
    #     fingerprint(x,toxic_list))

    # test_df['average_path_length'] = test_df[
    # 'igraph'].map(lambda x: x.average_path_length())
    # test_df['density'] = test_df[
    # 'igraph'].map(lambda x: x.density())
    # test_df['diameter'] = test_df[
    # 'igraph'].map(lambda x: x.diameter())
    # test_df['is_simple'] = test_df[
    #     'igraph'].map(lambda x: x.is_simple())
    # test_df['number of vertex'] = test_df[
    # 'igraph'].map(lambda x: x.vcount())
    # test_df['number of edges'] = test_df[
    # 'igraph'].map(lambda x: x.ecount())

    # test_df['Average area'] = test_df[[
    # 'Polar Surface Area','number of vertex']].apply(Cal_density,axis=1)
    # test_df['Average weight'] = test_df[[
    # 'Molecular Weight','number of vertex']].apply(Cal_density,axis=1)


    # test_df['automorphisms'] = test_df[
    #     'igraph'].map(lambda x: x.count_automorphisms_vf2())
    # test_df['degree'] = test_df[
    #     'igraph'].map(lambda x: x.degree())
    # test_df['degree'] = test_df['degree'].apply(np.asarray) 
    # test_df['Average degree'] = test_df['degree'].apply(np.mean)

    test_df['Vertex_betweenness'] = test_df[
    'igraph'].map(lambda x: x.betweenness())
    test_df['Edge_betweenness'] = test_df[
    'igraph'].map(lambda x: x.edge_betweenness())

    test_df['Vertex_betweenness'] = test_df['Vertex_betweenness'].apply(np.asarray)
    test_df['Max_Vertex_betweenness'] = test_df['Vertex_betweenness'].apply(np.max)
    test_df['Min_Vertex_betweenness'] = test_df['Vertex_betweenness'].apply(np.min)
    test_df['Mean_Vertex_betweenness'] = test_df['Vertex_betweenness'].apply(np.mean)
    # test_df['Median_Vertex_betweenness'] = test_df['Vertex_betweenness'].apply(np.std)

    test_df['Edge_betweenness'] = test_df['Edge_betweenness'].apply(np.asarray)
    test_df['Max_Edge_betweenness'] = test_df['Edge_betweenness'].apply(np_max)
    test_df['Min_Edge_betweenness'] = test_df['Edge_betweenness'].apply(np_min)
    test_df['Mean_Edge_betweenness'] = test_df['Edge_betweenness'].apply(np_mean)
    # test_df['Median_Edge_betweenness'] = test_df['Edge_betweenness'].apply(np_median)

    test_df['m_fps'] = test_df['mol'].map(lambda x: (AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=fp_len)).ToBitString())
    test_df['m_fps_intarray'] = test_df['m_fps'].apply(stringlist2intarray)
    test_df[fp_feature_list] = pd.DataFrame(test_df['m_fps_intarray'].values.tolist(), columns=fp_feature_list)

    print(test_df.head(7))

    print('-' * 50)

    drop_list = ['inchi_key', 'smiles', 'Graph', 'mol', 'igraph',  'Vertex_betweenness',
                'Edge_betweenness','Min_Vertex_betweenness','m_fps','m_fps_intarray',
            'Min_Edge_betweenness',
             'Mean_Vertex_betweenness',
             'Mean_Edge_betweenness',
             ]

    # drop_list = ['inchi_key', 'smiles', 'Graph','mol','igraph','Vertex_betweenness',
    # 'Edge_betweenness']

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


    print('----------- after cutting slices ------------')
    # print(train_df.head())

    print('----------- train info ------------')

    return train_df, test_df

train_df, test_df = features(train_df,test_df)


def hyperparameter_tunning(train_df, test_df):
    # tunning

    X_train = train_df.drop(["target", 'index'], axis=1)
    Y_train = train_df["target"]

    param_test1 = {
     'max_depth':[1,2,3],
     'min_child_weight':[4,5,6],
    }
    gsearch1 = GridSearchCV(estimator = XGBRegressor(learning_rate =0.1, n_estimators=100, max_depth=5,
     min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
     objective= 'binary:logistic', nthread=4, max_delta_step = 1, seed=27),
     param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    print('in test1')
    gsearch1.fit(X_train,Y_train)
    print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

    param_test3 = {
        'gamma':[i/10.0 for i in range(0,5)]
        'subsample':[i/10.0 for i in range(6,10)],
        'colsample_bytree':[i/10.0 for i in range(6,10)]
        'reg_alpha':[0,0.1,1,10,100]
        'reg_lambda':[0.5,1,2,5],
        'max_delta_step': [1,3,5,7,9],
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

# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/


def kfold_CV(train_df):
    # k-fold CV

    AUC_xgb_train_result = []
    AUC_xgb_test_result = []

    kfold = 10
    skf = StratifiedKFold(n_splits=kfold, random_state=42)

    train_x = train_df.drop(["target",'index'], axis=1)
    train_y = train_df["target"]

    train_x = train_x.values
    train_y = train_y.values
    # X_test = test_df.drop(['index'], axis=1).copy

    for i, (train_index, test_index) in enumerate(skf.split(train_x,train_y)):

        train_X, test_X = train_x[train_index], train_x[test_index]
        train_Y, test_Y = train_y[train_index], train_y[test_index]

        # xgboost
        # print('target: 0 and 1 numbers', train_Y.value_counts().tolist())
        # weight_balance = train_Y.value_counts().tolist(
        # )[0] / train_Y.value_counts().tolist()[1]
        # print('scale_pos_weight:', weight_balance)

        # xgb = XGBRegressor(learning_rate=0.01, n_estimators=1000, max_depth=6,
        #                 min_child_weight=6, gamma=0, subsample=0.6, colsample_bytree=0.9,
        #                 objective='binary:logistic', nthread=4, scale_pos_weight=weight_balance, 
        #                 seed=27)
        xgb = XGBRegressor(learning_rate=0.1, n_estimators=140, max_depth=5,
                min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                objective='binary:logistic', nthread=4, max_delta_step = 1,
                seed=27)
        # xgb = XGBRegressor(scale_pos_weight=weight_balance)
        # xgb = XGBRegressor()
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

kfold_CV(train_df)

# real predict with stacking

# X_train = train_df.drop(["target",'index'], axis=1)
# Y_train = train_df["target"]
# X_test = test_df.drop(['index'], axis=1).copy()

# # train, valid = train_test_split(train_df, test_size=0.2)

# # X_train = train.drop(["target",'index'], axis=1)
# # Y_train = train["target"]

# # X_test = valid.drop(["target",'index'], axis=1)
# # Y_test = valid["target"]

# # print("shape of X train set : ", X_train.shape)
# # print("shape of Y train set : ", Y_train.shape)
# # print("shape of X test set : ", X_valid.shape)
# # print("shape of Y test set : : ", Y_valid.shape)
# # print(X_train.head())
# # print(Y_train.head())

# xgb2 = XGBRegressor(learning_rate=0.1, n_estimators=140, max_depth=5,
#                     min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
#                     objective='binary:logistic', nthread=4, scale_pos_weight=weight_balance, 
#                     seed=27)
# xgb2.fit(X_train, Y_train)
# print(xgb2.feature_importances_)
# # plt.bar(range(len(xgb2.feature_importances_)), xgb2.feature_importances_)
# plot_importance(xgb2)
# plt.show()
# Y_pred = xgb2.predict(X_test)

############# stacking #############
# https://github.com/KOPFYF/vecstack
# https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
# https://www.dataquest.io/blog/introduction-to-ensembles/
# 将使用不同版本 Feature 的 Model 通过 Ensemble 组合起来

# stacking

decision_tree = DecisionTreeRegressor()
random_forest = RandomForestRegressor(n_estimators=100) 
logreg = LogisticRegression()

# Initialize 1-st level models.
models = [xgb, decision_tree, random_forest,logreg]

# Compute stacking features
def RMSLE(y, pred):
    return mean_squared_error(y, pred) ** 0.5

S_train, S_test = stacking(models, X_train, Y_train, X_test,regression = True, 
                 n_folds = 5, shuffle = True, metric = RMSLE, random_state = 2018, verbose = 2)

# Initialize 2-nd level model
model = logreg

# Fit 2-nd level model
model = model.fit(S_train, Y_train)

# Predict
y_final_pred = model.predict(S_test)
AUC = roc_auc_score(Y_test,y_final_pred)
print('AUC after stacking',AUC)


