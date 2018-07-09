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

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

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

train_df = pd.read_csv('molecule_training.csv')
test_df = pd.read_csv('molecule_TestFeatures.csv')
# https://iwatobipen.wordpress.com/2016/12/30/convert-rdkit-molecule-object-to-igraph-graph-object/
# print(train_df.head(10))
print('Before dropping dup:',train_df.shape)
drop_set = ['Graph','smiles','target','Molecular Weight','Polar Surface Area',
            'Number of Rings','Number of Rotatable Bonds','Number of H-Bond Donors',
            'Maximum Degree', 'Minimum Degree']
train_df = train_df.drop_duplicates(subset=drop_set, keep='first')
# train_df = train_df.drop_duplicates(subset=['Graph','smiles','target'], keep='first')
# train_df = train_df.drop_duplicates(subset=['Graph','smiles'], keep='first')

# print(train_df[train_df.duplicated(subset=['Graph','smiles','target'], keep='first')])


# print(train_df.duplicated(subset=['Graph','smiles'], keep='first'))


print('After dropping dup:',train_df.shape)

print(train_df.groupby(['target']).get_group(0))
print(train_df.groupby(['target']).get_group(1))

# df = train_df[["Graph","smiles","target"]]
# ids = df["Graph"]

# print(df[ids.isin(ids[ids.duplicated()])])

# print(train_df['target'].groupby([train_df['Graph'], train_df['smiles']]).mean())


# train_df.info()
# print('_' * 40)
# test_df.info()
# print('_' * 40)
# print(train_df.describe())
# print(train_df.describe(include=['O']))

# Feature Engineering
# train_df = train_df.drop(['index', 'inchi_key', 'smiles', 'Graph'], axis=1)
# test_df = test_df.drop(['inchi_key', 'smiles', 'Graph'], axis=1)
# train_df = train_df.drop(['index', 'inchi_key'], axis=1)
# test_df = test_df.drop(['inchi_key'], axis=1)
# combine = [train_df, test_df]

print(train_df["target"].value_counts())

# sns.jointplot(x="Maximum Degree", y="Minimum Degree", data=train_df)

sns.FacetGrid(train_df, hue="target", size=5) \
   .map(plt.scatter, "Molecular Weight", "Polar Surface Area") \
   .add_legend()

sns.boxplot(x="target", y="Molecular Weight", data=train_df)

sns.pairplot(train_df.drop(['index', 'inchi_key', 'smiles', 'Graph'], axis=1), hue="target")

# plt.show()
# https://www.kaggle.com/benhamner/python-data-visualizations

# new feature:

# Molecular Weight/ Polar Surface Area
# Sum of bonds
# Average degree

# print(train_df.head(10))
# print('_' * 40)
# print(test_df.head(10))



# Graph transformation
# graph_df = train_df[["Graph", "smiles"]]
# print('---------------- graph info  ------------------')

# graph_ex = graph_df.loc[4,'Graph']
# smile_ex = graph_df.loc[4,'smiles']
# print(graph_ex)
# print(type(graph_ex))
# print(smile_ex)
# print(type(smile_ex))

# mol2 = Chem.MolFromSmiles(smile_ex)
# g2=mol2graph(mol2)
# print(type(g2),g2)
# print('---------------- Igraph info  ------------------')
# # print('component:',g2.components())
# # print('degree',g2.degree())

# vb = g2.betweenness()
# eb = g2.edge_betweenness()
# ceb = g2.community_edge_betweenness()

# print('vertex_betweenness',g2.betweenness(),type(vb))
# print('edge_betweenness',g2.edge_betweenness())
# # print('community_edge_betweenness',g2.community_edge_betweenness())

# max_eb = max(g2.edge_betweenness())
# max_b = max(g2.betweenness())
# # max_ceb = max(g2.community_edge_betweenness())
# print('max vertex betweenness',max_b)
# print('max edge betweenness',max_eb,type(max_eb))
# # print('max community_edge_betweenness',max_ceb)

# train_df['mol'] = train_df['smiles'].apply(Chem.MolFromSmiles)
# train_df['igraph'] = train_df['mol'].apply(mol2graph)

# train_df['Vertex_betweenness'] = train_df['igraph'].map(lambda x: x.betweenness())
# train_df['Edge_betweenness'] = train_df['igraph'].map(lambda x: x.edge_betweenness())


#     xgb = XGBRegressor(learning_rate=0.1, n_estimators=140, max_depth=5,
#                     min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
#                     objective='binary:logistic', nthread=4, scale_pos_weight=weight_balance, 
#                     seed=27, No-reg_alpha=118)








