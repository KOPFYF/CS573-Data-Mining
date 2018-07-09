
# df1 = train_df[["Graph", "target"]].groupby(['Graph'], as_index=False).mean().sort_values(by='target', ascending=False)
# print(df1.head(100))

g1 = sns.FacetGrid(train_df, col='target')
g1.map(plt.hist, 'Molecular Weight', bins=200) # 0-700
g1.savefig('Molecular Weight.png')

g2 = sns.FacetGrid(train_df, col='target')
g2.map(plt.hist, 'Number of Rings',bins=6)
g2.savefig('Number of Rings.png')

g3 = sns.FacetGrid(train_df, col='target')
g3.map(plt.hist, 'Number of H-Bond Donors', bins=3)
g3.savefig('Number of H-Bond Donors.png')

g4 = sns.FacetGrid(train_df, col='target')
g4.map(plt.hist, 'Number of Rotatable Bonds', bins=4)
g4.savefig('Number of Rotatable Bonds.png')

g5 = sns.FacetGrid(train_df, col='target')
g5.map(plt.hist, 'Polar Surface Area', bins=100)
g5.savefig('Polar Surface Area.png')

g6 = sns.FacetGrid(train_df, col='target')
g6.map(plt.hist, 'Maximum Degree', bins=1)
g6.savefig('Maximum Degree.png')

g7 = sns.FacetGrid(train_df, col='target')
g7.map(plt.hist, 'Minimum Degree', bins=1)
g7.savefig('Minimum Degree.png')

plt.show()

# train_df['WeightBand'] = pd.cut(train_df['Molecular Weight'], 6)
# df2 = train_df[['WeightBand', 'target']].groupby(
#     ['WeightBand'], as_index=False).mean().sort_values(by='WeightBand', ascending=True)
# print(df2)

# train_df['HBond_Band'] = pd.cut(train_df['Number of H-Bond Donors'], 6)
# df2 = train_df[['HBond_Band', 'target']].groupby(
#     ['HBond_Band'], as_index=False).mean().sort_values(by='HBond_Band', ascending=True)
# print(df2)

# train_df['Ring_Band'] = pd.cut(train_df['Number of Rings'], 4)
# df2 = train_df[['Ring_Band', 'target']].groupby(
#     ['Ring_Band'], as_index=False).mean().sort_values(by='Ring_Band', ascending=True)
# print(df2)

# train_df['Rotate_Band'] = pd.cut(train_df['Number of Rotatable Bonds'], 4)
# df2 = train_df[['Rotate_Band', 'target']].groupby(
#     ['Rotate_Band'], as_index=False).mean().sort_values(by='Rotate_Band', ascending=True)
# print(df2)

# train_df['Area_Band'] = pd.cut(train_df['Polar Surface Area'], 5)
# df2 = train_df[['Area_Band', 'target']].groupby(
#     ['Area_Band'], as_index=False).mean().sort_values(by='Area_Band', ascending=True)
# print(df2)

# for dataset in combine:
#     dataset.loc[dataset['Molecular Weight'] <= 359.324, 'Molecular Weight'] = 0
#     dataset.loc[(dataset['Molecular Weight'] > 359.324) & (
#         dataset['Molecular Weight'] <= 677.596), 'Molecular Weight'] = 1
#     dataset.loc[(dataset['Molecular Weight'] > 677.596) & (
#         dataset['Molecular Weight'] <= 995.867), 'Molecular Weight'] = 2
#     dataset.loc[(dataset['Molecular Weight'] > 995.867) & (
#         dataset['Molecular Weight'] <= 1314.138), 'Molecular Weight'] = 3
#     dataset.loc[(dataset['Molecular Weight'] > 1314.138) & (
#         dataset['Molecular Weight'] <= 1632.41), 'Molecular Weight'] = 4
#     dataset.loc[dataset['Molecular Weight'] > 1632.41, 'Molecular Weight'] = 5

#     dataset.loc[dataset['Number of H-Bond Donors']
#                 <= 6, 'Number of H-Bond Donors'] = 0
#     dataset.loc[(dataset['Number of H-Bond Donors'] > 6) &
#                 (dataset['Number of H-Bond Donors'] <= 12), 'Number of H-Bond Donors'] = 1
#     dataset.loc[(dataset['Number of H-Bond Donors'] > 12) &
#                 (dataset['Number of H-Bond Donors'] <= 18), 'Number of H-Bond Donors'] = 2
#     dataset.loc[(dataset['Number of H-Bond Donors'] > 18) &
#                 (dataset['Number of H-Bond Donors'] <= 24), 'Number of H-Bond Donors'] = 3
#     dataset.loc[(dataset['Number of H-Bond Donors'] > 24) &
#                 (dataset['Number of H-Bond Donors'] <= 30), 'Number of H-Bond Donors'] = 4
#     dataset.loc[dataset['Number of H-Bond Donors']
#                 > 30, 'Number of H-Bond Donors'] = 5

#     dataset.loc[dataset['Number of Rings'] <= 7.5, 'Number of Rings'] = 0
#     dataset.loc[(dataset['Number of Rings'] > 7.5) &
#                 (dataset['Number of Rings'] <= 15), 'Number of Rings'] = 1
#     dataset.loc[dataset['Number of Rings'] > 15, 'Number of Rings'] = 2

#     dataset.loc[dataset['Number of Rotatable Bonds']
#                 <= 11.75, 'Number of Rotatable Bonds'] = 0
#     dataset.loc[(dataset['Number of Rotatable Bonds'] > 11.75) &
#                 (dataset['Number of Rotatable Bonds'] <= 23.5), 'Number of Rotatable Bonds'] = 1
#     dataset.loc[(dataset['Number of Rotatable Bonds'] > 23.5) &
#                 (dataset['Number of Rotatable Bonds'] <= 35.25), 'Number of Rotatable Bonds'] = 2
#     dataset.loc[dataset['Number of Rotatable Bonds']
#                 > 35.25, 'Number of Rotatable Bonds'] = 3

#     dataset.loc[dataset['Polar Surface Area']
#                 <= 219.17, 'Polar Surface Area'] = 0
#     dataset.loc[(dataset['Polar Surface Area'] > 219.17) &
#                 (dataset['Polar Surface Area'] <= 438.34), 'Polar Surface Area'] = 1
#     dataset.loc[(dataset['Polar Surface Area'] > 438.34) &
#                 (dataset['Polar Surface Area'] <= 657.51), 'Polar Surface Area'] = 2
#     dataset.loc[(dataset['Polar Surface Area'] > 657.51) &
#                 (dataset['Polar Surface Area'] <= 876.68), 'Polar Surface Area'] = 3
#     dataset.loc[dataset['Polar Surface Area']
#                 > 876.68, 'Polar Surface Area'] = 4




def quicksort(nums):
    if len(nums) <= 1:
        return nums
    less = []
    greater = []
    base = nums.pop()

    for x in nums:
        if x < base:
            less.append(x)
        else:
            greater.append(x)
    return quicksort(less) + [base] + quicksort(greater)

def max_list(x):
	y = quicksort(x)
	length = len(y)
	return y[length-1]