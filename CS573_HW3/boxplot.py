## agg backend is used to create plot as a .png file
# mpl.use('agg')

import matplotlib.pyplot as plt 
import numpy as np

with open('original_accuracy.txt', 'r') as f:
    a = [float(line.rstrip('\n')) for line in f]

with open('new_accuracy.txt', 'r') as f:
    b = [float(line.rstrip('\n')) for line in f]

print('Mean accuracy over test data: {}, std: {}'.format(np.asarray(a).mean(), np.asarray(a).std()))
print('Mean accuracy over test data: {}, std: {}'.format(np.asarray(b).mean(), np.asarray(b).std()))

# print(a)
data = [a,b]


# Create a figure instance
fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot
bp = ax.boxplot(data)

plt.xticks([1, 2], ['Original accuracy', 'New accuracy'])

# Save the figure
fig.savefig('boxplot.png')

from scipy import stats
print(stats.ttest_rel(a,b))