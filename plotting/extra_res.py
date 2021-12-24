import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pdb


# results for smaller dimensions
df = pd.read_csv("extra_results.csv")
df['avg_mcc'] = df.iloc[:, 2:].mean(1)
df['std_mcc'] = df.iloc[:, 2:].std(1)
pdb.set_trace()
# produce plot
length_ = np.unique(df['T'])
algos = ['SNICA (N=3, M=12)', 'SNICA (N=6, M=12)', 'SNICA (N=12, M=12)',
         'IIA-HMM (N=3, M=12)', 'IIA-HMM (N=6, M=12)', 'IIA-HMM (N=12, M=12)']
marker_dict = {'SNICA (N=3, M=12)': 'o', 'SNICA (N=6, M=12)': 'o',
             'SNICA (N=12, M=12)': 'o', 'IIA-HMM (N=3, M=12)': 's',
             'IIA-HMM (N=6, M=12)': 's', 'IIA-HMM (N=12, M=12)': 's'}
line_dict = {'SNICA (N=3, M=12)': 'o', 'SNICA (N=6, M=12)': 'o',
             'SNICA (N=12, M=12)': 'o', 'IIA-HMM (N=3, M=12)': 's',
             'IIA-HMM (N=6, M=12)': 's', 'IIA-HMM (N=12, M=12)': 's'}
col_dict = {'SNICA (L=2)': sns.color_palette()[3],
            'SNICA (L=5)': sns.color_palette()[1]}

SMALL_SIZE = 14
MEDIUM_SIZE = 14
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

sns.set_style("whitegrid")
sns.set_palette('deep')

f, ax1 = plt.subplots(1, sharey=True, figsize = (8, 4))

for a in algos:
    ax1.errorbar(df[df.model == a]['T'], df[df.model == a]['avg_mcc'],
                 yerr=df[df.model == a]['std_mcc'],
                 marker=marker_dict[a], label = str(a), linestyle=line_dict[a],
                 linewidth=2, color=col_dict[a])

ax1.set_xlabel('Length of time-series')
ax1.set_ylabel('Mean Correlation Coefficient')
ax1.set_ylim([0, 1])


#ax1.set_title('Identifiability experiment')
ax1.legend( loc='best', fontsize=7 )
f.tight_layout()

plt.savefig('./extra_results.pdf', dpi=1200)
plt.show()