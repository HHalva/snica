import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pdb


# results for smaller dimensions
df = pd.read_csv("mcc_results.csv")
df['avg_mcc'] = df.iloc[:, 2:].mean(1)
df['std_mcc'] = df.iloc[:, 2:].std(1)

dfn = pd.read_csv("denoise_results.csv")
dfn['avg_mcc'] = dfn.iloc[:, 2:].mean(1)
dfn['std_mcc'] = dfn.iloc[:, 2:].std(1)


# produce plot
n_layers_ = np.unique(df['L'])
algos = ['SNICA (N=3, M=12)', 'SNICA (N=6, M=24)',
         'IIA-HMM (N=3, M=12)', 'IIA-HMM (N=6, M=24)',
         'LGSSM (N=3, M=12)', 'LGSSM (N=6, M=24)', 'iVAE* (N=3, M=12)',
         'iVAE* (N=6, M=24)'
        ]

algos_slides = ['SNICA (N=3, M=12)', 'IIA-HMM (N=3, M=12)',
         'LGSSM (N=3, M=12)', 'iVAE* (N=3, M=12)'
        ]


algos_dn = ['SNICA (N=3, M=12)', 'SNICA (N=6, M=24)', 'LGSSM (N=3, M=12)',
            'LGSSM (N=6, M=24)', 'iVAE* (N=3, M=12)', 'iVAE* (N=6, M=24)'
           ]



marker_dict = {'SNICA (N=3, M=12)':'o', 'SNICA (N=6, M=24)':'o',
               'IIA-HMM (N=3, M=12)':'s', 'IIA-HMM (N=6, M=24)':'s',
               'LGSSM (N=3, M=12)':'x', 'LGSSM (N=6, M=24)':'x',
               'iVAE* (N=3, M=12)': 'D', 'iVAE* (N=6, M=24)': 'D'
              }
line_dict = {'SNICA (N=3, M=12)': 'solid', 'SNICA (N=6, M=24)': 'solid',
             'IIA-HMM (N=3, M=12)': '--',  'IIA-HMM (N=6, M=24)':'--',
             'LGSSM (N=3, M=12)':':', 'LGSSM (N=6, M=24)':':',
             'iVAE* (N=3, M=12)': '-.', 'iVAE* (N=6, M=24)': '-.'
            }
col_dict  = {'SNICA (N=3, M=12)': sns.color_palette("Paired")[5],
             'SNICA (N=6, M=24)': sns.color_palette("Paired")[4],
             'IIA-HMM (N=3, M=12)': sns.color_palette("Paired")[3],
             'IIA-HMM (N=6, M=24)': sns.color_palette("Paired")[2],
             'LGSSM (N=3, M=12)': sns.color_palette("Paired")[7],
             'LGSSM (N=6, M=24)': sns.color_palette("Paired")[6],
             'iVAE* (N=3, M=12)': sns.color_palette("Paired")[9],
             'iVAE* (N=6, M=24)': sns.color_palette("Paired")[8],
            }


MARKER_SIZE = 8
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

f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize = (8, 4))

for a in algos:
    ax1.errorbar(df[df.model == a]['L'], df[df.model == a]['avg_mcc'],
                 yerr=df[df.model == a]['std_mcc'],
                 marker=marker_dict[a], markersize=MARKER_SIZE,
                 label = str(a), linestyle=line_dict[a],
                 linewidth=3, color=col_dict[a])

for a in algos_dn:
    ax2.errorbar(dfn[dfn.model == a]['L'], dfn[dfn.model == a]['avg_mcc'],
                 yerr=dfn[dfn.model == a]['std_mcc'],
                 marker=marker_dict[a], markersize=MARKER_SIZE,
                 label = str(a), linestyle=line_dict[a],
                 linewidth=3, color=col_dict[a])



ax1.set_xlabel('Number of mixing layers')
ax2.set_xlabel('Number of mixing layers')
ax1.set_ylabel('Mean Correlation Coefficient')
#ax3.set_ylabel('Mean Correlation Coefficient')
ax1.set_ylim([0, 1])
ax2.set_ylim([0.8, 1])


ax1.set_title('(a) Identifiability experiment')
ax2.set_title('(b) Denoising experiment')
ax1.legend(loc='lower left', fontsize=7, prop={'size': 8})
f.tight_layout()

plt.savefig('./sim_results.pdf', dpi=1200)
plt.show()
