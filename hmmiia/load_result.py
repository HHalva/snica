import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import colorsys

import pdb

eval_dir_base = '/home/cbi-data32/morioka/python/projects/hmmiia/storage'


# load_bases = [os.path.join(eval_dir_base, 'model_m%dmax_t16p0.99_e5e2sl100ms64_tcl32i5e4x2_mk%d_ek%d.pkl')]
# load_bases = [os.path.join(eval_dir_base, 'model_m%dmax_t16p0.99_e5e2sl100ms64_tcl32i5e4x2_mk%d_ek%d.pkl')]

# load_bases = [os.path.join(eval_dir_base, 'model_m%dmax_t16p0.99_e5e2sl100ms64_tcl32i5e4x2_sd%d_se%d.pkl')]
# load_bases = [os.path.join(eval_dir_base, 'model_m%dmax_t16p0.99_e3e2sl100ms64_tcl32i5e4x2_sd%d_se%d.pkl')]
# load_bases = [os.path.join(eval_dir_base, 'model_m%drelu_t16p0.99_e2e2sl100ms64_tcl32i5e4x2_sd%d_se%d.pkl')]
# load_bases = [os.path.join(eval_dir_base, 'model_m%drelu_t16p0.99_e5e2sl100ms64_tcl32i5e4x2_sd%d_se%d.pkl')]
# load_bases = [os.path.join(eval_dir_base, 'model_m%drelu_t%dp0.99_e3e2sl100ms64_tcl32i5e4x2_sd%d_se%d.pkl')]
# load_bases = [os.path.join(eval_dir_base, '../../hmmiia_nvar/storage/nvar_m%drelu_t%dp0.99_sd%d_se%d.pkl')]
load_bases = [os.path.join(eval_dir_base, '../storage_nica/nica_m%drelu_t%dp0.99_e3e2sl100ms64_sd%d_se%d.pkl')]
# load_bases = [os.path.join(eval_dir_base, 'model_m%drelu_t%dp0.99_e3e2sl100ms64_tcl32i5e4x2_sd%d_se%d.pkl'),
#               os.path.join(eval_dir_base, '../storage_nvar/nvar_m%drelu_t%dp0.99_sd%d_se%d.pkl')]
load_bases = [os.path.join(eval_dir_base, 'model_m%drelu_t%dp0.99_e3e2sl100ms64_tcl32i5e4x2_sd%d_se%d.pkl'),
              os.path.join(eval_dir_base, '../storage_nica/nica_m%drelu_t%dp0.99_e3e2sl100ms64_sd%d_se%d.pkl'),
              os.path.join(eval_dir_base, '../storage_nvar/nvar_m%drelu_t%dp0.99_sd%d_se%d.pkl')]

# parm_list1 = np.arange(1,6)
parm_list1 = np.array([1,3,5])
# parm_list2 = np.array([0])
parm_list2 = np.array([10,12,14,16])
parm_list3 = np.arange(10)
# parm_list3 = np.array([10])
parm_list4 = np.arange(20)

# parm_list = np.array([9])

# =============================================================
# =============================================================

initialized = False
for fi, load_base in enumerate(load_bases):
    for i1,p1 in enumerate(parm_list1):
        for i2, p2 in enumerate(parm_list2):
            for i3, p3 in enumerate(parm_list3):
                for i4, p4 in enumerate(parm_list4):
                    loadfile = load_base % (p1,p2,p3,p4)
                    if os.path.exists(loadfile):
                        with open(loadfile, 'rb') as f:
                            eval = pickle.load(f)

                        if not initialized:
                            num_comp = eval['data_config']['N']
                            num_file = len(load_bases)
                            num_parm1 = len(parm_list1)
                            num_parm2 = len(parm_list2)
                            num_parm3 = len(parm_list3)
                            num_parm4 = len(parm_list4)
                            #
                            loglhist = np.full([num_file,num_parm1,num_parm2,num_parm3,num_parm4,len(eval['results'])], np.nan)
                            corrhist = np.full([num_file,num_parm1,num_parm2,num_parm3,num_parm4,len(eval['results'])], np.nan)
                            corrdiaghist = np.full([num_file,num_parm1,num_parm2,num_parm3,num_parm4,num_comp,len(eval['results'])], np.nan)
                            accuhist = np.full([num_file,num_parm1,num_parm2,num_parm3,num_parm4,len(eval['results'])], np.nan)
                            #
                            loglbest = np.full([num_file,num_parm1,num_parm2,num_parm3,num_parm4], np.nan)
                            corrbest = np.full([num_file,num_parm1,num_parm2,num_parm3,num_parm4], np.nan)
                            corrdiagbest = np.full([num_file,num_parm1,num_parm2,num_parm3,num_parm4,num_comp], np.nan)
                            accubest = np.full([num_file,num_parm1,num_parm2,num_parm3,num_parm4], np.nan)
                            #
                            loglinit = np.full([num_file,num_parm1,num_parm2,num_parm3,num_parm4], np.nan)
                            corrinit = np.full([num_file,num_parm1,num_parm2,num_parm3,num_parm4], np.nan)
                            corrdiaginit = np.full([num_file,num_parm1,num_parm2,num_parm3,num_parm4,num_comp], np.nan)
                            accuinit = np.full([num_file,num_parm1,num_parm2,num_parm3,num_parm4], np.nan)
                            #
                            tclloss = np.full([num_file,num_parm1,num_parm2,num_parm3,num_parm4], np.nan)
                            tclaccu = np.full([num_file,num_parm1,num_parm2,num_parm3,num_parm4], np.nan)
                            #
                            initialized = True

                        loglhist[fi,i1,i2,i3,i4,:] = [x['logl'] for x in eval['results']]
                        corrhist[fi,i1,i2,i3,i4,:] = [x['corr'] for x in eval['results']]
                        accuhist[fi,i1,i2,i3,i4,:] = [x['acc'] for x in eval['results']]
                        corrdiaghist[fi,i1,i2,i3,i4,:] = np.vstack([x['corrdiag'] for x in eval['results']]).T
                        #
                        loglbest[fi,i1,i2,i3,i4] = eval['best']['logl']
                        corrbest[fi,i1,i2,i3,i4] = eval['best']['corr']
                        accubest[fi,i1,i2,i3,i4] = eval['best']['acc']
                        corrdiagbest[fi,i1,i2,i3,i4] = eval['best']['corrdiag']
                        #
                        if 'init' in eval:
                            loglinit[fi,i1,i2,i3,i4] = eval['init']['logl']
                            corrinit[fi,i1,i2,i3,i4] = eval['init']['corr']
                            accuinit[fi,i1,i2,i3,i4] = eval['init']['acc']
                            corrdiaginit[fi,i1,i2,i3,i4] = eval['init']['corrdiag']
                            #
                            tclloss[fi,i1,i2,i3,i4] = eval['init']['tcl_loss']
                            tclaccu[fi,i1,i2,i3,i4] = eval['init']['tcl_accu']


                    else:
                        print('file %s does not exist' % (loadfile))

logl = loglhist[:,:,:,:,:,-1]
corr = corrhist[:,:,:,:,:,-1]
accu = accuhist[:,:,:,:,:,-1]

# from the last epoch ------------------------------------------
max_logl_idx = np.nanargmax(logl, axis=-1)
max_logl = np.nanmax(logl, axis=-1)

loglmax = logl[np.arange(logl.shape[0])[:, None, None, None],
               np.arange(logl.shape[1])[None, :, None, None],
               np.arange(logl.shape[2])[None, None, :, None],
               np.arange(logl.shape[3])[None, None, None, :],
               max_logl_idx]
corrmax = corr[np.arange(corr.shape[0])[:, None, None, None],
               np.arange(corr.shape[1])[None, :, None, None],
               np.arange(corr.shape[2])[None, None, :, None],
               np.arange(corr.shape[3])[None, None, None, :],
               max_logl_idx]
accumax = accu[np.arange(accu.shape[0])[:, None, None, None],
               np.arange(accu.shape[1])[None, :, None, None],
               np.arange(accu.shape[2])[None, None, :, None],
               np.arange(accu.shape[3])[None, None, None, :],
               max_logl_idx]

loglinitmax = loglinit[np.arange(loglinit.shape[0])[:, None, None, None],
                       np.arange(loglinit.shape[1])[None, :, None, None],
                       np.arange(loglinit.shape[2])[None, None, :, None],
                       np.arange(loglinit.shape[3])[None, None, None, :],
                       max_logl_idx]
corrinitmax = corrinit[np.arange(corrinit.shape[0])[:, None, None, None],
                       np.arange(corrinit.shape[1])[None, :, None, None],
                       np.arange(corrinit.shape[2])[None, None, :, None],
                       np.arange(corrinit.shape[3])[None, None, None, :],
                       max_logl_idx]
accuinitmax = accuinit[np.arange(accuinit.shape[0])[:, None, None, None],
                       np.arange(accuinit.shape[1])[None, :, None, None],
                       np.arange(accuinit.shape[2])[None, None, :, None],
                       np.arange(accuinit.shape[3])[None, None, None, :],
                       max_logl_idx]

# from the best epoch ------------------------------------------
best_max_logl_idx = np.nanargmax(loglbest, axis=-1)
best_max_logl = np.nanmax(loglbest, axis=-1)

loglbestmax = loglbest[np.arange(loglbest.shape[0])[:, None, None, None],
                       np.arange(loglbest.shape[1])[None, :, None, None],
                       np.arange(loglbest.shape[2])[None, None, :, None],
                       np.arange(loglbest.shape[3])[None, None, None, :],
                       best_max_logl_idx]
corrbestmax = corrbest[np.arange(corrbest.shape[0])[:, None, None, None],
                       np.arange(corrbest.shape[1])[None, :, None, None],
                       np.arange(corrbest.shape[2])[None, None, :, None],
                       np.arange(corrbest.shape[3])[None, None, None, :],
                       best_max_logl_idx]
accubestmax = accubest[np.arange(accubest.shape[0])[:, None, None, None],
                       np.arange(accubest.shape[1])[None, :, None, None],
                       np.arange(accubest.shape[2])[None, None, :, None],
                       np.arange(accubest.shape[3])[None, None, None, :],
                       best_max_logl_idx]




corrave = np.nanmean(corrmax, axis=-1)
corrinitave = np.nanmean(corrinitmax, axis=-1)

# # reshape
# corrave = np.reshape(corrave, [-1, corrave.shape[-1]])
# corrinitave = np.reshape(corrinitave, [-1, corrinitave.shape[-1]])

# concatenate
# corrave = np.concatenate([corrave, corrinitave], axis=0)
# corrave = np.concatenate([corrave[:3, :], corrinitave[:3, :], corrave[3:, :]], axis=0)
corrave = np.concatenate([corrave[0,:,:], corrave[1,:,:], corrave[2,:,:], corrinitave[0,:,:]], axis=0)


# base color
# colors = np.array([[0.8353,0.3686,0],
#                    [0.9020,0.6235,0],
#                    [0.9412,0.8941,0.2588],
#                    [0,0.6196,0.4510],
#                    [0,0.4471,0.6980]])
# markers = ['o','s','D','^','v']
# linestyle = ['-']*len(markers)
colors = np.array([[0.8353,0.3686,0],
                   [0.9020,0.6235,0],
                   [0,0.4471,0.6980]])
markers = ['o','^','s']
linestyle = ['-']*len(markers)
num_method = len(load_bases) + 1
# num_method = 2

# augmet colors
def rgb_to_hsv(x):
    y = np.zeros_like(x)
    for i in range(x.shape[0]):
        y[i,:] = colorsys.rgb_to_hsv(x[i,0], x[i,1], x[i,2])
    return y
def hsv_to_rgb(x):
    y = np.zeros_like(x)
    for i in range(x.shape[0]):
        y[i,:] = colorsys.hsv_to_rgb(x[i,0], x[i,1], x[i,2])
    return y
colors_hsv = rgb_to_hsv(colors)

temp = hsv_to_rgb(rgb_to_hsv(colors))
colors_light = hsv_to_rgb(rgb_to_hsv(colors)*[1,0.4,1])
colors_dark = hsv_to_rgb(rgb_to_hsv(colors)*[1,1,0.6])
# colors_black = hsv_to_rgb(rgb_to_hsv(colors)*[1,1,0.2])
colors_black = [[0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,0.5]]

colors = np.concatenate([colors, colors_light, colors_dark, colors_black], axis=0)
markers = markers*num_method
linestyle = ['-']*len(linestyle) + ['--']*len(linestyle) + [':']*len(linestyle) + ['-.']*len(linestyle)
# markers = markers*num_method
# linestyle = ['-']*num_method + ['--']*num_method + [':']*num_method + ['-.']*len(linestyle)


# correlation
fontsize = 16
fig = plt.figure(figsize=[6.4*1.2, 4.8*1.2])
# fig = plt.figure(figsize=[6.4*1.2, 4.8*1.7])
ax = fig.add_subplot(1,1,1)
for i in range(corrave.shape[0]):
# for i in reversed(range(corrave.shape[0])):
    plt.plot(corrave[i,:].T, linewidth=5, color=colors[i,:], marker=markers[i], markersize=15,
             markerfacecolor='white', markeredgewidth=3, linestyle=linestyle[i])
ax.set_xticks(np.arange(len(parm_list2)))
ax.set_xticklabels(parm_list2)
ax.set_xlabel('Number of data (2^x)', fontsize=fontsize)
ax.set_ylabel('Mean correlation', fontsize=fontsize)
plt.xlim([0, len(parm_list2)-1])
plt.ylim([0, 1])
ax.legend(['L' + str(x) for x in parm_list1]*num_method, fontsize=fontsize, bbox_to_anchor=(1.005,1), loc='upper left')
ax.tick_params(labelsize=fontsize)
ax.grid(b=None, which='major', axis='both', linestyle=':')



