import numpy as np
#import mne
import os
#from elbo import ELBO


#def load_train_data(num_subs=20, pca_dim=20, low_cutoff=4, high_cutoff=30, down_fs=100):
#    """ load resting-state data as training data
#    Args:
#        num_subs: the number of subjects to load
#        pca_dim: the num of dim PCA reduction to
#        low_cutoff: low frequency cutoff
#        high_cutoff: high frequency cutoff
#        down_fs: down sampling frequency
#    Return:
#        x_data: arrays[pca_dim,T]
#        pca_parm: parameter of pca {'W','b'}
#        sub_idx: which subjects load
#    """
#    ### resting data path
#    datapath = '/proj/zhuyong/camcan/cc700/meg/release004/BIDS_20190411/meg_rest_mf'
#    sub_idx = os.listdir(datapath)[:num_subs]
#    sub_names = [idx + '_ses-rest_task-rest_proc-sss.fif' for idx in sub_idx]
#    
#    list_files = [os.path.join(datapath,idx,'ses-rest','meg',name) for idx, name in zip(sub_idx,sub_names)]
#    
#    x_data, _ = gen_segment_data(preproc(list_files, low_cutoff=low_cutoff,
#                                         high_cutoff=high_cutoff, notch=False,
#                                         down_fs=down_fs),len_segment=10, meg='grad')
#    # preprocessing
#    x_data, pca_parm = pca(x_data,num_comp=pca_dim) # PCA
#    return x_data, pca_parm, sub_idx
#
#def load_test_data(sub_idx, task='passive', low_cutoff=4, high_cutoff=30, down_fs=100):
#    """ load task-session data as training data
#    Args:
#        num_subs: the number of subjects to load
#        low_cutoff: low frequency cutoff
#        high_cutoff: high frequency cutoff
#        down_fs: down sampling frequency
#    Return:
#        data_list: list of arrays[n_epochs,n_channels,T]
#        labels_list: parameter of pca {'W','b'}
#    """  
#    datapath = '/proj/zhuyong/camcan/cc700/meg/release004/BIDS_20190411/'
#    datapath = os.path.join(datapath,'meg_' + task + '_mf')
#    sub_names = [idx + '_ses-' + task + '_task-' + task + '_proc-sss.fif' for idx in sub_idx]
#    
#    list_files = [os.path.join(datapath,idx,'ses-' + task,'meg',name) \
#                  for idx, name in zip(sub_idx,sub_names)]
#    
#    print(sub_idx)
#    if task =='smt':
#        data_list, labels_list = load_smt_rawdata(list_files, low_cutoff=low_cutoff, high_cutoff=high_cutoff, notch=False,down_fs=down_fs)
#    else:
#        data_list, labels_list = load_pst_rawdata(list_files, low_cutoff=low_cutoff, high_cutoff=high_cutoff, notch=False,down_fs=down_fs)
#    return data_list, labels_list
#
#def downstearm_svm(data_list, labels_list, results_dic, args):
#    ## for leave one out
#    subid = []
#    for idx in range(len(data_list)):
#        n_epo = data_list[idx].shape[0]
#        subid.append(np.ones([n_epo,],dtype=int)*idx)
#    subid = np.concatenate(subid)
#    ##
#    # catenated across subjects
#    data, labels = np.concatenate(data_list,axis=0), np.concatenate(labels_list,axis=0)
#    
#    # pca
#    n_ep, n_t = data.shape[0], data.shape[2]
#    data = np.transpose(data, [1,2,0])
#    data = np.reshape(data,[data.shape[0],data.shape[2]*data.shape[1]], order='F') # matlab-like
#    
#    data, _ = pca(data, num_comp=args.m)
#    
#    #### features
#    elbo, (qz, qzlag_z, qu, quu) = ELBO(data, results_dic['Rest'], results_dic['lds'], results_dic['hmm'], results_dic['phi'],
#             results_dic['theta'], args.evalkey, args.inference_iters, args.num_samples)
#    # get posterior means of ICs
#    qs = qz[0][:, :, 0]
#    ####
#    n_comp = args.n
#    feats = np.reshape(qs,[n_comp,n_t,n_ep],order='F')
#    feats = np.transpose(feats, [2,0,1])
#    
#    feats = scale4task(feats,pre_sti=int(0.3*args.fs))[:,:,int(0.3*args.fs):]
#    svm_parm = {'w':0.08, 'stride':0.025, 'subid':subid, 'num_sub':len(data_list)}
#    # sliding windows==============================================================
#    # =============================================================================
#    
#    sfreq, n_times = args.fs, feats.shape[2]
#    n_wpts, n_stride = int(sfreq*svm_parm['w']), int(sfreq*svm_parm['stride'])
#    n_winds = int((n_times - n_wpts) / n_stride) + 1
#    
#    Xn = []
#    for i in range(n_winds):
#        Xi = np.mean(feats[:,:,i*n_stride:i*n_stride+n_wpts], axis=2)
#        Xn.append(Xi)
#    
#    X = np.concatenate(Xn, axis=1)
#    y = labels
#    
#    # SVM classification===========================================================
#    # =============================================================================
#    from sklearn.model_selection import LeaveOneOut
#    from sklearn import svm
#    
#    subid = svm_parm['subid']
#    cnt, msc = 0, 0
#    for tr, test in LeaveOneOut().split(range(svm_parm['num_sub'])):
#        
#        X_train, y_train = X[subid!=test[0],:], y[subid!=test[0]]
#        X_test, y_test = X[subid==test[0],:], y[subid==test[0]]
#        
#        clf = svm.LinearSVC(penalty='l2', dual=False, C=2, max_iter=2000)
#        clf.fit(X_train, y_train)
#        sc = clf.score(X_test, y_test)
#        #print('Test acu:%.4f' % sc)
#        msc +=sc
#        cnt +=1
#    print('Test acu:%.4f' % (msc/cnt))
#    return (msc/cnt)
###=============================================================================
#def scale(X):
#    """Standard scaling of data
#    Args:
#        X: array, shape (num_channels,num_times)
#    Returns:
#        Xt: sacaled data, array with shape (num_channels,num_times)
#    """
#    if X.ndim == 2:
#        X -= np.mean(X, axis=1)[:, None]
#        Xt = X / np.std(X, axis=1)[:, None]
#    elif X.ndim == 3:
#        X -= np.mean(X, axis=2)[:,:, None]
#        Xt = X / np.std(X, axis=2)[:,:, None]
#    else:
#        assert False, "ndim erro in scaleing"
#    
#    return Xt
#
###=============================================================================
#def scale4task(X, pre_sti=40):
#    """Standard scaling of data based on pre-stimulus baseline
#    Args:
#        X: array, shape (num_epoch, num_channels,num_times)
#        pre_sti: time points pre onset
#    Returns:
#        Xt: sacaled data, array with shape (num_epoch, num_channels,num_times)
#    """
#    x0 = X[:,:,:pre_sti]
#    X -=np.mean(x0, axis=2)[:,:, None]
#    return X #/ np.std(x0, axis=2)[:,:, None]
#
##==============================================================================
#def load_smt_rawdata(list_files, low_cutoff=0.2, high_cutoff=None, notch=True, down_fs=200):
#    
#    print('Preprocessing smt raw data....')
#    
#    list_data = []
#    list_label = []
#    for fname in list_files:
#        # in case error happen when mne.io.reading
#        try: 
#            raw = mne.io.read_raw_fif(fname, preload=True)
#            
#            if notch: raw.notch_filter(freqs=(50, 100, 150, 200))
#            raw.resample(sfreq=down_fs)
#            raw.filter(l_freq=low_cutoff, h_freq=high_cutoff)
#            
#            events = mne.find_events(raw,stim_channel='STI101', min_duration=0.003,uint_cast=True)
#            events = mne.pick_events(events, include=[1,2,3])
#            picks = mne.pick_types(raw.info,meg='grad')
#            
#            epochs = mne.epochs.Epochs(raw,events,tmin=-.3,tmax=1.6,detrend=1,reject={'grad':3000e-13},picks=picks)
#            #epochs.equalize_event_counts(['9','10'])
#            del raw
#            
#            data = epochs.get_data()
#            
#            labels = epochs.events[:,2]-1
#            del epochs
#            
#            list_data.append(data)
#            list_label.append(labels)
#            
#        except IOError:
#            continue
#        except ValueError:
#            continue
#        except KeyError:
#            continue
#        
#    #return np.concatenate(list_data,axis=0), np.concatenate(list_label,axis=0)
#    return list_data, list_label
#
#def load_pst_rawdata(list_files, low_cutoff=0.2, high_cutoff=None, notch=True, down_fs=200):
#    list_data = []
#    list_label = []
#    for fname in list_files:
#        # in case error happen when mne.io.reading
#        try: 
#            raw = mne.io.read_raw_fif(fname, preload=True)
#            
#            if notch: raw.notch_filter(freqs=(50, 100, 150, 200))
#            raw.resample(sfreq=down_fs)
#            raw.filter(l_freq=low_cutoff, h_freq=high_cutoff)
#            
#            events = mne.find_events(raw,stim_channel='STI101', min_duration=0.003,output='onset')
#            #events = mne.merge_events(events,[6,7,8],10)
#            #events = mne.pick_events(events, include=[6,7,8])
#            picks = mne.pick_types(raw.info,meg='grad')
#            
#            epochs = mne.epochs.Epochs(raw,events,tmin=-.3,tmax=1.2,detrend=1,reject={'grad':4000e-13},picks=picks)
#            #epochs.equalize_event_counts(['9','10'])
#            epochs.equalize_event_counts(['6','7','8','9'])
#            del raw
#            
#            data = epochs.get_data()
#            ####### add
#            labels = epochs.events[:,2]-6
#            del epochs
#            
#            list_data.append(data)
#            list_label.append(labels)
#            
#        except IOError:
#            continue
#        except ValueError:
#            continue
#        except KeyError:
#            continue
#        
#    #return np.concatenate(list_data,axis=0), np.concatenate(list_label,axis=0)
#    return list_data, list_label
###=============================================================================
#def preproc(list_files, low_cutoff=0.2, high_cutoff=None, notch=True, down_fs=200):
#    """Preprocessing: 1, remove low-freq drifts
#                      2, notch filter 50, 100... Hz
#                      3, resample to 200 Hz
#    Args:
#        list_files: list of raw.fif path
#        low_cutoff: low frequency cutoff
#        high_cutoff: high frequency cutoff
#        notvh: do notvh filter
#        down_fs: down sampling frequency
#        
#    Returns:
#        list_proc: list of preprocessed raw MNE data
#    """
#    print('Preprocessing....')
#    list_proc = []
#    for fname in list_files:
#        try:
#            raw = mne.io.read_raw_fif(fname, preload=True)
#            raw.filter(l_freq=low_cutoff, h_freq=high_cutoff)
#            if notch:
#                raw.notch_filter(freqs=(50, 100, 150, 200))
#                
#            raw.resample(sfreq=down_fs)
#            list_proc.append(raw)
#        except IOError:
#            continue
#        except ValueError:
#            continue
#        except KeyError:
#            continue
#    
#    return list_proc
##==============================================================================
#def gen_segment_data(list_proc, len_segment, meg=True):
#    """Generate segment data and the labels
#    Args:
#        list_proc: list of preprocessed raw MNE data
#        len_segment: lengh of segment in seconds
#        meg: mne.pick_types para， If True include MEG channels. 
#            If string it can be ‘mag’, ‘grad’, ‘planar1’ or ‘planar2’ to select only magnetometers, all gradiometers
#    Returns:
#        concatedData: arraty data with (num_channels,num_times) from all subs
#        labels: labels (num_times,)
#    """
#    print('Generating labels...')
#    list_data = []
#    sfreq = list_proc[0].info['sfreq']
#    num_samples_seg = int(sfreq*len_segment) # number of data-points in each segment
#    num_segments = 0 # number of all segments 
#    for raw in list_proc:
#        n_segments = int(raw.n_times/num_samples_seg) # number of segments for each sub
#        meg_channels = mne.pick_types(raw.info, meg=meg, eeg=False, eog=False)
#        data = raw.get_data(picks=meg_channels)
#        #data = scale(data)
#        num_segments += n_segments
#        
#        list_data.append(data[:,:n_segments*num_samples_seg])
#    
#    # labels
#    labels = np.zeros(num_samples_seg*num_segments)
#    for si in range(num_segments):
#        start_idx = num_samples_seg*si
#        end_idx = num_samples_seg*(si+1)
#        labels[start_idx:end_idx] = si
#        
#        
#    return np.concatenate(list_data,axis=1), labels

# ============================================================
# from Dr. Hiroshi Morioka
# ============================================================
def pca(x, num_comp=None, params=None, zerotolerance = 1e-7):
    """Apply PCA whitening to data.
    Args:
        x: data. 2D ndarray [num_comp, num_data]
        num_comp: number of components
        params: (option) dictionary of PCA parameters {'mean':?, 'W':?, 'A':?}. If given, apply this to the data
        zerotolerance: (option)
    Returns:
        x: whitened data
        parms: parameters of PCA
            mean: subtracted mean
            W: whitening matrix
            A: mixing matrix
    """
    print("PCA...")

    # Dimension
    if num_comp is None:
        num_comp = x.shape[0]
    print("    num_comp={0:d}".format(num_comp))

    # From learned parameters --------------------------------
    if params is not None:
        # Use previously-trained model
        print("    use learned value")
        data_pca = x - params['mean']
        x = np.dot(params['W'], data_pca)

    # Learn from data ----------------------------------------
    else:
        # Zero mean
        xmean = np.mean(x, 1).reshape([-1, 1])
        x = x - xmean

        # Eigenvalue decomposition
        xcov = np.cov(x)
        d, V = np.linalg.eigh(xcov)  # Ascending order
        # Convert to descending order
        d = d[::-1]
        V = V[:, ::-1]

        zeroeigval = np.sum((d[:num_comp] / d[0]) < zerotolerance)
        if zeroeigval > 0: # Do not allow zero eigenval
            raise ValueError

        # Calculate contribution ratio
        contratio = np.sum(d[:num_comp]) / np.sum(d)
        print("    contribution ratio={0:f}".format(contratio))

        # Construct whitening and dewhitening matrices
        dsqrt = np.sqrt(d[:num_comp])
        dsqrtinv = 1 / dsqrt
        V = V[:, :num_comp]
        # Whitening
        W = np.dot(np.diag(dsqrtinv), V.transpose())  # whitening matrix
        A = np.dot(V, np.diag(dsqrt))  # de-whitening matrix
        x = np.dot(W, x)

        params = {'mean': xmean, 'W': W, 'A': A}

        # Check
        datacov = np.cov(x)

    return x, params
