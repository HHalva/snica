import numpy as np
import pdb
import codecs
filename = "k_res_39.txt"

logl_list = []
mcc_list = []
with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        res = {}
        if "Epoch" and "LogL" and "mean corr between" in line:
            lsplit = line.split("\t")
            try:
                for unit in lsplit:
                    if 'LogL' in unit:
                        key, val = unit.split(" ")
                        _logl = float(val)
                    if 'mean corr between' in unit:
                        _mcc = float(unit.split(" ")[-1])
            except:
                continue
            logl_list.append(_logl)
            mcc_list.append(_mcc)

argmax_logl = np.array(logl_list).argmax()
print("Best logl: ", np.array(logl_list).max(),
      "MCC: ", np.array(mcc_list)[argmax_logl])
