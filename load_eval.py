import numpy as np
import pdb
import codecs
filename = "f_res_0.txt"

all_res = []
with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        res = {}
        if "mcc" in line:
            lsplit = line.split("\t")
            for unit in lsplit:
                key = unit.split(" ")[0]
                val = unit.split(" ")[-1]
                if 'ELBO' in key or 'mcc:' in key or 'denoise' in key \
                   or 'eseed:' in key:
                    val = float(val)
                    key = key.replace(":", "")
                    res[key] = float(val)
        if len(res) > 0:
            all_res.append(res)

argmax_elbo = np.argmax([i['ELBO'] for i in all_res])
avg_elbo = np.mean([i['ELBO'] for i in all_res])
elbo_stdev = np.std([i['ELBO'] for i in all_res])
mcc_avg = np.mean([i['mcc'] for i in all_res])
mcc_std = np.std([i['mcc'] for i in all_res])


print("num sims :", len(all_res))
print("best result :", all_res[argmax_elbo])
print("elbo avg: ", avg_elbo)
print("elbo std: ", elbo_stdev)
print("mcc avg: ", mcc_avg)
print("mcc std: ", mcc_std)
