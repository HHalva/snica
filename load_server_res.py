import numpy as np
import pdb
import codecs
filename = "k_res_0.txt"

for i in range(20):
    with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        max_elbo = -99999999999
        for line in f:
            if "mcc" in line and "minibatch" not in line:
                es = line.split("\t")[3]
                try:
                    _, seedval = es.split(" ")
                except:
                    continue 
                if int(seedval) == i+1:
                    lb = line.split("\t")[1]
                    _, elboval = lb.split(" ")
                    if float(elboval) > max_elbo:
                        cur_seed = int(seedval)
                        max_elbo = float(elboval)
                        mcc = line.split("\t")[2]
                        _, max_elbo_mcc = mcc.split(" ")
    print( "eseed:", cur_seed,
          "max ELBO:", '{:,}'.format(max_elbo).replace(',', ' '), 
          "mcc:", float(max_elbo_mcc))
