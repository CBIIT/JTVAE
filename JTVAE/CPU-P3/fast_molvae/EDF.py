import torch
import torch.nn as nn
from torch.autograd import Variable
from optparse import OptionParser

import rdkit
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import rdmolops

import numpy as np  
from fast_jtnn import *
import time
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore")
lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

#JI - Wrapper for encode_latent_mean 
def wrap_encode(smiles):
    warnings.filterwarnings("ignore")
    mol_v = model.encode_latent_mean(smiles)
    return mol_v

#JI - Wrapper for decode_2 
def wrap_decode_2(all_vec):
    warnings.filterwarnings("ignore")
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    smiles_recon = []
    for i in range(len(all_vec)):
        tree_vec,mol_vec = torch.split(torch.reshape(all_vec[i], (1,128)), 64, dim=1)
        smiles = model.decode_2(tree_vec, mol_vec, prob_decode=False)
        smiles_recon.append(smiles) 

    return smiles_recon

start_time = time.time()

#******************************************************************************** 
#** User available options and default settings below 
#** Model parameters should equal that used for training 
#** ncpus set to 18 default, use your choice keeping in mind queue limits
#** Batch default of 40 is good for 5000 SMILES, increase to 200 for big datasets 
#** Include -f if you want to print out all failed SMILES reconstructs
#** Include -p if you want lots of joblib.Parallel printing/output
#********************************************************************************

parser = OptionParser()
parser.add_option("-d", "--data", dest="data_path")   # File containing SMILES strings
parser.add_option("-v", "--vocab", dest="vocab_path") # File containing Vocabulary
parser.add_option("-m", "--model", dest="model_path") # File containg Model/Autoencoder
parser.add_option("-w", "--hidden", dest="hidden_size", default=450) # Hidden size in Model
parser.add_option("-l", "--latent", dest="latent_size", default=128) # Latent vector size in Model
parser.add_option("-t", "--depthT", dest="depthT", default=20)       # Tree depth in Model
parser.add_option("-g", "--depthG", dest="depthG", default=3)        # Graph depth in Model
parser.add_option("-c", "--ncpus", dest="ncpus", default=18) # Number of core/processes to use
parser.add_option("-b", "--bats", dest="bats", default=40)   # Batch size in joblib
parser.add_option("-f", "--fail", action="store_true", dest="pr_fail", default=False) 
parser.add_option("-p", "--verb", action="store_true", dest="verbose", default=False)

opts,args = parser.parse_args()

hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depthT = int(opts.depthT)
depthG = int(opts.depthG)
n_cpus = int(opts.ncpus)
bat_size = int(opts.bats)
pr_fail = bool(opts.pr_fail)
verbose = bool(opts.verbose)

if verbose == True:
    n_verb = 100  
else:
    n_verb = 0

with open(opts.data_path) as f:
    smiles_list = [line.strip("\r\n ").split()[0] for line in f]

#JI - We generate smiles_rdkit here now rather than later
#JI - Note that currently this Fast-JTVAE doesn't use stereochemistry
 
smiles_rdkit = []
for s in smiles_list:
    mol = MolFromSmiles(s)
    smi = MolToSmiles(mol,isomericSmiles=False)
    smiles_rdkit.append(smi)

n_data = len(smiles_rdkit)
vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)] 
vocab = Vocab(vocab)

#JI - Load Fast-JTVAE Model/Autoencoder

model = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
model.load_state_dict(torch.load(opts.model_path, map_location='cpu'))
model = model.cpu()

print ('Encoding-Decoding is using   ', n_cpus, 'processes/cores')
print ('Number of SMILES           = ', n_data)
print ('Batch size in joblib       = ', bat_size)
print ('Verbose in joblib.Parallel = ', verbose)
print ('Print failed reconstructs  = ', pr_fail)
print ('Starting latent_vectors: Total time (to load data/model) = %.0f seconds \n' % (time.time() - start_time))
curr_time = time.time()

smi_recon = []
smi_list = []
fail = [[] for x in range(n_data)]

batches = [smiles_rdkit[i:i+bat_size] for i in range(0, n_data, bat_size)]
num_good = 0

#JI - Encode SMILES in parallel

all_vec = Parallel(n_jobs=n_cpus,batch_size=1,max_nbytes=None,mmap_mode=None,verbose=n_verb)\
          (delayed(wrap_encode)(batch) for batch in batches)
print ('Encoding computation time, Total time = %.0f, %0.f seconds' % \
      ((time.time() - curr_time), (time.time() - start_time)))
curr_time = time.time()

#JI - Decode SMILES in parallel

smi_recon = Parallel(n_jobs=n_cpus,batch_size=1,max_nbytes=None,mmap_mode=None,verbose=n_verb)\
            (delayed(wrap_decode_2)(all_vec[i]) for i in range(len(all_vec)))
smi_list = np.concatenate(smi_recon).flat
print ('Decoding computation time, Total time = %.0f, %0.f seconds' % \
      ((time.time() - curr_time), (time.time() - start_time)))
curr_time = time.time()

#JI - Count number of decoded SMILES' = SMILES (original)

if pr_fail == False:
    for j in range(0, n_data):
        if smi_list[j] == smiles_rdkit[j]: 
            num_good += 1
else:
    for j in range(0, n_data):
        if smi_list[j] == smiles_rdkit[j]: 
            num_good += 1
            continue
        fail[j].append(smi_list[j])
    
per_good = (float(num_good)/float(n_data))*100.0
print ('Number, percent of successfully reconstructed SMILES = %d, %.2f %%' % \
      (num_good, per_good))
print ()
if pr_fail == True:
    print ('Failed Encode-Decode reconstructs are below: \n')
    for i in range(n_data):
        if len(fail[i]) > 0: print (i+1, ' ', smiles_rdkit[i], ' ', *fail[i])

