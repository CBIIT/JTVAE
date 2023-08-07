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

start_time = time.time()

#******************************************************************************** 
#** User available options and default settings below 
#** Model parameters should equal that used for training 
#** ncpus set to 36 default, use your choice keeping in mind queue limits
#** Batch default of 200 is good for 250K SMILES, decrease for smaller datasets
#** Include -p if you want lots of joblib.Parallel printing/output
#** Latent vectors are stored/output to file specifed by -o
#********************************************************************************

parser = OptionParser()
parser.add_option("-d", "--data", dest="data_path")   # File containing SMILES strings
parser.add_option("-v", "--vocab", dest="vocab_path") # File containing Vocabulary
parser.add_option("-m", "--model", dest="model_path") # File containg Model/Autoencoder
parser.add_option("-w", "--hidden", dest="hidden_size", default=450) # Hidden size in Model
parser.add_option("-l", "--latent", dest="latent_size", default=128) # Latent vector size in Model
parser.add_option("-t", "--depthT", dest="depthT", default=20)       # Tree depth in Model
parser.add_option("-g", "--depthG", dest="depthG", default=3)        # Graph depth in Model
parser.add_option("-c", "--ncpus", dest="ncpus", default=36) # Number of core/processes to use
parser.add_option("-b", "--bats", dest="bats", default=200)  # Batch size in joblib
parser.add_option("-p", "--verb", action="store_true", dest="verbose", default=False)
parser.add_option("-o", "--out", dest="lat_out", default="latent_features.txt")

opts,args = parser.parse_args()

hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depthT = int(opts.depthT)
depthG = int(opts.depthG)
n_cpus = int(opts.ncpus)
bat_size = int(opts.bats)
verbose = bool(opts.verbose)
lat_out = opts.lat_out

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

print ('Encoding is using            ', n_cpus, 'processes/cores')
print ('Number of SMILES           = ', n_data)
print ('Batch size in joblib       = ', bat_size)
print ('Verbose in joblib.Parallel = ', verbose)
print ('Latent vectors stored in   = ', lat_out)
print ('Starting latent_vectors: Total time (to load data/model) = %.0f seconds \n' % (time.time() - start_time))
curr_time = time.time()

latent_points = []
batches = [smiles_rdkit[i:i+bat_size] for i in range(0, n_data, bat_size)]

#JI - Encode SMILES in parallel

all_vec = Parallel(n_jobs=n_cpus,batch_size=1,max_nbytes=None,mmap_mode=None,verbose=n_verb)\
          (delayed(wrap_encode)(batch) for batch in batches)
print ('Encoding computation time, Total time = %.0f, %0.f seconds' % \
      ((time.time() - curr_time), (time.time() - start_time)))
#curr_time = time.time()

for i in range(0, len(all_vec)):
    latent_points.append(all_vec[i].data.cpu().numpy())

latent_points = np.vstack(latent_points)
np.savetxt(lat_out, latent_points)


