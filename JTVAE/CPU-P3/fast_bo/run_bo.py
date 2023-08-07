import pickle

import gzip
from sparse_gp import SparseGP
import scipy.stats as sps
import numpy as np
import os.path

import rdkit
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops

import torch
import torch.nn as nn
from fast_jtnn import create_var, JTNNVAE, Vocab
import time
import sascorer
import networkx as nx

from optparse import OptionParser



lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

# We define the functions used to load and save objects
def save_object(obj, filename):
    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
    dest.close()

def load_object(filename):
    with gzip.GzipFile(filename, 'rb') as source: result = source.read()
    ret = pickle.loads(result)
    source.close()
    return ret

start_time = time.time()
curre_time = start_time

parser = OptionParser()
parser.add_option("-v", "--vocab", dest="vocab_path")   # Vocabulary file
parser.add_option("-m", "--model", dest="model_path")   # Model location
parser.add_option("-d", "--data_dir", dest="data_dir")  # Data directory
parser.add_option("-r", "--save_dir", dest="save_dir")  # Results directory
parser.add_option("-w", "--hidden", dest="hidden_size", default=450) 
parser.add_option("-l", "--latent", dest="latent_size", default=128)
parser.add_option("-t", "--depthT", dest="depthT", default=20)
parser.add_option("-g", "--depthG", dest="depthG", default=3)
parser.add_option("-s", "--seed", dest="random_seed", default=None) # Random seed 1-10
parser.add_option("-i", "--iters", dest="num_iters", default=5)     # Num iters
parser.add_option("-k", "--keep", dest="keep_per", default=100)     # % top scores to use
opts,args = parser.parse_args()

print("args input")
vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)] 
vocab = Vocab(vocab)

hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depthT = int(opts.depthT)
depthG = int(opts.depthG)
random_seed = int(opts.random_seed)
num_iters = int(opts.num_iters)
keep_per  = int(opts.keep_per)

if not os.path.exists(opts.save_dir): os.makedirs(opts.save_dir) 

model = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
model.load_state_dict(torch.load(opts.model_path, map_location='cpu'))
model = model.cpu()

# We load the random seed
np.random.seed(random_seed)

# We load the data (y is minused!)
X = np.loadtxt(opts.data_dir + '/latent_features.txt')
y = -np.loadtxt(opts.data_dir + '/targets.txt')

#JI - Sort by score, and keep (keep_per %) of top molecules by score for Sparse GP training
print("sort by score")
if keep_per != 100:
    num_keep = int(float(len(X))*float(keep_per)/100.0)
    print ('Using top %.2f %% = %d molecules for Sparse GP training' % (keep_per, num_keep))
    sort_X = []
    sorted_idxs = np.argsort(y)
    sort_y = np.array(y)[sorted_idxs]
    for i in range(len(X)):
        sort_X.append(X[sorted_idxs[i]])

    X = sort_X[0:num_keep]
    X = np.array(X)
    y = sort_y[0:num_keep]

y = y.reshape((-1, 1))

n = X.shape[ 0 ]
permutation = np.random.choice(n, n, replace = False)

X_train = X[ permutation, : ][ 0 : int(np.round(0.9 * n)), : ]
X_test = X[ permutation, : ][ int(np.round(0.9 * n)) :, : ]

y_train = y[ permutation ][ 0 : int(np.round(0.9 * n)) ]
y_test = y[ permutation ][ int(np.round(0.9 * n)) : ]

np.random.seed(random_seed)

logP_values = np.loadtxt(opts.data_dir + '/logP_values.txt')
SA_scores = np.loadtxt(opts.data_dir + '/SA_scores.txt')
cycle_scores = np.loadtxt(opts.data_dir + '/cycle_scores.txt')
SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)

iteration = 0
while (iteration < num_iters):
# We fit the GP
    np.random.seed(iteration * random_seed)
    M = 500
    sgp = SparseGP(X_train, 0 * X_train, y_train, M)
    sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0, y_test, minibatch_size = 10 * M, max_iterations = 100, learning_rate = 0.001)

    print ()
    print ('Iteration = ', iteration)
    print ('Time to do sgp.train_via_ADAM, Total time = %.0f, %.0f' % \
          ((time.time()-curre_time), (time.time()-start_time)))
    curre_time = time.time()

    pred, uncert = sgp.predict(X_test, 0 * X_test)
    error = np.sqrt(np.mean((pred - y_test)**2))
    testll = np.mean(sps.norm.logpdf(pred - y_test, scale = np.sqrt(uncert)))
    print ('Test RMSE: ', error)
    print ('Test ll: ', testll)

    pred, uncert = sgp.predict(X_train, 0 * X_train)
    error = np.sqrt(np.mean((pred - y_train)**2))
    trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
    print ('Train RMSE: ', error)
    print ('Train ll: ', trainll)

# We get 60 new/optimized latent vectors/molecules
    next_inputs = sgp.batched_greedy_ei(60, np.min(X_train, 0), np.max(X_train, 0))

    print ()
    print ('Time to generate new molecules, Total time = %.0f, %.0f' % \
          ((time.time()-curre_time), (time.time()-start_time)))
    curre_time = time.time()

    valid_smiles = []
    new_features = []
    for i in range(60):
        all_vec = next_inputs[i].reshape((1,-1))
        tree_vec,mol_vec = np.hsplit(all_vec, 2)
        tree_vec = create_var(torch.from_numpy(tree_vec).float())
        mol_vec = create_var(torch.from_numpy(mol_vec).float())
        s = model.decode_2(tree_vec, mol_vec, prob_decode=False)
        if s is not None: 
            valid_smiles.append(s)
            new_features.append(all_vec)
    
    print (len(valid_smiles), 'molecules are found')
    new_features = np.vstack(new_features)

    print ()
    print ('Time to decode new molecules, Total time = %.0f, %.0f' % \
          ((time.time()-curre_time), (time.time()-start_time)))
    curre_time = time.time()

    scores = []
    for i in range(len(valid_smiles)):
        current_log_P_value = Descriptors.MolLogP(MolFromSmiles(valid_smiles[ i ]))
        current_SA_score = -sascorer.calculateScore(MolFromSmiles(valid_smiles[ i ]))
        cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(valid_smiles[ i ]))))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([ len(j) for j in cycle_list ])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6

        current_cycle_score = -cycle_length
     
        current_SA_score_normalized = (current_SA_score - np.mean(SA_scores)) / np.std(SA_scores)
        current_log_P_value_normalized = (current_log_P_value - np.mean(logP_values)) / np.std(logP_values)
        current_cycle_score_normalized = (current_cycle_score - np.mean(cycle_scores)) / np.std(cycle_scores)

        score = current_SA_score_normalized + current_log_P_value_normalized + current_cycle_score_normalized
        scores.append(-score) #target is always minused

    save_object(scores, opts.save_dir + "/scores{}.dat".format(iteration))
    save_object(valid_smiles, opts.save_dir + "/valid_smiles{}.dat".format(iteration))

    sorted_idxs = np.argsort(scores)
    sort_smiles = np.array(valid_smiles)[sorted_idxs]
    sort_scores = np.array(scores)[sorted_idxs]

    for i in range(len(valid_smiles)): print (sort_smiles[i],sort_scores[i])

    if len(new_features) > 0:
        X_train = np.concatenate([ X_train, new_features ], 0)
        y_train = np.concatenate([ y_train, np.array(scores)[ :, None ] ], 0)

    iteration += 1

    print ()
    print ('Time to score and save new molecules, Total time = %.0f, %.0f' % \
          ((time.time()-curre_time), (time.time()-start_time)))

