from optparse import OptionParser

import rdkit
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import rdmolops
import sascorer
import networkx as nx
import time
import numpy as np  
import warnings
warnings.filterwarnings("ignore")

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("-d", "--data", dest="data_path")
parser.add_option("-o", "--dir", dest="dir_out", default="")

opts,args = parser.parse_args()

start_time = time.time()

with open(opts.data_path) as f:
    smiles_list = [line.strip("\r\n ").split()[0] for line in f]

#JI - We generate smiles_rdkit here now rather than later
#JI - Note that currently this Fast-JTVAE doesn't use stereochemistry
 
smiles_rdkit = []
for s in smiles_list:
    mol = MolFromSmiles(s)
    smi = MolToSmiles(mol,isomericSmiles=False)
    smiles_rdkit.append(smi)

print ('Starting property computation: Total time (to load data) = %.0f' % (time.time()-start_time))
curr_time = time.time()

#JI - Computation of properties for logP octanol-water partition coefficient test case

print ()
print ('Starting logP_values: total time =    %.0f' % (curr_time-start_time))

logP_values = []
for i in range(len(smiles_rdkit)):
    logP_values.append(Descriptors.MolLogP(MolFromSmiles(smiles_rdkit[ i ])))

print ('logP_values computation time =        %.0f' % (time.time()-curr_time))
print ()
print ('Starting SA_scores: total time =      %.0f' % (time.time() - start_time))
curr_time = time.time()

SA_scores = []
for i in range(len(smiles_rdkit)):
    SA_scores.append(-sascorer.calculateScore(MolFromSmiles(smiles_rdkit[ i ])))

print ('SA_scores computation time =          %.0f' % (time.time() - curr_time))
print ()
print ('Starting cycle_scores: total time =   %.0f' % (time.time() - start_time))
curr_time = time.time()

cycle_scores = []
for i in range(len(smiles_rdkit)):
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(smiles_rdkit[ i ]))))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([ len(j) for j in cycle_list ])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_scores.append(-cycle_length)

SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)

print ('cycle_scores computation time =       %.0f' % (time.time() - curr_time)) 
print ()

targets = SA_scores_normalized + logP_values_normalized + cycle_scores_normalized
np.savetxt(opts.dir_out+'/targets.txt', targets)
np.savetxt(opts.dir_out+'/logP_values.txt', np.array(logP_values))
np.savetxt(opts.dir_out+'/SA_scores.txt', np.array(SA_scores))
np.savetxt(opts.dir_out+'/cycle_scores.txt', np.array(cycle_scores))

print ('Total run time = %.0f' % (time.time()-start_time))

