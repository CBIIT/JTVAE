import sys
import gzip
import pickle
import rdkit.Chem as Chem
#from rdkit.Chem import Draw            #STB
#from rdkit.Chem import Descriptors     #STB

from optparse import OptionParser 

def save_object(obj, filename):
    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
    dest.close()

def load_object(filename):
    with gzip.GzipFile(filename, 'rb') as source: result = source.read()
    ret = pickle.loads(result, encoding='latin1') 
    source.close()
    return ret

parser = OptionParser()
parser.add_option("-d", "--direct", dest="res_dir") # Results directory prefix, e.g. RESULTS-
parser.add_option("-n", "--niters", dest="ite_num") # Number of iterations to collate resuls for
opts,args = parser.parse_args()

ite_num = int(opts.ite_num)

all_smiles = []
smiles_list = []

#for i in range(1,11):
for i in range(1,4):
   for j in range(ite_num):
       fn = opts.res_dir + '%d/scores%d.dat' % (i, j)
       scores = load_object(fn)
       fn = opts.res_dir + '%d/valid_smiles%d.dat' % (i, j)
       smiles = load_object(fn)

       for k in range(len(smiles)):
           if smiles[k] not in smiles_list: 
               smiles_list.append(smiles[k])
               all_smiles.extend(zip([smiles[k]], [scores[k]]))
       
all_smiles = [(x,-y) for x,y in all_smiles]
all_smiles = sorted(all_smiles, key=lambda x:x[1], reverse=True)
for s,v in all_smiles:
    print (s,v)

#mols = [Chem.MolFromSmiles(s) for s,_ in all_smiles[:50]]
#vals = ["%.2f" % y for _,y in all_smiles[:50]]
#img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200,135), legends=vals, useSVG=True)
#print img
