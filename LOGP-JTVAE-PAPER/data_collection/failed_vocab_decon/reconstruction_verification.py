vocab = []
with open("/mnt/projects/ATOM/blackst/GMD/LOGP-JTVAE-PAPER/Vocabulary/all_vocab.txt", 'r') as f:
    vocab = f.readlines()

print(len(vocab))

smiles = []
with open("/mnt/projects/ATOM/blackst/GMD/LOGP-JTVAE-PAPER/Raw-Data/ZINC/all.txt", 'r') as f:
    smiles = f.readlines()

print(len(smiles))

count = 0
for i in range(len(smiles)):
    if '\n' in smiles[i]:
        smiles[i] = smiles[i][:-1]
        count += 1

count == len(smiles)

count = 0
for i in range(len(vocab)):
    if '\n' in vocab[i]:
        vocab[i] = vocab[i][:-1]
        count += 1

count == len(vocab)

failed = []

for s in smiles:
    if s in vocab:
        failed.append(s)

with open("/mnt/projects/ATOM/blackst/GMD/LOGP-JTVAE-PAPER/data_collection/failed_vocab_decon/failed_vocab_smiles.txt", 'w') as f:
    for smile in failed:
        f.write(f"{smile}\n")