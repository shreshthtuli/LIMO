from .wordextract import *
from .cmethods import *
from tqdm import tqdm
import sys, pickle

def loadEmbeddings(LRNPATH):
    embeddings_index = {}

    f = open(os.path.join(LRNPATH)) #'word.11l.100d.txt'
    next(f)
    vsize = 0
    for line in f:
        values = line.split()
        word = values[0]
        vsize = len(values)-1
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    return embeddings_index, vsize




def getSMIVector(LINGOembds, smiles, q=8, wordOrChar="wd"):


    lingoList = []
    if wordOrChar == "wd":
        lingoList = createLINGOs(smiles, q)
    #elif wordOrChar == "ch":
    #    lingoList = createCHRs(smiles, "l") #ligand, q=1

    smilesVec = vectorAddAvg(LINGOembds, lingoList)

    return smilesVec


def returnSMIVector(smiles_path, emb_file = './source/utils/drug.pubchem.canon.l8.ws20.txt'):
    EMB, vsize = loadEmbeddings(emb_file)

    smiless = [line.strip() for line in open(smiles_path)]
    print("Constructing SMILES vectors..")
    smiVectors = [] 
    for smi in tqdm(smiless):
        smiVectors.append(getSMIVector(EMB, smi))

    return smiVectors


if __name__=="__main__":
    #emb_file, smiles_path
    returnSMIVector(emb_path, smiles_path)
