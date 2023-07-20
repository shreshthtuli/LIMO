import os
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--smiles', type=str, default='zinc250k.smi')
args = parser.parse_args()
print('Preprocessing..')
dm = MolDataModuleSMI(512, args.smiles) # MolDataModuleSMI / MolDataModule
pickle.dump(dm, open('dmSMI.pkl', 'wb'))
print('Done!')