import torch
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import MACCSkeys

warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 1024
#a vector representation (1x2048) for molecular feature 
def morgan_binary_features_generator(mol,
                                     radius: int = MORGAN_RADIUS,
                                     num_bits: int = MORGAN_NUM_BITS):
    """
    Generates a binary Morgan fingerprint for a molecule.
    :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1D numpy array containing the binary Morgan fingerprint.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    #features_vec = MACCSkeys.GenMACCSKeys(mol)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features
    
def ECFP(PATH_x):
    df = pd.read_csv(PATH_x)
    fingerprints = []
    for smiles in df['SMILES']:
        fingerprint = morgan_binary_features_generator(smiles)
        fingerprints.append(fingerprint)

    df_num = np.array(fingerprints)
    return df_num
    

def create_dataset_number(PATH_x):
    df = pd.read_csv(PATH_x)
    df_num = ECFP (PATH_x)
    dfy = pd.read_csv(PATH_x)
    y = dfy['label'].values.reshape(-1,1)
    y = y.astype('float32')
    df_seq = dfy['SMILES']
    return df_num,y,df_seq


def func(PATH):
    df_num, y_true,df_seq = create_dataset_number(PATH)
    tensor_data_num = torch.tensor(df_num, dtype=torch.float32)
    y = torch.tensor([item for item in y_true]).to(torch.float)
    return df_seq, y, y_true,tensor_data_num