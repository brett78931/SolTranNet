"""
Code based on:
Shang et al "Edge Attention-based Multi-Relational Graph Convolutional Networks" -> https://github.com/Luckick/EAGCN
Coley et al "Convolutional Embedding of Attributed Molecular Graphs for Physical Property Prediction" -> https://github.com/connorcoley/conv_qsar_fast
"""
import csv
import sys
try:
    from rdkit import Chem
    from rdkit.Chem import MolFromSmiles
except:
    sys.exit('rdkit is not installed. Install with:\nconda install -c rdkit rdkit')

import logging
import numpy as np
import torch
import sys
from torch.utils.data import Dataset
import os
import pandas as pd
import pickle

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor

def load_data_from_df(dataset_path, add_dummy_node=True, one_hot_formal_charge=False, use_data_saving=True):
    '''load and featurize data stored in a CSV file
    
    Args:
        dataset_path (str): A path to the CSV file containing the data. It should have two columns:
                            the first one contains SMILES strings of the compounds,
                            the second one contains labels.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded. Defaults to False.
        use_data_saving (bool): If True, saved features will be loaded from the dataset directory; if no feature file
                                is present, the features will be saved after calculations. Defaults to True.
                                
    Returns:
        A tuple (X, y) in which X is a list of graph descriptors (node features, adjacency matrices, distance matrices),
        and y is a list of the corresponding labels.
    '''
    
    #feat_stamp = f'{"_dn" if add_dummy_node else ""}{"_ohfc" if one_hot_formal_charge else ""}'
    #feature_path = dataset_path.replace('.csv', f'{feat_stamp}.p')
    #if use_data_saving and os.path.exists(feature_path):
    #    logging.info(f"Loading features stored at '{feature_path}'")
    #    x_all, y_all = pickle.load(open(feature_path, "rb"))
    #    return x_all, y_all
    
    data_df = pd.read_csv(dataset_path)
    
    data_x = data_df.iloc[:, 0].values
    data_y = data_df.iloc[:, 1].values
    
    if data_y.dtype == np.float64:
        data_y = data_y.astype(np.float32)
        
    x_all, y_all = load_data_from_smiles(data_x, data_y, add_dummy_node=add_dummy_node, one_hot_formal_charge=one_hot_formal_charge)
    
    #if use_data_saving and not os.path.exists(feature_path):
    #        logging.info(f"Saving features at '{feature_path}'")
    #        pickle.dump((x_all, y_all), open(feature_path, "wb"))
            
    return x_all, y_all

def load_data_from_smiles(x_smiles, labels, add_dummy_node=True, one_hot_formal_charge=True):
    """Load and featurize data from lists of SMILES strings and labels.

    Args:
        x_smiles (list[str]): A list of SMILES strings.
        labels (list[float]): A list of the corresponding labels
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded. Defaults to True.

    Returns:
        A tuple of lists of graph descriptors (node features, adjacency matrices)
    """

    x_all, y_all = [], []
    
    for smiles, label in zip(x_smiles, labels):
        try:
            name = smiles
            mol = MolFromSmiles(smiles)
            afm, adj = featurize_mol(mol, name, add_dummy_node)
            x_all.append([afm, adj])
            y_all.append([label])
        except ValueError as e:
            logging.warning('the SMILES ({}) can not be converted to a graph.\nREASON: {}'.format(smiles,e))

    return x_all, y_all

def featurize_mol(mol, name, add_dummy_node):
    """Featurize molecule.

    Args:
        mol (rdchem.Mol): An RDKit Mol object.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph.

    Returns:
        A tuple of molecular graph descriptors (node features, adjacency matrix).
    """
    

    atom_features = np.array([get_atom_features(atom) for atom in mol.GetAtoms()])
    
    dict_from_csv = {}
          
    with open('Solubility_GFN2_alpb_SolTranNet.csv', mode='r') as inp:
        reader = csv.reader(inp)
        dict_from_csv = {rows[1]:rows[0] for rows in reader}
        
    #smile = Chem.MolToSmiles(mol)    
    
    
    keyv = dict_from_csv[name]
        
    with open('Q_Data/'+keyv+'.csv') as q_data:
        qarray = np.loadtxt(q_data, delimiter=',', skiprows=1, usecols=(4,5,6,7))
            #node_features = qarray
    
    node_features = np.concatenate((atom_features, qarray), axis=1)

    
    adj_matrix = np.eye(mol.GetNumAtoms())
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom().GetIdx()
        end_atom = bond.GetEndAtom().GetIdx()
        adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1

    if add_dummy_node:
        m = np.zeros((node_features.shape[0] + 1, node_features.shape[1] + 1))
        m[1:, 1:] = node_features
        m[0, 0] = 1.
        node_features = m

        m = np.zeros((adj_matrix.shape[0] + 1, adj_matrix.shape[1] + 1))
        m[1:, 1:] = adj_matrix
        adj_matrix = m

    return node_features, adj_matrix


#lookup table for one-hot elements
anummap = {5:0, 6: 1, 7:2, 8: 3, 9: 4,  15:5, 16:6, 17:7, 35:8, 53:9 }
anumtable = np.full(128,10)
for i,val in anummap.items():
    anumtable[i] = val
    
def get_atom_features(atom):
    """Calculate atom features.

            Identity            -- [B,C,N,O,F,P,S,Cl,Br,I,Dummy,Other]
            #Heavy Neighbors    -- [0,1,2,3,4,5]
            #H atoms            -- [0,1,2,3,4]
            Formal Charge       -- [-1,0,1]
            Is in a Ring        -- [0,1]
            Is Aromatic         -- [0,1]
        Dummy and Other types, have the same one-hot encoding, but the dummy node is unconnected.
    Args:
        atom (rdchem.Atom): An RDKit Atom object.

    Returns:
        A 1-dimensional array (ndarray) of atom features.
    """
    attributes = np.zeros(27)    
    anum = atom.GetAtomicNum()        
    attributes[anumtable[anum]] = 1.0
        
    ncnt = min(len(atom.GetNeighbors()),5)
    attributes[11+ncnt] = 1.0

    hcnt = min(atom.GetTotalNumHs(),4)
    attributes[17+hcnt] = 1.0
    
    charge = atom.GetFormalCharge()
    if charge == 0:
        attributes[23] = 1.0
    elif charge < 0:
        attributes[22] = 1.0
    else:
        attributes[24] = 1.0
        
    attributes[25] = atom.IsInRing()
    attributes[26] = atom.GetIsAromatic()
    return attributes

def one_hot_vector(val, lst):
    """Converts a value to a one-hot vector based on options in lst"""
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)


class Molecule:
    """
        Class that represents a train/validation/test datum
        - self.label: 0 neg, 1 pos -1 missing for different target.
    """

    def __init__(self, x, y, index, smile=''):
        self.node_features = x[0]
        self.adjacency_matrix = x[1]
        self.y = y
        self.index = index
        self.smile = smile
        


class MolDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_list):
        """
        @param data_list: list of Molecule objects
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        if type(key) == slice:
            return MolDataset(self.data_list[key])
        return self.data_list[key]


def pad_array(array, shape, dtype=np.float32):
    """Pad a 2-dimensional array with zeros.

    Args:
        array (ndarray): A 2-dimensional array to be padded.
        shape (tuple[int]): The desired shape of the padded array.
        dtype (data-type): The desired data-type for the array.

    Returns:
        A 2-dimensional array of the given shape padded with zeros.
    """
    padded_array = np.zeros(shape, dtype=dtype)
    padded_array[:array.shape[0], :array.shape[1]] = array
    return padded_array


def mol_collate_func(batch):
    """Create a padded batch of molecule features.

    Args:
        batch (list[Molecule]): A batch of raw molecules.

    Returns:
        A list of FloatTensors with padded molecule features: (adjacency matrices, node features).
    """
    adjacency_list, features_list, smiles_list, index_list = [], [], [], []
    labels = []
    
    max_size = 0
    for molecule in batch:
        if type(molecule.y[0]) == np.array:
            labels.append(molecule.y[0])
        else:
            labels.append(molecule.y)
        if molecule.adjacency_matrix.shape[0] > max_size:
            max_size = molecule.adjacency_matrix.shape[0]
        
    for molecule in batch:
        adjacency_list.append(pad_array(molecule.adjacency_matrix, (max_size, max_size)))
        features_list.append(pad_array(molecule.node_features, (max_size, molecule.node_features.shape[1])))
        smiles_list.append(molecule.smile)
        index_list.append(molecule.index)

    #do not use cuda memory during data loading
    #return [torch.FloatTensor(adjacency_list), torch.FloatTensor(features_list), smiles_list, index_list]
    return [FloatTensor(features) for features in (adjacency_list, features_list, labels)]
    
def construct_dataset(x_all, y_all):
    """Construct a MolDataset object from the provided data.

    Args:
        x_all (list): A list of molecule features.
        y_all (list): a list of the corresponding labels
    Returns:
        A MolDataset object filled with the provided data.
    """
    output = [Molecule(data[0], data[1], i)
              for i, data in enumerate(zip(x_all, y_all))]
    return MolDataset(output)

def construct_loader(x, y, batch_size, shuffle=True):
    """Construct a data loader for the provided data.

    Args:
        x (list): A list of molecule features.
        y (list): A list of the corresponding labels
        batch_size (int): The batch size. Defaults to 32
        shuffle (bool): If True the data will be loaded in a random order. Defaults to False.

    Returns:
        A DataLoader object that yields batches of padded molecule features.
    """
    data_set = construct_dataset(x, y)
    loader = torch.utils.data.DataLoader(dataset=data_set,batch_size=batch_size,
                                         collate_fn=mol_collate_func,shuffle=shuffle)
    return loader


class MolIterableDataset(torch.utils.data.IterableDataset):
    '''An iterable dataset over Molecule objects.'''
    
    def __init__(self, x_smiles, add_dummy_node=True):
        '''Initialize iterable dataset with list of smiles'''
        super(MolIterableDataset).__init__()
        self.x_smiles = x_smiles
        self.add_dummy_node=add_dummy_node
                
    def __iter__(self):
        
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            index = worker_info.id
            nworkers = worker_info.num_workers
        else:
            index = 0
            nworkers = 1
                    
        for i,smiles in enumerate(self.x_smiles):
            try:
                if i%nworkers == index:
                    mol = MolFromSmiles(smiles.split()[0])
                    afm, adj = featurize_mol(mol, self.add_dummy_node)
                    yield Molecule([afm, adj],i,smiles)
            except ValueError as e:
                logging.warning('the SMILES ({}) can not be converted to a graph.\nREASON: {}'.format(smiles,e))


def construct_loader_from_smiles(smiles, batch_size=32, num_workers=1, shuffle=False):
    """Construct a data loader for the provided data.

    Args:
        x (list): A list of molecule features
        y (list): A list of the corresponding labels
        smiles (list): A list of smiles.
        batch_size (int): The batch size. Defaults to 32
        shuffle (bool): If True the data will be loaded in a random order. Defaults to False.

    Returns:
        A DataLoader object that yields batches of padded molecule features.
    """
    #data_set = construct_dataset(x)
    #loader = torch.utils.data.DataLoader(dataset=data_set,batch_size=batch_size,collate_fn=mol_collate_func,shuffle=shuffle)
    data_set = MolIterableDataset(smiles)
    loader = torch.utils.data.DataLoader(dataset=data_set,batch_size=batch_size,collate_fn=mol_collate_func,shuffle=shuffle,num_workers=num_workers)
    return loader
