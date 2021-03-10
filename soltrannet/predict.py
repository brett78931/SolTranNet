import torch
import numpy as np
from .data_utils import load_data_from_smiles
from .data_utils import construct_loader

def predict(smiles, batch_size=32):
    """Predict Solubilities for a list of SMILES.
    Args:
        smiles ([str]): A list of SMILES strings, upon which we wish to predict the solubilities for.
    Returns:
        A list of tuples (prediction, SMILES, Warning).
    """

    #first we calculate the molecular graphs from the SMILES strings
    X = load_data_from_smiles(smiles)
    assert X[0][0].shape[1]==28
    data_loader = construct_loader(X, batch_size=batch_size)
    
    #Then we ensure the model is set up properly
    weights=pkg_resources.resource_filename(__name__,"soltrannet_aqsol_trained.weights")
    model=make_model()#self.model
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device=torch.device("cuda")
        model.load_state_dict(torch.load(weights))
        model.to(device)
    else:
        device=torch.device('cpu')
        model.load_state_dict(torch.load(weights,map_location=device))
    model.eval()

    #Now we can generate our predictions.
    predictions=np.array([])
    with torch.no_grad():
        for batch in data_loader:
            adjacency_matrix, node_features = batch
            batch_mask = torch.sum(torch.abs(node_features),dim=-1) != 0
            pred = model(node_features, batch_mask, adjacency_matrix, None)
            predictions=np.append(predictions,pred.tolist())


    #lastly we determine which SMILES might have less accurate predictions
    warnings=check_smiles(smiles, X)

    return [(pred, smi, warn) for pred, smi, warn in zip(predictions, smiles, warnings)]

def check_smiles(smiles, features):
    """Check the inputted SMILES to see if the predictions will be less reliable. E.G. are a salt or have Other typed atoms

    Args:
        smiles ([str]): A list of SMILES strings
        features [(node features, adjacency matrix)]: A list of the graph represenation for the corresponding SMILES string.

    Returns:
        A list of warning strings to be output with the predictions. The string will be empty if nothing was detected.

    """
    warnings = []

    for smi, f in zip(smiles,features):
        warn=""

        #start with a check for salts
        if '.' in smi:
            warn+='Salt, '

        #use the calculated molecular features to determine if an "Other" typed atom exists in the molecule.
        other_type=False
        other_filter=np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        if (f[0]*other_filter).sum() >= 1:
            other_type=True

        if other_type:
            warn+='Other-typed Atom(s), '

        if warn!='':
            warn+='Detected Prediction less reliable'
        warnings.append(warn)

    return warnings