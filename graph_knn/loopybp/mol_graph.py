import rdkit
import rdkit.Chem as Chem
import numpy as np

elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
bond_fdim = 6
max_nb = 6

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return map(lambda s: x == s, allowable_set)

def atom_features(atom):
    return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetExplicitValence(), [1,2,3,4,5,6])
            + onek_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5])
            + [atom.GetIsAromatic()], dtype=np.float32)

def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()], dtype=np.float32)

def smiles2graph(smiles, idxfunc=lambda x:x.GetIdx()):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Could not parse smiles string:", smiles)

    n_atoms = mol.GetNumAtoms()
    n_bonds = max(mol.GetNumBonds(), 1)
    fatoms = np.zeros((n_atoms, atom_fdim))
    fbonds = np.zeros((n_bonds*2, atom_fdim + bond_fdim))
    atom_nb = np.zeros((n_atoms, n_bonds*2), dtype=np.int32)
    bond_nb = np.zeros((n_bonds*2, n_bonds*2), dtype=np.int32)
    
    for atom in mol.GetAtoms():
        idx = idxfunc(atom)
        fatoms[idx] = atom_features(atom)

    inbonds = [[] for i in xrange(n_atoms)]
    bonds = [0] * (n_bonds + n_bonds)
    for bond in mol.GetBonds():
        a1 = idxfunc(bond.GetBeginAtom())
        a2 = idxfunc(bond.GetEndAtom())
        idx = bond.GetIdx()
        bonds[idx] = (a1,a2)
        bonds[idx+n_bonds] = (a2,a1)
        inbonds[a2].append(idx)
        inbonds[a1].append(idx+n_bonds)
        atom_nb[a1,idx+n_bonds] = 1
        atom_nb[a2,idx] = 1
        fbonds[idx,:atom_fdim] = fatoms[a1]
        fbonds[idx,atom_fdim:] = bond_features(bond)
        fbonds[idx+n_bonds,:atom_fdim] = fatoms[a2]
        fbonds[idx+n_bonds,atom_fdim:] = bond_features(bond)
    
    for i in xrange(len(bonds)):
        if type(bonds[i]) is int:
            continue
        a1,a2 = bonds[i]
        for j in inbonds[a1]:
            if bonds[j][0] != a2:
                bond_nb[i,j] = 1

    return fatoms, fbonds, atom_nb, bond_nb

def pack2D(arr_list):
    N = max([x.shape[0] for x in arr_list])
    M = max([x.shape[1] for x in arr_list])
    a = np.zeros((len(arr_list), N, M))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        m = arr.shape[1]
        a[i,0:n,0:m] = arr
    return a

def get_mask(arr_list):
    N = max([x.shape[0] for x in arr_list])
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        for j in xrange(arr.shape[0]):
            a[i][j] = 1
    return a

def smiles2graph_list(smiles_list, idxfunc=lambda x:x.GetIdx()):
    res = map(lambda x:smiles2graph(x,idxfunc), smiles_list)
    fatom_list, fbond_list, gatom_list, gbond_list = zip(*res)
    return pack2D(fatom_list), pack2D(fbond_list), pack2D(gatom_list), pack2D(gbond_list), get_mask(fatom_list)

if __name__ == "__main__":
    np.set_printoptions(threshold='nan')
    #a,b,c,d = smiles2graph("c1cccnc1")
    a,b,c,d = smiles2graph('c1nccc2n1ccc2')
    print c
    print d
    print np.matmul(np.transpose(c), c) - d
