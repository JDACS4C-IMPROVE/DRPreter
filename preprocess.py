# From drug_graph.py
from rdkit import Chem
import numpy as np
import pandas as pd
import torch
import torch_geometric
from dgllife.utils import *

# From cellline_graph.py
import os
import csv
import scipy
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import graclus, max_pool
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy import sparse
import pickle
from tqdm import trange, tqdm

# From CANDLE
import sys
import argparse
from pathlib import Path
import candle
import improve_utils as imp
from improve_utils import improve_globals as ig


# CANDLE implementation of the DRPreter preprocess scripts

fdir = Path(__file__).resolve().parent

with open("Data/Cell/34pathway_score990.pkl", "rb") as file:
    kegg = pickle.load(file)

"""
Functions below are used to generate Cell Line Graph
"""


def download_string_dataset():
    string_data = get_file(
        "9606.proteins.links.v12.0.txt.gz",
        "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
        unpack=True,
    )

    return string_data


def download_kegg_pathways():
    kegg_pathway = get_file("34pathway_score990.pkl", "", datadir="./data")

    return kegg_pathway


def save_cell_graph(gene_expression, save_path, graph_type):
    if Path(Path(save_path) / "cell_feature_std_{type}.npy").exists():
        print("already exists!")
    else:
        exp = gene_expression
        exp.columns = exp.columns.str[3:]
        index = exp.index
        columns = exp.columns

        scaler = StandardScaler()
        exp = scaler.fit_transform(exp)
        # cn = scaler.fit_transform(cn)

        imp_mean = SimpleImputer()
        exp = imp_mean.fit_transform(exp)

        exp = pd.DataFrame(exp, index=index, columns=columns)
        # cn = pd.DataFrame(cn, index=index, columns=columns)
        # mu = pd.DataFrame(mu, index=index, columns=columns)
        cell_names = exp.index

        cell_dict = {}

        for i in tqdm((cell_names)):
            # joint graph (without pathway)
            if type == "joint":
                gene_list = exp.columns.to_list()
                gene_list = set()
                for pw in kegg:
                    for gene in kegg[pw]:
                        if gene in exp.columns.to_list():
                            gene_list.add(gene)
                gene_list = list(gene_list)
                cell_dict[i] = Data(x=torch.tensor([exp.loc[i, gene_list]], dtype=torch.float).T)

            # disjoint graph (with pathway)
            else:
                genes = exp.columns.to_list()
                x_mask = []
                x = []
                gene_list = {}
                for p, pw in enumerate(list(kegg)):
                    gene_list[pw] = []
                    for gene in kegg[pw]:
                        if gene in genes:
                            gene_list[pw].append(gene)
                            x_mask.append(p)
                    x.append(exp.loc[i, gene_list[pw]])
                x = pd.concat(x)
                cell_dict[i] = Data(
                    x=torch.tensor([x], dtype=torch.float).T,
                    x_mask=torch.tensor(x_mask, dtype=torch.int),
                )

        print(cell_dict)
        np.save(os.path.join(save_path, f"cell_feature_std_{graph_type}.npy"), cell_dict)
        print("finish saving cell data!")
        return gene_list


def get_STRING_edges(gene_path, ppi_threshold, graph_type, gene_list):
    save_path = ig.ml_data_dir / f"edge_index_{ppi_threshold}_{graph_type}.npy"
    if not os.path.exists(save_path):
        # gene_list
        ppi = pd.read_csv(gene_path / f"CCLE_2369_{ppi_threshold}.csv", index_col=0)

        print("Loaded File")

        # joint graph (without pathway)
        if type == "joint":
            ppi = ppi.loc[gene_list, gene_list].values
            sparse_mx = sparse.csr_matrix(ppi).tocoo().astype(np.float32)
            edge_index = np.vstack((sparse_mx.row, sparse_mx.col))

        # disjoint graph (with pathway)
        else:
            edge_index = []
            for pw in gene_list:
                sub_ppi = ppi.loc[gene_list[pw], gene_list[pw]]
                sub_sparse_mx = sparse.csr_matrix(sub_ppi).tocoo().astype(np.float32)
                sub_edge_index = np.vstack((sub_sparse_mx.row, sub_sparse_mx.col))
                edge_index.append(sub_edge_index)
            edge_index = np.concatenate(edge_index, 1)

        # Conserve edge_index
        print(len(edge_index[0]))
        np.save(
            ig.ml_data_dir / f"edge_index_{ppi_threshold}_{graph_type}.npy",
            edge_index,
        )
    else:
        edge_index = np.load(save_path)

    return edge_index


# Drug Graph Preprocessing
def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    8 features are canonical, 2 features are from OGB
    """
    featurizer_funcs = ConcatFeaturizer(
        [
            atom_type_one_hot,
            atom_degree_one_hot,
            atom_implicit_valence_one_hot,
            atom_formal_charge,
            atom_num_radical_electrons,
            atom_hybridization_one_hot,
            atom_is_aromatic,
            atom_total_num_H_one_hot,
            atom_is_in_ring,
            atom_chirality_type_one_hot,
        ]
    )
    atom_feature = featurizer_funcs(atom)
    return atom_feature


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    featurizer_funcs = ConcatFeaturizer(
        [
            bond_type_one_hot,
            # bond_is_conjugated,
            # bond_is_in_ring,
            # bond_stereo_one_hot,
        ]
    )
    bond_feature = featurizer_funcs(bond)

    return bond_feature


def smiles2graph(mol):
    """
    Converts SMILES string or rdkit's mol object to graph Data object without remove salt
    :input: SMILES string (str)
    :return: graph object
    """

    if isinstance(mol, Chem.rdchem.Mol):
        pass
    else:
        mol = Chem.MolFromSmiles(mol)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    #     num_bond_features = 3  # bond type, bond stereo, is_conjugated
    num_bond_features = 1  # bond type
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr),
        dtype=torch.float,
    )

    return graph


def raw_to_preprocessed(args):
    root = ig.ml_data_dir
    os.makedirs(root, exist_ok=True)

    # download = True
    download = False
    if download:
        ftp_origin = f"https://ftp.mcs.anl.gov/pub/candle/public/improve/IMP_data"
        data_file_list = [f"data.{args.train_data_name}.zip"]
        for f in data_file_list:
            candle.get_file(
                fname=f,
                origin=os.path.join(ftp_origin, f.strip()),
                unpack=True,
                md5_hash=None,
                cache_subdir=None,
                datadir=ig.raw_data_dir,
            )

    # Load Train, Test, Val Response Data

    print("\nLoad response train data ...")
    rs_tr = imp.load_single_drug_response_data_v2(
        source=args.train_data_name,
        split_file_name=args.train_split_file_name,
        y_col_name=args.y_col_name,
        sep=",",
        verbose=True,
    )

    print("\nLoad response val data ...")
    rs_vl = imp.load_single_drug_response_data_v2(
        source=args.val_data_name,
        split_file_name=args.val_split_file_name,
        y_col_name=args.y_col_name,
        sep=",",
        verbose=True,
    )

    print("\nLoad response test data ...")
    rs_te = imp.load_single_drug_response_data_v2(
        source=args.test_data_name,
        split_file_name=args.test_split_file_name,
        y_col_name=args.y_col_name,
        sep=",",
        verbose=True,
    )

    # Load gene expression data
    ge_path = Path(ig.raw_data_dir) / f"x_data/ge.csv"
    ge = pd.read_csv(ig.gene_expression_file_path, sep=",", index_col=0)

    gene_list = save_cell_graph(ge, root, "disjoint")

    edge_index = get_STRING_edges(
        gene_path=fdir / "Data/Cell",
        ppi_threshold="PPI_990",
        graph_type="disjoint",
        gene_list=gene_list,
    )
    print(f"edge_index: {edge_index}")

    # Load SMILES and build drug features
    smi = imp.load_smiles_data()
    drug_dict = {}
    for i in range(len(smi)):
        drug_dict[smi.iloc[i, 0]] = smiles2graph(smi.iloc[i, 1])
    np.save(Path(root) / "drug_feature_graph.npy", drug_dict)  # Check this path

    return ig.ml_data_dir


def parse_args(args):
    """Parse input arguments"""

    parser = argparse.ArgumentParser()

    # IMPROVE Required args
    parser.add_argument(
        "--train_data_name",
        type=str,
        required=True,
        help="Data source name.",
    )
    parser.add_argument(
        "--val_data_name",
        type=str,
        default=None,
        required=False,
        help="Data target name (not required for GraphDRP).",
    )
    parser.add_argument(
        "--test_data_name",
        type=str,
        default=None,
        required=False,
        help="Data target name (not required for GraphDRP).",
    )
    parser.add_argument(
        "--train_split_file_name",
        type=str,
        nargs="+",
        required=True,
        help="The path to the file that contains the split ids (e.g., 'split_0_tr_id',  'split_0_vl_id').",
    )
    parser.add_argument(
        "--val_split_file_name",
        type=str,
        nargs="+",
        required=True,
        help="The path to the file that contains the split ids (e.g., 'split_0_tr_id',  'split_0_vl_id').",
    )
    parser.add_argument(
        "--test_split_file_name",
        type=str,
        nargs="+",
        required=True,
        help="The path to the file that contains the split ids (e.g., 'split_0_tr_id',  'split_0_vl_id').",
    )
    parser.add_argument(
        "--y_col_name",
        type=str,
        required=True,
        help="Drug sensitivity score to use as the target variable (e.g., IC50, AUC).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output dir to store the generated ML data files (e.g., 'split_0_tr').",
    )
    parser.add_argument("--receipt", type=str, required=False, help="...")

    args = parser.parse_args(args)
    return args


def main(args):
    # Load in arguments needed for preprocessing
    args = parse_args(args)
    ml_data_path = raw_to_preprocessed(args)
    print(f"\nML data path:\t\n{ml_data_path}")
    print("\nFinished pre-processing (transformed raw DRP data to model input ML data).")

    return ml_data_path


"""
Main Run Method
"""

if __name__ == "__main__":
    main(sys.argv[1:])
