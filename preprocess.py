# From drug_graph.py
from rdkit import Chem
import numpy as np
import pandas as pd
import torch
from dgllife.utils import *


# From cellline_graph.py
import os
from torch_geometric.data import Data
from torch_scatter import scatter_add
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy import sparse
import pickle
from tqdm import tqdm

# Utils
from utils import save_data_stage

# From CANDLE
import argparse
from pathlib import Path
import candle

# IMPROVE imports
from improve import framework as frm
from improve import drug_resp_pred as drp

# CANDLE implementation of the DRPreter preprocess scripts

filepath = Path(__file__).resolve().parent

# [Req] App-specific params (App: monotherapy drug response prediction)
app_preproc_params = [
    # These arg should be specified in the [modelname]_default_model.txt:
    # y_data_files, x_data_canc_files, x_data_drug_files
    {
        "name": "y_data_files",
        "type": str,
        "help": "List of files that contain the y (prediction variable) data. \
        Example: [['response.tsv']]",
    },
    {
        "name": "x_data_canc_files",
        "type": str,
        "help": "List of feature files including gene_system_identifer. Examples: \n\
             1) [['cancer_gene_expression.tsv', ['Gene_Symbol']]] \n\
             2) [['cancer_copy_number.tsv', ['Ensembl', 'Entrez']]].",
    },
    {
        "name": "x_data_drug_files",
        "type": str,
        "help": "List of feature files. Examples: \n\
             1) [['drug_SMILES.tsv']] \n\
             2) [['drug_SMILES.tsv'], ['drug_ecfp4_nbits512.tsv']]",
    },
    {
        "name": "canc_col_name",
        "default": "improve_sample_id",
        "type": str,
        "help": "Column name in the y (response) data file that contains the cancer sample ids.",
    },
    {
        "name": "drug_col_name",
        "default": "improve_chem_id",
        "type": str,
        "help": "Column name in the y (response) data file that contains the drug ids.",
    },
]

# [DRPreter] Model-specific params (Model: DRPreter)
model_preproc_params = [
    {
        "name": "graph_type",
        "type": str,
        "default": "disjoint",
        "help": "Utilize a joint or disjoint graph",
    },
    {
        "name": "edge",
        "type": str,
        "default": "STRING",
        "help": "Where the information from the cellline graph edges is from.",
    },
    {
        "name": "string_edge",
        "type": str,
        "default": "990",
        "help": "The weight of the graph edges.",
    },
]

preprocess_params = app_preproc_params + model_preproc_params
req_preprocess_args = [ll["name"] for ll in preprocess_params]


# def download_string_dataset():
#     string_data = get_file(
#         "9606.proteins.links.v12.0.txt.gz",
#         "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
#         unpack=True,
#     )

#     return string_data


# def download_kegg_pathways():
#     kegg_pathway = get_file("34pathway_score990.pkl", "", datadir="./data")

#     return kegg_pathway


def save_cell_graph(gene_expression, params):
    graph_type = params["graph_type"]
    save_path = params["ml_data_outdir"]

    file_path = os.path.join(params["pathway_data_dir"], "34pathway_score990.pkl")
    with open(file_path, "rb") as file:
        kegg = pickle.load(file)

    if Path(Path(save_path) / f"cell_feature_std_{graph_type}.npy").exists():
        print("already exists!")

    else:
        exp = gene_expression
        exp = exp.set_index("improve_sample_id")
        index = exp.index
        columns = exp.columns

        scaler = StandardScaler()
        exp = scaler.fit_transform(exp)

        imp_mean = SimpleImputer()
        exp = imp_mean.fit_transform(exp)

        exp = pd.DataFrame(exp, index=index, columns=columns)

        cell_names = exp.index

        cell_dict = {}

        for i in tqdm((cell_names)):
            # joint graph (without pathway)
            if graph_type == "joint":
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
        return gene_list, cell_dict


def get_STRING_edges(ppi_threshold, graph_type, gene_list, params):
    save_path = params["ml_data_outdir"]
    data_path = Path(save_path) / f"edge_index_{ppi_threshold}_{graph_type}.npy"
    ppi_path = params["pathway_data_dir"]
    if not os.path.exists(data_path):
        # gene_list
        ppi = pd.read_csv(Path(ppi_path) / f"CCLE_2369_{ppi_threshold}.csv", index_col=0)

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
            os.path.join(save_path, f"edge_index_{ppi_threshold}_{graph_type}.npy"),
            edge_index,
        )
    else:
        edge_index = np.load(data_path)

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


def run(params):
    """Execute data pre-processing for GraphDRP model.

    :params: Dict params: A dictionary of CANDLE/IMPROVE keywords and parsed values.
    """
    # ------------------------------------------------------
    # [Req] Build paths and create ML data dir
    # ----------------------------------------
    # Build paths for raw_data, x_data, y_data, splits
    params = frm.build_paths(params)

    # Create outdir for ML data (to save preprocessed data)
    frm.create_outdir(outdir=params["ml_data_outdir"])
    # ----------------------------------------

    # ------------------------------------------------------
    # Construct data frames for drug and cell features
    # ------------------------------------------------------
    # df_drug, df_cell_all, smile_graphs = build_common_data(params, indtd)

    # ------------------------------------------------------
    # [Req] Load omics data
    # ---------------------
    print("\nLoading omics data ...")
    oo = drp.OmicsLoader(params)
    # print(oo)
    gene_expression = oo.dfs["cancer_gene_expression.tsv"]  # get only gene expression dataframe

    # ------------------------------------------------------
    # [DRPreter] Prep gene features
    # ------------------------------------------------------
    gene_list, cell_dict = save_cell_graph(gene_expression, params)

    edge_index = get_STRING_edges(
        ppi_threshold="PPI_990",
        graph_type="disjoint",
        gene_list=gene_list,
        params=params,
    )
    print(f"edge_index: {edge_index}")

    example = cell_dict["ACH-000001"]
    params["num_feature"] = example.x.shape[1]
    print(f'Num Features: {params["num_feature"]}')
    params["num_genes"] = example.x.shape[0]  # 4646
    print(f'Num Genes: {params["num_genes"]}')
    if "disjoint" in params["graph_type"]:
        gene_list = scatter_add(
            torch.ones_like(example.x.squeeze()), example.x_mask.to(torch.int64)
        ).to(torch.int)
        params["max_gene"] = gene_list.max().item()
        print(f"Max Gene: {params['max_gene']}")
        params["cum_num_nodes"] = torch.cat(
            [gene_list.new_zeros(1), gene_list.cumsum(dim=0)], dim=0
        )
        print(f"cum_num_nodes: {params['cum_num_nodes']}")
        params["n_pathways"] = gene_list.size(0)
        print(f"N Pathways: {params['n_pathways']}")
        print(f"gene distribution: {gene_list}")
        print(f"mean degree:{len(edge_index[0]) / params['num_genes']}")
    else:
        print(f"num_genes:{params['num_genes']}, num_edges:{len(edge_index[0])}")
        print(f"mean degree:{len(edge_index[0]) / params['num_genes']}")

    # [Req] Load drug data
    # --------------------
    print("\nLoading drugs data...")
    dd = drp.DrugsLoader(params)
    # print(dd)
    smi = dd.dfs["drug_SMILES.tsv"]  # get only the SMILES data
    # --------------------
    print(f"SMILES: {smi}")

    # ------------------------------------------------------
    # [DRPreter] Prep drug features
    # ------------------------------------------------------
    save_path = params["ml_data_outdir"]
    drug_dict = {}
    for i in range(len(smi)):
        drug_dict[smi.index[i]] = smiles2graph(smi.iloc[i, 0])
    np.save(os.path.join(save_path, "drug_feature_graph.npy"), drug_dict)

    # ------------------------------------------------------
    # [DRPreter] Construct Dataloaders
    # Construct ML data for every stage (train, val, test)
    # [Req] All models must load response data (y data) using DrugResponseLoader().
    # Below, we iterate over the 3 split files (train, val, test) and load response
    # data, filtered by the split ids from the split files.
    # -------------------------------------------
    stages = {
        "train": params["train_split_file"],
        "val": params["val_split_file"],
        "test": params["test_split_file"],
    }
    scaler = None

    for stage, split_file in stages.items():
        # ------------------------
        # [Req] Load response data
        # ------------------------
        rr = drp.DrugResponseLoader(params, split_file=split_file, verbose=True)
        # print(rr)
        df_response = rr.dfs["response.tsv"]
        # -----------------------

        rs, ge = drp.get_common_samples(
            df1=df_response,
            df2=gene_expression,
            ref_col=params["canc_col_name"],
        )
        print(
            rs[
                [
                    params["canc_col_name"],
                    params["drug_col_name"],
                ]
            ].nunique()
        )

        # Sub-select desired response column (y_col_name)
        # And reduce response dataframe to 3 columns: drug_id, cell_id and selected drug_response
        rs = rs[[params["drug_col_name"], params["canc_col_name"], params["y_col_name"]]]
        # Further prepare data (model-specific)
        # xd, xc, y = compose_data_arrays(
        #    ydf, smi, df_canc, params["drug_col_name"], params["canc_col_name"]
        # )
        # print(stage.upper(), "data --> xd ", xd.shape, "xc ", xc.shape, "y ", y.shape)
        # ------------------------

        # -----------------------
        # [Req] Save ML data files in params["ml_data_outdir"]
        # The implementation of this step, depends on the model.
        # -----------------------
        # import ipdb; ipdb.set_trace()
        # [Req] Create data name
        data_fname = frm.build_ml_data_name(
            params,
            stage,
        )

        # Create the ml data and save it as data_fname in params["ml_data_outdir"]
        # Note! In the *train*.py and *infer*.py scripts, functionality should
        # be implemented to load the saved data.
        # -----
        # In GraphDRP, TestbedDataset() is used to create and save the file.
        # TestbedDataset() which inherits from torch_geometric.data.InMemoryDataset
        # automatically creates dir called "processed" inside root and saves the file
        # inside. This results in: [root]/processed/[dataset],
        # e.g., ml_data/processed/train_data.pt
        # -----

        save_data_stage(
            smi,
            ge,
            rs,
            edge_index=edge_index,
            data_name=data_fname,
            params=params,
        )

        # [Req] Save y dataframe for the current stage
        frm.save_stage_ydf(rs, params, stage)


def main():
    # Load in arguments needed for preprocessing
    additional_definitions = preprocess_params
    params = frm.initialize_parameters(
        filepath,
        default_model="drpreter_default_model.txt",
        additional_definitions=additional_definitions,
        required=req_preprocess_args,
    )
    processed_outdir = run(params)
    print("\nFinished DRPreter pre-processing (transformed raw DRP data to model input ML data).")


if __name__ == "__main__":
    main()
