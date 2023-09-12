#!/usr/bin/env python3
"""Train DRPreter predictor."""
import os
import json
from pathlib import Path

import torch
from torch import nn
import numpy as np
import pandas as pd

from torch_scatter import scatter_add

import candle


from utils import (
    set_random_seed,
    train,
    validate,
    save_results,
    r4,
    get_path,
    EarlyStopping,
    load_sim_data,
    load_sim_graph,
    load_data,
)

from Model.DRPreter import DRPreter
from Model.Similarity import Similarity

file_path = os.path.dirname(os.path.realpath(__file__))

additional_definitions = [
    {"name": "", "type": str, "help": "Column name of drug ID"},
]

required = [
    "epochs",
    "model_name",
    "learning_rate",
    "patience",
    "output_dir",
]


class DRPreter_candle(candle.Benchmark):

    """Benchmark for DRPreter"""

    def set_locals(self):
        """Set parameters for the benchmark

        Args:
            required: set of required parameters for the benchmark.
            additional_definitions: list of dictionaries describing the additional
            parameters for the benchmarks

        """
        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions


def launch(args):
    set_random_seed(args.seed)

    rpath = file_path
    result_path = args.output_dir
    edge_type = "PPI_" + str(args.string_edge) if args.edge == "STRING" else args.edge
    edge_index = np.load(rpath + f"data/edge_index_{edge_type}_{args.dataset}.npy")

    data = pd.read_csv(rpath + "data/sorted_IC50_82833_580_170.csv")

    drug_dict = np.load(
        rpath + "data/drug_feature_graph.npy", allow_pickle=True
    ).item()  # pyg format of drug graph
    cell_dict = np.load(
        rpath + f"data/cell_feature_std_{args.dataset}.npy", allow_pickle=True
    ).item()  # pyg data format of cell graph

    example = cell_dict["ACH-000001"]
    args.num_genes = example.x.shape[0]  # 4646

    if "disjoint" in args.dataset:
        gene_list = scatter_add(
            torch.ones_like(example.x.squeeze()), example.x_mask.to(torch.int64)
        ).to(torch.int)
        args.max_gene = gene_list.max().item()
        args.cum_num_nodes = torch.cat(
            [gene_list.new_zeros(1), gene_list.cumsum(dim=0)], dim=0
        )
        args.n_pathways = gene_list.size(0)
        print(f"gene distribution: {gene_list}")
        print(f"mean degree:{len(edge_index[0]) / args.num_genes}")
    else:
        print(f"num_genes:{args.num_genes}, num_edges:{len(edge_index[0])}")
        print(f"mean degree:{len(edge_index[0]) / args.num_genes}")

    # ---- [1] Pathway + Transformer ----
    if args.sim is False:
        train_loader, val_loader, test_loader = load_data(
            data, drug_dict, cell_dict, torch.tensor(edge_index, dtype=torch.long), args
        )
        print(
            "total: {}, train: {}, val: {}, test: {}".format(
                len(data),
                len(train_loader.dataset),
                len(val_loader.dataset),
                len(test_loader.dataset),
            )
        )

        model = DRPreter(args).to(args.device)
        # ---- [2] Add similarity information after obtaining embeddings ----
    else:
        train_loader, val_loader, test_loader = load_sim_data(data, args)
        print(
            "total: {}, train: {}, val: {}, test: {}".format(
                len(data),
                len(train_loader.dataset),
                len(val_loader.dataset),
                len(test_loader.dataset),
            )
        )
        drug_nodes_data, cell_nodes_data, drug_edges, cell_edges = load_sim_graph(
            torch.tensor(edge_index, dtype=torch.long), args
        )

        model = Similarity(
            drug_nodes_data, cell_nodes_data, drug_edges, cell_edges, args
        ).to(args.device)

    result_col = "mse\trmse\tmae\tpcc\tscc"

    result_type = "results_sim" if args.sim is True else "results"
    results_path = get_path(args, result_path, result_type=result_type)

    with open(results_path, "w", encoding="utf-8") as f:
        f.write(result_col + "\n")
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    state_dict_name = (
        f"{rpath}weights/weight_sim_seed{args.seed}.pth"
        if args.sim is True
        else f"{rpath}weights/weight_seed{args.seed}.pth"
    )
    stopper = EarlyStopping(
        mode="lower", patience=args.patience, filename=state_dict_name
    )

    for epoch in range(1, args.epochs + 1):
        print(f"===== Epoch {epoch} =====")
        train_loss = train(model, train_loader, criterion, opt, args)

        mse, rmse, mae, pcc, scc, _ = validate(model, val_loader, args)
        results = [epoch, mse, rmse, mae, pcc, scc]
        save_results(results, results_path)

        print(f"Validation mse: {mse}")
        test_MSE, test_RMSE, test_MAE, test_PCC, test_SCC, df = validate(
            model, test_loader, args
        )
        print(f"Test mse: {test_MSE}")
        early_stop = stopper.step(mse, model)
        if early_stop:
            break

    stopper.load_checkpoint(model)
    train_MSE, train_RMSE, train_MAE, train_PCC, train_SCC, _ = validate(
        model, train_loader, args
    )
    val_MSE, val_RMSE, val_MAE, val_PCC, val_SCC, _ = validate(model, val_loader, args)
    test_MSE, test_RMSE, test_MAE, test_PCC, test_SCC, df = validate(
        model, test_loader, args
    )

    print("-------- DRPreter -------")
    print(f"sim: {args.sim}")
    print(f"##### Seed: {args.seed} #####")
    print("\t\tMSE\tRMSE\tMAE\tPCC\tSCC")
    print(
        "Train result: {}\t{}\t{}\t{}\t{}".format(
            r4(train_MSE), r4(train_RMSE), r4(train_MAE), r4(train_PCC), r4(train_SCC)
        )
    )
    print(
        "Val result: {}\t{}\t{}\t{}\t{}".format(
            r4(val_MSE), r4(val_RMSE), r4(val_MAE), r4(val_PCC), r4(val_SCC)
        )
    )
    print(
        "Test result: {}\t{}\t{}\t{}\t{}".format(
            r4(test_MSE), r4(test_RMSE), r4(test_MAE), r4(test_PCC), r4(test_SCC)
        )
    )
    df.to_csv(
        get_path(args, result_path, result_type=result_type + "_df", extension="csv"),
        sep="\t",
        index=0,
    )
    scores = df
    return scores


def run(gParameters):
    args = candle.ArgumentStruct(**gParameters)
    scores = launch(args)

    # Supervisor HPO

    val_scores = {
        "val_loss": scores["val_MSE"],
        "mae": scores["val_MAE"],
        "scc": scores["val_SCC"],
        "pcc": scores["val_PCC"],
        "rmse": scores["val_RMSE"],
    }

    print(f"\nIMPROVE_RESULT val_loss:\t{val_scores}\n")
    with open(Path(args.output_dir) / "scores.json", "w", encoding="utf-8") as f:
        json.dump(val_scores, f, ensure_ascii=False, indent=4)
    return scores


def initialize_parameters(default_model="drpreter_default_model.txt"):
    """Initialize the parameters for the DRPreter Benchmark"""
    print("Initializing Parameters\n")
    drpreter_bmk = DRPreter_candle(
        file_path,
        default_model,
        "pytorch",
        prog="DRPreter",
        desc="CANDLE compliant DRPreter",
    )

    # Initialize the parameters
    gParameters = candle.finalize_parameters(drpreter_bmk)

    return gParameters


def main():
    gParameters = initialize_parameters()
    print(gParameters)
    run(gParameters)
    print("Done.")


if __name__ == "__main__":
    main()
