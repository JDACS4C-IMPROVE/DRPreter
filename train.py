# Placeholder train python script
from pathlib import Path
import argparse
from utils import *
import improve_utils
from improve_utils import improve_globals as ig

from torch_scatter import scatter_add

fdir = Path(__file__).resolve().parent


def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate (default: 0.0001)")
    parser.add_argument("--layer", type=int, default=3, help="Number of cell layers")
    parser.add_argument("--hidden_dim", type=int, default=8, help="hidden dim for cell")
    parser.add_argument("--layer_drug", type=int, default=3, help="Number of drug layers")
    parser.add_argument(
        "--dim_drug", type=int, default=128, help="hidden dim for drug (default: 128)"
    )
    parser.add_argument(
        "--dim_drug_cell", type=int, default=256, help="hidden dim for drug and cell (default: 256)"
    )
    parser.add_argument(
        "--dropout_ratio",
        type=float,
        default=0.1,
        help="Dropout ratio (default: 0.1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Maximum number of epochs (default: 300)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="patience for early stopping (default: 10)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="train, test",
    )
    parser.add_argument(
        "--edge",
        type=str,
        default="STRING",
        help="STRING, BIOGRID",
    )  # BIOGRID: removed
    parser.add_argument(
        "--string_edge",
        type=float,
        default=990,
        help="Threshold for edges of cell line graph",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="disjoint",
        help="joint, disjoint, COSMIC",
    )
    parser.add_argument(
        "--trans",
        type=bool,
        default=True,
        help="Use Transformer or not",
    )
    parser.add_argument(
        "--sim",
        type=bool,
        default=False,
        help="Construct homogeneous similarity networks or not",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        required=False,
    )
    parser.add_argument(
        "--y_col_name",
        type=str,
        default="auc",
        required=False,
    )
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
        help="The path to the file that contains the split ids ('split_0_tr_id', 'split_0_vl_id').",
    )
    parser.add_argument(
        "--val_split_file_name",
        type=str,
        nargs="+",
        required=True,
        help="The path to the file that contains the split ids ('split_0_tr_id', 'split_0_vl_id').",
    )
    parser.add_argument(
        "--test_split_file_name",
        type=str,
        nargs="+",
        required=True,
        help="The path to the file that contains the split ids ('split_0_tr_id', 'split_0_vl_id').",
    )

    return parser.parse_args()


def main():
    args = arg_parse()
    result_path = fdir / "Result/"

    split = 0
    source_data_name = "CCLE"

    print(f"seed: {args.seed}")
    set_random_seed(args.seed)

    drug_dict = np.load(ig.ml_data_dir / "drug_feature_graph.npy", allow_pickle=True).item()

    cell_dict = np.load(
        ig.ml_data_dir / f"cell_feature_std_{args.dataset}.npy", allow_pickle=True
    ).item()

    edge_type = "PPI_" + str(args.string_edge) if args.edge == "STRING" else args.edge
    edge_index = np.load(ig.ml_data_dir / f"edge_index_{edge_type}_{args.dataset}.npy")

    example = cell_dict["CCLE.22RV1"]
    args.num_feature = example.x.shape[1]  # 1
    args.num_genes = example.x.shape[0]  # 4646
    # print(f'num_feature: {args.num_feature}, num_genes: {args.num_genes}')

    if "disjoint" in args.dataset:
        gene_list = scatter_add(
            torch.ones_like(example.x.squeeze()), example.x_mask.to(torch.int64)
        ).to(torch.int)
        args.max_gene = gene_list.max().item()
        args.cum_num_nodes = torch.cat([gene_list.new_zeros(1), gene_list.cumsum(dim=0)], dim=0)
        args.n_pathways = gene_list.size(0)
        print("num_genes:{}, num_edges:{}".format(args.num_genes, len(edge_index[0])))
        print("gene distribution: {}".format(gene_list))
        print("mean degree:{}".format(len(edge_index[0]) / args.num_genes))
    else:
        print("num_genes:{}, num_edges:{}".format(args.num_genes, len(edge_index[0])))
        print("mean degree:{}".format(len(edge_index[0]) / args.num_genes))

    print("\nLoad response train data ...")
    rs_tr = improve_utils.load_single_drug_response_data_v2(
        source=args.train_data_name,
        split_file_name=args.train_split_file_name,
        y_col_name=args.y_col_name,
        sep=",",
        verbose=True,
    )

    print("\nLoad response val data ...")
    rs_vl = improve_utils.load_single_drug_response_data_v2(
        source=args.val_data_name,
        split_file_name=args.val_split_file_name,
        y_col_name=args.y_col_name,
        sep=",",
        verbose=True,
    )

    print("\nLoad response test data ...")
    rs_te = improve_utils.load_single_drug_response_data_v2(
        source=args.test_data_name,
        split_file_name=args.test_split_file_name,
        y_col_name=args.y_col_name,
        sep=",",
        verbose=True,
    )

    # Load Train, Test, Val Response Data
    ge = pd.read_csv(ig.gene_expression_file_path, sep=",", index_col=0)
    ge = ge.reset_index()

    # Retain (canc, drug) response samples for which we have the omic data
    rs_tr, ge_tr = improve_utils.get_common_samples(df1=rs_tr, df2=ge, ref_col=ig.canc_col_name)
    rs_vl, ge_vl = improve_utils.get_common_samples(df1=rs_vl, df2=ge, ref_col=ig.canc_col_name)
    rs_te, ge_te = improve_utils.get_common_samples(df1=rs_te, df2=ge, ref_col=ig.canc_col_name)
    print(rs_tr[[ig.canc_col_name, ig.drug_col_name]].nunique())
    print(rs_vl[[ig.canc_col_name, ig.drug_col_name]].nunique())
    print(rs_te[[ig.canc_col_name, ig.drug_col_name]].nunique())

    # ---- [1] Pathway + Transformer ----
    if args.sim == False:
        train_loader, val_loader, test_loader = load_data(
            rs_tr,
            rs_te,
            rs_vl,
            drug_dict,
            cell_dict,
            torch.tensor(edge_index, dtype=torch.long),
            args,
        )
        #       print('total: {}, train: {}, val: {}, test: {}'.format(len(data), len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)))

        model = DRPreter(args).to(args.device)
        # print(model)

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

        model = Similarity(drug_nodes_data, cell_nodes_data, drug_edges, cell_edges, args).to(
            args.device
        )
        # print(model)

    # -----------------------------------------------------------------

    result_col = "mse\trmse\tmae\tpcc\tscc\tr_squared"

    result_type = "results_sim" if args.sim == True else "results"
    results_path = get_path(args, result_path, result_type=result_type)

    with open(results_path, "w+") as f:
        f.write(result_col + "\n")
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    state_dict_name = (
        f"{rpath}weights/weight_sim_seed{args.seed}.pth"
        if args.sim == True
        else f"{rpath}weights/weight_seed{args.seed}.pth"
    )
    stopper = EarlyStopping(mode="lower", patience=args.patience, filename=state_dict_name)

    for epoch in range(1, args.epochs + 1):
        print(f"===== Epoch {epoch} =====")
        train_loss = train(model, train_loader, criterion, opt, args)

        mse, rmse, mae, pcc, scc, r_squared, _ = validate(model, val_loader, args)
        results = [epoch, mse, rmse, mae, pcc, scc, r_squared]
        save_results(results, results_path)

        print(f"Validation mse: {mse}")
        test_MSE, test_RMSE, test_MAE, test_PCC, test_SCC, test_r_squared, df = validate(
            model, test_loader, args
        )
        print(f"Test mse: {test_MSE}")
        early_stop = stopper.step(mse, model)
        if early_stop:
            break

    print("EarlyStopping! Finish training!")
    print("Best epoch: {}".format(epoch - stopper.counter))

    stopper.load_checkpoint(model)

    train_MSE, train_RMSE, train_MAE, train_PCC, train_SCC, train_r_squared, _ = validate(
        model, train_loader, args
    )
    val_MSE, val_RMSE, val_MAE, val_PCC, val_SCC, val_r_squared, _ = validate(
        model, val_loader, args
    )
    test_MSE, test_RMSE, test_MAE, test_PCC, test_SCC, test_r_squared, df = validate(
        model, test_loader, args
    )

    print("-------- DRPreter -------")
    print(f"sim: {args.sim}")
    print(f"##### Seed: {args.seed} #####")
    print("\t\tMSE\tRMSE\tMAE\tPCC\tSCC\tr_squared")
    print(
        "Train result: {}\t{}\t{}\t{}\t{}\t{}".format(
            r4(train_MSE),
            r4(train_RMSE),
            r4(train_MAE),
            r4(train_PCC),
            r4(train_SCC),
            r4(train_r_squared),
        )
    )
    print(
        "Val result: {}\t{}\t{}\t{}\t{}\t{}".format(
            r4(val_MSE), r4(val_RMSE), r4(val_MAE), r4(val_PCC), r4(val_SCC), r4(val_r_squared)
        )
    )
    print(
        "Test result: {}\t{}\t{}\t{}\t{}\t{}".format(
            r4(test_MSE),
            r4(test_RMSE),
            r4(test_MAE),
            r4(test_PCC),
            r4(test_SCC),
            r4(test_r_squared),
        )
    )
    df.to_csv(
        get_path(args, result_path, result_type=result_type + "_df", extension="csv"),
        sep="\t",
        index=0,
    )


if __name__ == "__main__":
    main()
