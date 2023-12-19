"""Script for training the DRPreter Model"""
from pathlib import Path

import numpy as np
import torch
from improve import drug_resp_pred as drp
from improve import framework as frm
from torch_scatter import scatter_add
from typing import Dict
from utils import *

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm
from improve.metrics import compute_metrics

# [Req] Imports from preprocess script
from preprocess import preprocess_params

from Model.DRPreter import DRPreter
from Model.Similarity import Similarity


filepath = Path(__file__).resolve().parent

# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_train_params
# 2. model_train_params
#
# The values for the parameters in both lists should be specified in a
# parameter file that is passed as default_model arg in
# frm.initialize_parameters().

# 1. App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific params for this script.
app_train_params = []

# 2. Model-specific params (Model: LightGBM)
# All params in model_train_params are optional.
# If no params are required by the model, then it should be an empty list.
model_train_params = [
    {
        "name": "cuda_name",
        "action": "store",
        "type": str,
        "help": "Cuda device (e.g.: cuda:0, cuda:1.",
    },
    {
        "name": "learning_rate",
        "type": float,
        "default": 0.0001,
        "help": "Learning rate for the optimizer.",
    },
]

# [Req] Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
train_params = app_train_params + model_train_params
# req_train_params = ["model_outdir", "train_ml_data_dir", "val_ml_data_dir"]
# ---------------------

# [Req] List of metrics names to compute prediction performance scores
metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]


def run(params: Dict):
    # ------------------------------------------------------
    # [Req] Create output dir and build model path
    # ------------------------------------------------------
    # Create output dir for trained model, val set predictions, val set performance scores
    frm.create_outdir(outdir=params["model_outdir"])

    # Build model path
    modelpath = frm.build_model_path(params, model_dir=params["model_outdir"])

    # ------------------------------------------------------
    # [Req] Create data names for train and val sets
    # ------------------------------------------------------
    train_data_fname = frm.build_ml_data_name(params, stage="train")
    val_data_fname = frm.build_ml_data_name(params, stage="val")

    # ------------------------------------------------------
    # Load model input data (ML data)
    # ------------------------------------------------------
    train_loader = stage_dataloader(Path(params["train_ml_data_dir"]) / train_data_fname, params)
    val_loader = stage_dataloader(Path(params["val_ml_data_dir"]) / val_data_fname, params)

    # ------------------------------------------------------
    # Prepare, train, and save model
    # ------------------------------------------------------
    if os.getenv("CUDA_VISIBLE_DEVICES") is not None:
        print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
        # cuda_name = f"cuda:{int(os.getenv('CUDA_VISIBLE_DEVICES'))}"
        cuda_name = "cuda:0"
    else:
        cuda_name = params["device"]

    model = DRPreter(params).to(cuda_name)

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    result_col = "mse\trmse\tpcc\tscc\tr2"

    result_type = "results_sim" if params["sim"] == True else "results"
    results_path = get_path(params, result_type=result_type)

    with open(results_path, "w+") as f:
        f.write(result_col + "\n")
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    state_dict_name = (
        f"{results_path.as_posix()}weights/weight_sim_seed{params['seed']}.pth"
        if params["sim"] == True
        else f"{results_path.as_posix()}weights/weight_seed{params['seed']}.pth"
    )
    stopper = EarlyStopping(mode="lower", patience=params["patience"], filename=state_dict_name)

    for epoch in range(1, params["epochs"] + 1):
        print(f"===== Epoch {epoch} =====")
        train_loss = train(model, train_loader, criterion, opt, params)
        val_true, val_pred = validate(
            model,
            val_loader,
            params,
        )
        val_scores = frm.compute_metrics(val_true, val_pred, metrics_list)
        val_mse = val_scores["mse"]
        # results = [epoch, mse, rmse, mae, pcc, scc, r_squared]
        save_results(val_scores, results_path)

        print(f"Validation mse: {val_mse}")

        early_stop = stopper.step(val_mse, model)
        if early_stop:
            break

    print("EarlyStopping! Finish training!")
    print(f"Best epoch: {epoch - stopper.counter}")

    stopper.load_checkpoint(model)

    # train_MSE, train_RMSE, train_MAE, train_PCC, train_SCC, train_r_squared, _ = validate(
    #    model, train_loader, params
    # )
    # val_MSE, val_RMSE, val_MAE, val_PCC, val_SCC, val_r_squared, _ = validate(
    #    model, val_loader, params
    # )

    val_true, val_pred = validate(
        model,
        val_loader,
        params,
    )

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        params, y_true=val_true, y_pred=val_pred, stage="val", outdir=params["model_outdir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    val_scores = frm.compute_performace_scores(
        params,
        y_true=val_true,
        y_pred=val_pred,
        stage="val",
        outdir=params["model_outdir"],
        metrics=metrics_list,
    )

    return val_scores


def main():
    # [Req]
    additional_definitions = preprocess_params + train_params
    params = frm.initialize_parameters(
        filepath,
        default_model="drpreter_default_model.txt",
        additional_definitions=additional_definitions,
        required=None,
    )

    val_scores = run(params)
    print("\nFinished training DRPreter Model")


if __name__ == "__main__":
    main()
