# Candelized DRPreter (Drug Response PREdictor and interpreTER)
DRPreter: Interpretable Anticancer Drug Response Prediction Using Knowledge-Guided Graph Neural Networks and Transformer

This repository introduces the CANDLE-compliant codebase for the DRPreter Model.

## Installation

To install the necessary packages to run the training we can use `conda`:

```bash
conda create -n drpreter python=3.11 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

conda activate drpreter
```

This will create and activate an environment called `drpreter`.

Next we will need to fill the rest of the dependencies that rely on an active torch installation, we can do this with `pip` and `requirements.txt`

```bash
pip install -r requirements.txt
```

This will install the remaining packages needed into the environment.

There are a few specific packages that requires their own individual install depending on the version of PyTorch and version of CUDA your system has.

```bash
pip install pyg-lib torch-cluster torch-scatter torch-sparse torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html`

pip install  dgl -f https://data.dgl.ai/wheels/$(CUDA)/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
```

Where `$(TORCH)` is your current version of PyTorch and `$(CUDA)` is the version of CUDA. For the requirements file listed, this model runs on PyTorch 2.0 with CUDA 11.8

https://data.pyg.org/whl/torch-2.0.1+cu118.html


## To Run the Model

### Without Singularity

To run the model after creating the conda environment, we can run the following commands:

To preprocess:
```bash
python preprocess.py --train_data_name ccle --val_data_name ccle --test_data_name ccle --train_split_file_name split_0_tr_id --val_split_file_name split_0_vl_id --test_split_file_name split_0_te_id --y_col_name CancID --outdir csa_data/ml_data
```

To train:
```bash
python train.py --train_data_name ccle --val_data_name ccle --test_data_name ccle --train_split_file_name split_0_tr_id --val_split_file_name split_0_vl_id --test_split_file_name split_0_te_id --y_col_name CancID
```

```bash
sh DRPreter/train.sh $(CUDA_VISIBLE_DEVICES) $(CANDLE_DATA_DIR)
```


### With Singularity

1. The first step is to build the singularity container. 
2. Set the `$CANDLE_DATA_DIR` and `$CUDA_VISIBLE_DEVICES` environment variables.
3. Use the different shell scripts for training and evaluation. (`train.sh` & `infer.sh`)

#### Building the Container

To build the container, you will need to run the following command:

```bash
singularity build --fakeroot DRPreter.sif DRPreter.def
```

This requires the `DRPreter.def` file in order to build `DRPreter.sif`

