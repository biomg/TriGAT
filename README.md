# TriGAT

TriGAT (**Tri**-layer **G**raph **A**ttention network) predicts the binding specificity between T-cell receptors and antigen epitopes. Built on the GTE graph-learning backbone, it introduces three changes:

- **Difference-enhanced edge features.** The absolute difference and the element-wise product of the two endpoint node embeddings are concatenated with the original vectors, expanding the edge representation from `2d` to `4d`, so that both the similarity and the discrepancy between the two endpoints are captured.
- **Three-layer residual GAT.** A third graph attention layer is stacked on the two-layer network and the outputs of the three layers are summed element-wise, which alleviates gradient attenuation and fuses multi-hop neighbourhood information.
- **Multi-trajectory fusion training.** A candidate pool is built from multiple random seeds, an epoch snapshot bank, SWA weight averaging and EMA shadow weights, and all candidates are evaluated under a unified protocol. This substantially reduces the variance caused by the randomness of a single training run.

On the four public benchmarks, TriGAT reaches AUC of **0.9529 / 0.9181 / 0.9274 / 0.8929** and AUPR of **0.7612 / 0.6430 / 0.6753 / 0.5326** on pMTnet / TEINet / VDJdb / McPAS.


## Quick Start

1. **Create a Conda Environment**:

   Start by creating a Conda environment with Python 3.11. If you haven't already installed Conda, you can get it from [Anaconda](https://www.anaconda.com/products/individual).

   ```
   conda create -n TriGAT python=3.11
   ```

   Activate the environment:

   ```
   conda activate TriGAT
   ```

2. **Install Dependencies**:

   Use pip to install the required packages listed in the requirements.txt file.

   ```
   pip install -r requirements.txt
   ```

   The code is tested with PyTorch, PyTorch Geometric and `libauc==1.3.0`; `matplotlib` is needed for the training curves and for `figs_paper_v2.py`.

3. **Prepare the data**:

   Place each dataset under `processed_data/<dataset>/<split>/` and its embedding file under `models/<dataset>/<dataset>_embeddings.pkl`. Pre-trained embeddings are archived on Zenodo — see [Data and Materials](#data-and-materials).

## How to Train

```bash
python train.py --configs_path configs/pMTnet.yml --gpu 0 --droup_out 0.1
```

Which dataset and which split are used is determined by the YAML file, not by a command-line flag: `path` points to `processed_data/<dataset>/StrictTCR/`. To switch dataset, point `--configs_path` at another config; to switch split, edit `path` in that config.

Available options:

- `--configs_path`:
  - Default: `configs/pMTnet.yml`
  - Choices: `configs/pMTnet.yml`, `configs/TEINet.yml`, `configs/VDJdb.yml`, `configs/McPAS.yml`

- `--gpu`:
  - Default: 0
  - ID of the GPU to use.

- `--lr` / `--epochs`:
  - Default: 0.001 / 300

- `--droup_out`:
  - Default: 0.1
  - Dropout rate (spelled as in the code).

- `--opt_strategy`:
  - Default: `triple`
  - Choices: `adam`, `pesg`, `sgd`, `dual`, `triple`
  - `triple` combines Adam, PESG and SGD.

- `--w_celoss`, `--w_aucloss`, `--w_focalloss`:
  - Weights of the cross-entropy, AUC-margin and focal loss terms.

### Extended switches (`--xx`)

Multi-trajectory fusion and the other training-level options are controlled by a separate `--xx key=value` namespace, which is stripped from `argv` before the regular arguments are parsed, so it never conflicts with the options above.

```bash
python train.py --configs_path configs/pMTnet.yml --xx seeds=5,snapk=5,ema=1,sched=cos
python train.py --configs_path configs/pMTnet.yml --xhelp    # print the current values
```

| Key | Default | Meaning |
|---|---|---|
| `seeds` | 5 | number of random seeds; `N>1` trains N trajectories and averages their predictions |
| `pool` | 1 | enable the end-of-run candidate pool; `0` falls back to the single-model protocol |
| `snapk` | 5 | keep the `N` best-scoring epoch snapshots |
| `swalast` | 0 | additionally average the weights of the last `N` epochs |
| `ema` / `emad` | 1 / 0.995 | evaluate the EMA shadow weights / EMA decay |
| `sched` / `cycles` / `lrmin` | none / 3 / 0.05 | `none`, `cos`, `cos_restart` or `step` schedule |
| `tailk` | 0 | average the predicted probabilities of the last `N` epochs |
| `aucm` | off | handling of the AUC-M loss term: `off`, `shared` or `legacy` |
| `pw` | batch | `pos_weight` normalisation: `batch` or `legacy` |
| `pairn` | 8000 | number of samples in the pairwise AUC loss |
| `wf` / `wp` | 0.5 / 0.2 | weights of the focal and pairwise terms |
| `restore` | McPAS=1 | restore the initial weights after the focal-loss search |
| `clip` | 0 | gradient-clipping threshold (`0` disables it) |

The defaults above already correspond to the full TriGAT model, so a plain `python train.py` reproduces the paper setting.



## Data and Materials

Due to file size limitations, this repository does not include the following model weights/datasets. These files have been archived on Zenodo and can be accessed via the links below:

- **Model file**: `pMTnet_embeddings.pkl`   `TEINet_embeddings.pkl`   `VDJdb_embeddings.pkl` 
- **Zenodo record**: zhao& zhou. (2026). ResGAT [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.21731301
