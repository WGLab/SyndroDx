# Learning craniofacial representations for multimodal rare disease diagnosis

Training code for the selected craniofacial and phenotype models. The visual encoder learns patient-level facial representations, adapts to disease diagnosis, and supplies frozen features to a phenotype model with gated cross-attention.

## Training stages

| Stage | Script | Training objective |
|---|---|---|
| 1. Craniofacial adaptation | `training/identity_cpt/train.py` | Patient-identity ArcFace training of TransFace-L on clinical facial images |
| 2. Disease adaptation | `training/disease_finetune/train.py` | Full-encoder ArcFace fine-tuning on the GMDB training split |
| 3. Phenotype pretraining | `training/phenotype/pretrain_hpo.py` | Public HPO cases and disease-profile supervision |
| 4. HPO diagnosis | `training/phenotype/train.py` | GMDB adaptation of the independent HPO branch |
| 5. Facial + HPO diagnosis | `training/fusion/train.py` | Gated cross-attention over frozen facial features and HPO tokens |

The selected visual route uses an ArcFace scale of 64 and margin of 0.5. Identity adaptation uses a backbone learning rate of `5e-7`, a head learning rate of `1e-3`, and two augmented views per patient draw. The selected encoder is the step-1,000 checkpoint. Disease adaptation uses a backbone/projection learning rate of `2e-6`, a head learning rate of `1e-3`, batch size 64, and 50 epochs, with two initial head-only epochs.

The HPO model projects frozen BiomedBERT term embeddings into a 256-dimensional, two-layer transformer. In the fusion model, a projected 512-dimensional facial embedding queries the HPO tokens through eight-head cross-attention. A learned gate combines this evidence with the pooled HPO representation. Both the visual encoder and the BiomedBERT term embeddings remain frozen during fusion training. The HPO-only branch is trained separately.

HPO adaptation and fusion use AdamW, learning rate `1e-4`, weight decay `0.01`, and batch size 128. Checkpoints are selected by validation macro Top-1, with macro Top-5 as the tie-breaker. Training runs for at least 15 and at most 50 epochs, with patience 10. The selected fusion condition is `no_chop`: it uses the CHOP-adapted visual encoder but does not add weakly paired CHOP HPO records to fusion training.

## Environment

Python 3.11 and a CUDA-enabled PyTorch installation are required. Install the packages in `requirements.txt` in an isolated environment compatible with your CUDA driver.

Obtain the upstream [TransFace](https://github.com/DanJun6737/TransFace) source and pretrained TransFace-L weights, and the [GestaltMatcher](https://github.com/igsb/GestaltMatcher) source used for image augmentation. Their source and weights remain subject to their upstream licenses and access terms.

Set these paths locally:

```bash
export CRL_WORKSPACE=/path/to/prepared-private-workspace
export CRL_DEVELOPMENT_INPUTS=/path/to/frozen-development-inputs
export TRANSFACE_WEIGHTS=/path/to/glint360k_model_TransFace_L.pt
export TRANSFACE_VIT_FILE=/path/to/transface_vit_original.py
export ARCFACE_LOSS_FILE=/path/to/author_margin_losses.py
export GESTALTMATCHER_SOURCE=/path/to/gestaltmatcher-source
```

`CRL_DEVELOPMENT_INPUTS` contains the frozen `validation/` directory. `GESTALTMATCHER_SOURCE` contains `lib/datasets/gestalt_matcher_dataset.py`. For the upstream snapshots used by this training route, the SHA-256 checksums are:

- TransFace ViT source: `6f14a397b545a644eb8824632245b485972ebceed5fe7f841c1d54ff4465195d`
- ArcFace loss source: `8b74b94cd250d96f2ac08b24bcc711c7bc8f926842dde96323ab430a096806e1`

## Prepared inputs

These scripts consume the prepared, checksum-verified workspace used for training. They do not prepare raw hospital exports. The original workspace directory names are retained to preserve checkpoint and receipt compatibility:

| Workspace directory | Required local inputs |
|---|---|
| `transface_arcface_identity_cpt_20261026` | `protocol.json`, patient class map, released crop manifest, cache index/receipt, packed RGB cache |
| `transface_objective_comparison_20261028` | `protocol.json`, development cohort index under `private/` |
| `gmdb_supervised_transface_20261026` | `private/images.uint8.npy`, used for frozen development feature extraction |
| `arcface_transfer_evaluation_20261028` | `selection.json` with the selected encoder path, step and SHA-256 |
| `matched_craniofacial_benchmark_20261031` | `protocol.json`, `cases.json`, initialization/data pins, passing split audit, image array and patient records under `private/` |
| `hpo_fusion_staged_20261031` | `protocol.json`, `cases.json`, data/term-embedding receipts, `features/term_embeddings.npy`, and `private/{gmdb,public_cases,profiles}.npz` |
| `chop_hpo_increment_20261101` | Corresponding fusion protocol, cases, receipts, frozen features and HPO arrays |

GMDB and public-case arrays contain integer HPO token IDs, assertion states, class labels, and disjoint `train` / `validation` indices. GMDB arrays also contain the prepared facial feature vectors. Profile arrays contain disease HPO token IDs, assertion states and availability masks. Token 0 is padding. Assertion states 0 and 1 denote present and absent findings; fusion additionally supports state 2 for unresolved assertion. Missing findings are never converted into confirmed absences.

Patient identifiers, images, arrays, runtime manifests and checkpoints stay outside this repository. The workspace and all checkpoint inputs must come from a trusted local source. Keep patients and duplicate images separated across partitions, and exclude the held-out CHOP expert and BCH patients from every training stage. The supplied split checks complement the upstream cohort audit; they do not discover cross-institution identity matches automatically.

## Run

Run the following from the repository root after preparing the local workspace. Seed 11 identifies the selected release; additional seeds require corresponding entries in the local `cases.json` files.

```bash
# 1. Clinical patient-identity adaptation. The GPU count must divide the global patient draws.
torchrun --standalone --nproc_per_node=4 training/identity_cpt/train.py arcface_lr5e7

# 2. Disease adaptation from the selected step-1,000 encoder.
python training/disease_finetune/train.py transface_chop_cpt__arcface_full__lr2e-06__seed11

# 3. Public phenotype pretraining.
python training/phenotype/pretrain_hpo.py public_pretrained_knowledge__seed11

# 4. Independent HPO diagnosis branch.
python training/phenotype/train.py public_pretrained_knowledge__hpo__seed11

# 5. Facial + HPO fusion using frozen features from stage 2.
python training/fusion/train.py no_chop__seed11
```

Stage 1 writes recovery checkpoints and distributed development features. Its protocol defines the training duration and extraction steps; the original run evaluated through step 4,000 and selected step 1,000. The selection receipt must point to that encoder before stage 2. Stages 4 and 5 both initialize from stage 3. Stage 5 requires facial features extracted with the selected stage-2 encoder and matched to the HPO rows; feature preparation is not part of these training scripts.

Resume uses the saved optimizer, scheduler and random states where applicable. Use a separate private workspace for a new run. Training outputs must not be placed inside this repository.

## Scope

This repository contains the final training route and its required model, sampling and validation helpers. Historical experiments, alternate backbones, cluster resource management and the separate BCH inference package are excluded. The code has been extracted from the selected experiments with configurable workspace paths and final-route guards; packaging checks do not constitute a new full training reproduction.
