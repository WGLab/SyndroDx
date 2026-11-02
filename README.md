# Learning craniofacial representations from clinical images for multimodal rare disease diagnosis

Research code for a multimodal model that combines clinically adapted craniofacial representations with Human Phenotype Ontology (HPO) observations for rare disease diagnosis.

The visual encoder follows TransFace-L → CHOP patient-identity ArcFace adaptation → GMDB disease-supervised ArcFace fine-tuning. A phenotype encoder uses BiomedBERT term embeddings and a set transformer. Facial and HPO features interact through cross-attention and gated fusion for disease classification. Case retrieval uses independent visual and HPO similarities.

## Current release

The repository includes the frozen evaluation implementation and the research source used to develop it: data inventory, facial annotation and cropping, DINO/SSL experiments, TransFace/ArcFace adaptation, disease fine-tuning, HPO and multimodal training, matched evaluation, multi-seed controls, and experimental evidence-acquisition and variant modules.

- **[Research workflow and training entry points](research/README.md)** — stage-by-stage guide and the selected checkpoint lineage.
- **[Full source index](research/SOURCE_INDEX.md)** — historical experiments, tests and scheduler scripts.
- **[Training parameters](research/TRAINING_PARAMETERS.json)** — numeric hyperparameters extracted through an explicit field allowlist.
- **[Reproduction requirements](research/REPRODUCTION.md)** — local data contracts, dependencies and deployment configuration.
- **[Frozen inference](inference/README.md)** — the independently runnable evaluation package.

The research tree preserves dated experimental directories so that final methods can be distinguished from alternatives and diagnostics. It is a source archive with site-neutral path placeholders; running historical training requires locally authorized data, regenerated configuration/provenance files and a configured compute environment. Historical runtime JSON/YAML and derived metadata remain local. The experimental agent and variant code is separate from the frozen inference checkpoint.

The default is seed 11, with 216 diagnostic candidates. Inputs without eligible HPO use the visual branch; inputs without facial images use the HPO branch. If neither modality is usable, the model abstains. Predictions are closed-set model scores, not calibrated clinical probabilities.

## Run

Model weights are distributed separately through the project collaboration channel. Obtain `bch_evaluation_v1_20261102.tar.gz`, extract it, and copy its four `weights/*.pt` files into `inference/weights/`. Keep the filenames unchanged; the runner verifies all packaged files against their checksums.

From the `inference` directory, install dependencies once using Python 3.11:

```bash
python3.11 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
```

Then run:

```bash
bash run.sh --input /path/to/cases.jsonl --output /path/to/results_v1 --device cuda
```

One JSONL row represents one patient, with a study pseudonym, local facial crop paths, and explicitly present or absent HPO terms. The default command produces facial-only, HPO-only, and facial + HPO rankings on the same inputs. Use `--device cpu` for CPU inference.

See the [evaluation README](inference/README.md) for the complete input schema, preprocessing, optional retrieval, output definitions and evaluation protocol. The supplied example is synthetic. Patient data and model weights are not stored in this repository. Inference runs locally without an API or model download.

## Validation and scope

The exported inference implementation was tested against the locked source implementation on synthetic inputs. Visual features, diagnosis scores and HPO retrieval features matched exactly under the tested CPU/FP32 configuration. A relocated-package test covered multiple images, missing modalities, retrieval and output protection. See [verification](inference/VERIFICATION.json).

These checks establish software equivalence. External clinical evaluation is ongoing. Report disease vocabulary coverage and compare modalities on the same eligible patients. Keep patient views and visits together, audit cross-source overlap, and use only phenotypes available at the intended diagnostic time. No external test outcomes should be used to select this checkpoint.

## Acknowledgments

The visual architecture is based on [TransFace](https://github.com/DanJun6737/TransFace). The phenotype branch uses BiomedBERT-derived term embeddings and HPO concepts. Preserve the applicable terms for upstream models, ontologies and datasets. This repository does not confer additional rights to those resources.
