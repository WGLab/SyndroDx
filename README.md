# Learning craniofacial representations from clinical images for multimodal rare disease diagnosis

Research code for a multimodal model that combines clinically adapted craniofacial representations with Human Phenotype Ontology (HPO) observations for rare disease diagnosis.

The visual encoder follows TransFace-L → CHOP patient-identity ArcFace adaptation → GMDB disease-supervised ArcFace fine-tuning. A phenotype encoder uses BiomedBERT term embeddings and a set transformer. Facial and HPO features interact through cross-attention and gated fusion for disease classification. Case retrieval uses independent visual and HPO similarities.

## Current release

This initial code release provides the frozen evaluation implementation: image preprocessing, patient-level aggregation, three modality comparisons, and optional local-gallery retrieval. Training orchestration and the experimental agent and variant modules are not included in this release.

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
