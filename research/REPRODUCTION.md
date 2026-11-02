# Reproduction requirements

## Source archive versus runnable inference

Use `../inference/` and the separately distributed checkpoint package for external evaluation. Its code has been exercised from a relocated directory. This research tree preserves the actual development programs, with deployment paths normalized. Numeric training parameters are supplied in TRAINING_PARAMETERS.json; historical runtime configurations and derived metadata remain local. It is not a single end-to-end command that can recreate a private cohort from a public checkout.

The two sibling source roots are `disease-diagnosis-model/` and `SyndroDx-python-slurm/`. Historical imports often expect this relative organization. An absolute project prefix appears as `/opt/craniofacial`, restricted inputs as `/controlled-data`, the scheduler as `/opt/site/slurm`, and an account placeholder as `research_user`. Replace these deployment settings in a separate working copy with your institution's paths. Configure Python environments, scheduler partitions, nodes, accounts, mail recipients and allocation IDs before launching any resource controller. Historical job IDs are not active allocations for a new run.

Changing paths changes file hashes. Regenerate the relevant preparation receipts and lock a new protocol for a new run; do not disable provenance assertions or reuse a receipt for different inputs. Configuration-generating code is included, but the historical runtime JSON/YAML files are not. `SOURCE_MANIFEST.json` allows the exported source snapshot to be verified without loading any patient data.

## Local inputs

| Stage | Required inputs and checks |
|---|---|
| Annotation/cropping | Authorized local image inventory; local Qwen and face-detector weights; patient grouping; review/admission outputs |
| Patient-identity adaptation | Separate train/validation/test TSV manifests with `group_id`, `source`, `split`, `training_admitted`, `view`, `crop_path`, `crop_sha256`, and pixel hashes where available; training-only identity class map and matching cache receipts |
| GMDB training | Licensed GMDB inputs and its split version; normalized training-only disease vocabulary; image and patient overlap exclusions; model initialization pins |
| HPO/fusion training | Local aligned arrays of visual `features`, HPO `ids`, assertion `states`, `labels`, and train/validation indices; frozen term embeddings and vocabulary; preparation receipts |
| Expert evaluation | Patient grouping, reference diagnoses, candidate crosswalk, image/view aggregation rules and eligible HPO assertions; separate model-selection and test cohorts |
| Agent replay | Audited local case evidence with availability times, a locked diagnostic model and train/development policy splits; hidden replay answers must not enter question selection |
| Variant review | Local observed-variant records with resolvable identity and source evidence; public annotations alone do not establish that a patient carries a variant |

Actual patient manifests, group maps, train/test membership, clinical observations, image hashes tied to patients, cached features and evaluation outputs are not included. Each preparation script specifies its input schema and verification checks. New patient cohorts need new manifests and receipts. Mapping tables that join real and pseudonymous identifiers must remain local.

HPO state 0 denotes present and state 1 denotes explicitly absent in these models; state 2 appears in historical work as unresolved. The frozen inference release excludes unresolved observations. Review the source's eligibility logic before adopting a historical experiment as a clinical protocol.

## Dependencies

The verified inference dependencies are pinned in `../inference/requirements.txt`. Research dependencies vary by route. `IMPORT_INVENTORY.json` records imports across the archive, including local modules; it is not a pip requirements file. `ENVIRONMENT_REFERENCE.json` records observed versions from the development environment and is not a tested universal installation lock.

Third-party implementations and pretrained weights should be obtained through their upstream projects, preserving their terms:

- [TransFace](https://github.com/DanJun6737/TransFace): original ViT and ArcFace implementations expected by the research wrappers.
- [DINOv3](https://github.com/facebookresearch/dinov3): official SSL code and separately accessible pretrained weights.
- [InsightFace](https://github.com/deepinsight/insightface): facial model/detector dependencies used in preparation and baseline work.
- [GestaltMatcher](https://github.com/igsb/GestaltMatcher): baseline implementation dependencies; use the specific source/checkpoint receipts for a matched reproduction.
- BiomedBERT and Qwen model identifiers and tokenizer settings are recorded in the corresponding preparation/model source and recipe files.

Some development dependencies were installed into experiment-local `dependencies/` or `vendor/` directories rather than the shared environment. These installed trees are not copied into this repository. `THIRD_PARTY_SNAPSHOTS.json` records available upstream revision and file-hash information without distributing their code or weights. Match those implementations before interpreting numerical reproduction differences.

## Validation

Run the source-only integrity check from the repository root:

```bash
python tools/verify_research_source.py
```

This checks source hashes and Python syntax without executing training, loading patient data or launching scheduler jobs. Selected evidence-engine unit tests can run on synthetic fixtures; the publication validation receipt documents the actual tests executed. A syntax/integrity pass is not a claim that every historical training experiment has been rerun in a clean environment.
