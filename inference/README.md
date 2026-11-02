# Craniofacial + HPO model: BCH evaluation release

**Version:** `bch_evaluation_v1_20261102`  
**Purpose:** frozen external research evaluation at BCH. No training is required.

This package accepts locally stored facial crops and patient-level HPO observations. One run produces **facial-only, HPO-only, and facial + HPO** disease rankings on the same patients. All inference is local; no API, account, Hugging Face token, or internet connection is needed after dependency installation.

## 1. Quick start

Extract the archive and enter its directory. Use Python **3.11** on Linux. Create the environment once:

```bash
python3.11 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
```

For a CUDA machine, use a PyTorch build compatible with the site's NVIDIA driver. The pinned version is PyTorch 2.8.0. An existing environment can be selected with `PYTHON_BIN=/path/to/python` before the command. Installation can use your institution's package mirror; inference does not download anything.

**Run the BCH cohort with one command:**

```bash
bash run.sh --input /path/to/bch_cases.jsonl --output /path/to/bch_results_v1 --device cuda
```

The output directory must be new. Default image batch size is 8, and the default mode is `all` (three modality comparisons). A single GPU is sufficient for inference; distributed training is not needed. If memory is limited, add `--batch-size 1`. CPU inference is also supported with `--device cpu`.

**First-run smoke test, without patient data:**

```bash
bash run.sh --input examples/cases.jsonl --output /path/to/synthetic_smoke_v1 --device cpu
```

This example contains one invented HPO-only input. It checks software execution, not clinical performance or image preprocessing. The export's separate verification used synthetic images/tensors to check the visual path too; see `VERIFICATION.json`.

## 2. Input format

Create a UTF-8 JSONL file with **one patient per line**. Use a study pseudonym, not an MRN or a name. Example with image paths:

```json
{"case_id":"BCH_STUDY_001","images":["crops/001_front.jpg","crops/001_profile.jpg"],"hpo_present":["HP:0000316"],"hpo_absent":["HP:0001250"],"hpo_unknown":[]}
```

This is an invented schema example, not a patient record. Image paths may be absolute or relative to the **JSONL file's directory**, regardless of the working directory. List all included photographs for a patient in the same row. Keep the patient key consistent across any query and gallery manifests.

| Field | Meaning |
|---|---|
| `case_id` | Unique study pseudonym: letters, digits, `_`, `-`, or `.`; at most 100 characters |
| `images` | List of local, reviewed facial crop paths; use `[]` when unavailable |
| `hpo_present` | HPO IDs explicitly documented as present for this patient |
| `hpo_absent` | HPO IDs explicitly assessed and documented as absent |
| `hpo_unknown` | Optional unresolved observations; ignored by the model |

Do not infer an absent finding from an omitted term. Do not add disease-profile HPO terms based on a known diagnosis. If the meaning of an exported HPO record is unresolved, put it in `hpo_unknown` or omit it until reviewed. Duplicate terms are deduplicated; contradictory states are rejected. Unknown ontology IDs are excluded and listed in the local output. More than 128 eligible HPO terms causes an error instead of silent truncation.

Only these input fields are accepted. Keep reference diagnoses in a separate evaluation table, away from the inference inputs.

### Image preparation

Supply **reviewed face crops**, including frontal, oblique, or profile views where available. This runner does not detect faces, choose a face from a group photograph, crop raw photographs, or assess image quality. Do not send MRI, screenshots, full-body photographs, or non-face images through this interface. Preserve the source facial anatomy and background within the crop; no background removal or synthetic frontalization is applied.

The runner reproduces the locked image transform: OpenCV decode, bilinear resize to 112 × 112, BGR → RGB, then `pixel / 127.5 - 1`. No test-time flipping is used. Each image yields a normalized 512-dimensional vector. Multiple images are averaged per patient and normalized again. This is **all-image pooling**, not the separate view-balanced pooling used in some earlier internal experiments. Keep this choice fixed for BCH evaluation and document crop preparation.

## 3. Model and checkpoint

| Component | Frozen version |
|---|---|
| Visual encoder | TransFace-L → CHOP patient-identity ArcFace adaptation → GMDB disease ArcFace fine-tuning; seed 11, epoch 50 |
| Visual representation | Native 512-dimensional patient embedding |
| HPO encoder | Precomputed BiomedBERT term embeddings + trained set transformer; seed 11 |
| Multimodal diagnosis | Facial query cross-attention over HPO tokens, followed by learned gated fusion and a 216-class disease head |
| Missing HPO | Visual classifier fallback |
| Missing face | Independent HPO classifier fallback |
| Neither modality usable | Abstention: empty ranking, `insufficient_input` |
| Optional case retrieval | Equal-weight independent visual/HPO cosine scores when both patients have eligible HPO; otherwise visual cosine |

The classification head is **cross-attention + gating**. The optional retrieval rule is **score-level fusion**. They are separate outputs and must not be described as the same method. Facial-only scores use the original ArcFace disease weights restricted to the same 216 candidates and the locked temperature of 4.0. Inference is FP32.

This is a tensor-only export of the locked development release `craniofacial_multimodal_v1_20261101`, not a new checkpoint selected on BCH. Original source checkpoint hashes and settings are in `config.json`; portable file hashes are in `checksums.json`. The single seed 11 checkpoint is shipped. Previously reported four-seed means are not the expected output of an ensemble in this package.

Weights:

```text
weights/visual.pt       Facial encoder, without optimizer or training state
weights/visual_head.pt  Visual disease weights for the 216 shared candidates
weights/hpo.pt          Independent HPO model
weights/fusion.pt       Cross-attention diagnosis model
assets/diseases.json    Ordered disease IDs, names and available exact MONDO mappings
assets/term_index.json  Ordered supported HPO IDs
```

The HPO checkpoints already include the term embeddings. A separate BiomedBERT download is unnecessary. The package includes no patient photographs, patient records, patient embeddings, patient-ID mappings, or reference case gallery. It does not include the experimental agent or variant branch.

## 4. Outputs

The output directory contains:

| File | Content |
|---|---|
| `predictions.jsonl` | One patient per line; actual modality route, Top-10 disease ranking, and all 216 scores for each requested mode |
| `features.npz` | Patient keys, native visual embeddings (512), and independent HPO embeddings (256); unavailable modalities are zero vectors |
| `summary.json` | Aggregate patient counts, fallback/abstention counts, release, precision and runtime versions |

Each ranked disease includes `gmdb_id`, `name`, `mondo_id` when a unique exact mapping exists, and `model_score`. Full score arrays follow the order of `assets/diseases.json`. `model_score` is a normalized model output **within the 216 candidates**, not a calibrated probability that the patient has that disease. A high score does not establish that a diagnosis outside this vocabulary is absent. Three candidate classes currently have no unique exact MONDO mapping; their GMDB IDs and names remain available.

All individual results and embeddings remain in the local output directory. Standard output contains aggregate counts only. Check `route`: the `face_hpo` request may return `face_only` or `hpo_only` when a modality is missing. Compare modalities on the same eligible patients rather than mixing fallback patients into a paired multimodal endpoint.

### Optional local case retrieval

Prepare a second JSONL file with the same schema for an approved, independent local gallery:

```bash
bash run.sh --input /path/to/bch_queries.jsonl --gallery /path/to/local_gallery.jsonl --output /path/to/bch_retrieval_v1 --device cuda
```

The output adds a ranked list of gallery case IDs and similarity scores; these are **case matches**, not classifier probabilities or disease-level retrieval metrics. No CHOP/GMDB patient gallery is distributed. Identical case IDs are excluded automatically. Both sides need a facial image for this released retrieval rule. Query/gallery patient and family overlap must also be checked locally: different pseudonyms for the same patient cannot be detected by this runner.

## 5. BCH evaluation plan

Freeze this version before inspecting BCH outcomes. Start with candidate coverage: map each reference diagnosis to `assets/diseases.json`, preserving exact/ambiguous/out-of-vocabulary status. Do not silently map a broad clinical label to a narrower trained diagnosis. Report the full cohort size and disease coverage alongside covered-disease performance.

For diagnosis, report patient-micro and disease-macro Top-1/Top-5, patient/disease counts, and patient-level paired confidence intervals. Compare the three modality modes on the **same patients with valid facial inputs, eligible HPO, and covered diagnoses**. Separately report all-patient fallback, abstention and coverage results. Use only HPO available at the intended diagnostic time; exclude information derived from the reference diagnosis or obtained after it.

Keep all views and visits from one patient together. Complete local identity/family and cross-source overlap checks before calling this an independent external test. This package alone cannot verify BCH overlap with prior training sources. Do not tune thresholds, choose checkpoints, or adjust fusion on the final test cohort.

The current release supports external research evaluation. Clinical multimodal benefit at BCH has not yet been established. In particular, historical CHOP HPO results with unresolved polarity are not confirmatory clinical evidence.

## 6. Troubleshooting and provenance

- **`CUDA_unavailable...`**: use `--device cpu`, or use the site's compatible CUDA PyTorch installation.
- **Memory error**: reduce `--batch-size` to 1; use CPU if needed. No training process is launched.
- **`output_directory_already_exists`**: choose a new output path. Completed or partial prior results are not overwritten.
- **`image_file_missing`**: resolve image paths relative to the input JSONL directory.
- **`package_integrity_check_failed`**: re-extract the original bundle; do not bypass the check.
- **Mostly visual fallbacks**: inspect eligible HPO counts and unsupported terms locally; do not force uncertain terms to positive.

Visual architecture provenance: [TransFace](https://github.com/DanJun6737/TransFace). This package contains a compact inference implementation and derived research weights; it does not grant new rights to upstream models or source datasets. Preserve institutional sharing terms and upstream attribution. Evaluation artifacts should remain within the approved research environment.

Verification details are recorded in `VERIFICATION.json`. Export verification is a software-equivalence test, not clinical validation or a fresh BCH experiment.
