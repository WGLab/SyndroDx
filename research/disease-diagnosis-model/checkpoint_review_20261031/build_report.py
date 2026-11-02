"""Compile existing aggregate results; never load images, identities or features."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
MATCH = ROOT / 'matched_craniofacial_benchmark_20261031'
sources = {}
source_cache = {}


def read(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if path not in source_cache:
        raw = path.read_bytes()
        sources[path] = hashlib.sha256(raw).hexdigest()
        source_cache[path] = json.loads(raw)
    return source_cache[path]


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def pct(v):
    return '' if v is None else round(100 * float(v), 6)


def metrics(row, prefix, data, suffix=''):
    for k in (1, 5, 10):
        row[f'{prefix}_top{k}_pct'] = pct(data.get(f'top{k}{suffix}'))
        if f'top{k}_SD' in data:
            row[f'{prefix}_top{k}_seed_SD_pp'] = pct(data[f'top{k}_SD'])


def md_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def pair(row, prefix):
    a, b = row.get(prefix + '_top1_pct'), row.get(prefix + '_top5_pct')
    return '—' if a is None or a == '' else f'{a:.2f} / {b:.2f}'


gm = read(MATCH / 'results.json')
ex = read(MATCH / 'expert/results.json')
selection = read(MATCH / 'selection.json')
pins = read(MATCH / 'initialization_pins.json')
case_defs = read(MATCH / 'cases.json')['cases']
assert gm['state'] == ex['state'] == 'complete'
assert gm['selection'] == selection
assert ex['selection_source_sha256'] == hashlib.sha256((MATCH / 'selection.json').read_bytes()).hexdigest()

labels = {'r50': 'R50', 'transface_original': 'TransFace',
          'transface_chop_cpt': 'TransFace + CHOP identity CPT step1000'}
matched, fewshot, registry = [], [], []
for name, g in gm['frozen_initializations'].items():
    arm = name.removeprefix('init__')
    e = ex['frozen_initializations'][name]
    r = dict(model_id=name, model=labels[arm], setting='frozen_encoder',
             GMDB_supervision='none', downstream_seeds=0, learning_rate='',
             checkpoint_paths=pins[arm]['path'], checkpoint_sha256=pins[arm]['sha256'])
    metrics(r, 'GMDB_frequent', g['frequent'])
    metrics(r, 'GMDB_rare', g['rare'])
    metrics(r, 'CHOP_expert', e['retrieval'])
    matched.append(r)
    f = dict(model_id=name, model=r['model'], setting=r['setting'])
    for k in (1, 5, 10):
        metrics(f, f'GMDB_{k}shot', g['fewshot_common'][str(k)])
        metrics(f, f'CHOP_{k}shot_TWO_CLASSES_ONLY', e['fewshot'][str(k)])
    fewshot.append(f)
    registry.append(dict(strategy='frozen_encoder', model_id=name, path=pins[arm]['path'],
                         sha256=pins[arm]['sha256'], exists=Path(pins[arm]['path']).is_file(),
                         selection='Public initialization or previously development-locked CPT step1000',
                         matched_GMDB_test=True, matched_CHOP_test=True))

for setting, cells in gm['summary'].items():
    for cell, g in cells.items():
        arm, regime = cell.split('__')
        names = selection['settings'][setting][cell]
        assert len(names) == 4
        lr = case_defs[names[0]]['lr']
        paths = [read(MATCH / 'cases' / n / 'result.json')['path'] for n in names]
        e = ex['summary'][setting][cell]
        r = dict(model_id=setting + '__' + cell, model=labels[arm], setting=setting,
                 GMDB_supervision=regime, downstream_seeds=4, learning_rate=lr,
                 checkpoint_paths=';'.join(paths), checkpoint_sha256='see checkpoint_registry.csv')
        metrics(r, 'GMDB_frequent', g['frequent'], '_mean')
        metrics(r, 'GMDB_rare', g['rare'], '_mean')
        metrics(r, 'CHOP_expert', e['retrieval'], '_mean')
        r['CHOP_expert_top1_seed_SD_pp'] = pct(e['retrieval_top1_SD'])
        matched.append(r)
        f = dict(model_id=r['model_id'], model=r['model'], setting=setting, regime=regime)
        for k in (1, 5, 10):
            metrics(f, f'GMDB_{k}shot', g['fewshot_common'][str(k)], '_mean')
            metrics(f, f'CHOP_{k}shot_TWO_CLASSES_ONLY', e['fewshot'][str(k)], '_mean')
        fewshot.append(f)

for name, case in case_defs.items():
    q = MATCH / 'cases' / name / 'result.json'
    if not q.exists():
        continue
    r = read(q)
    registry.append(dict(strategy='matched_' + case['regime'], model_id=name,
                         path=r['path'], sha256=r['sha256'], exists=Path(r['path']).is_file(),
                         arm=case['arm'], learning_rate=case['lr'], seed=case['seed'],
                         epoch=r['epoch'], validation_top1_pct=pct(r['validation']['top1']),
                         validation_top5_pct=pct(r['validation']['top5']),
                         selection='Fixed epoch50; all four seeds retained, no test-based seed selection',
                         matched_GMDB_test=(MATCH / 'case_metrics' / (name + '.json')).exists(),
                         matched_CHOP_test=(MATCH / 'expert/receipts' / (name + '.json')).exists()))

historical = []


def add_hist(name, source, model, layout='patient', notes='', checkpoint=''):
    d = read(source)
    value = d['models'][model]
    row = dict(model=name, source=str(ROOT / source), evaluation_protocol=layout,
               notes=notes, checkpoint=checkpoint,
               current_matched_CHOP_status='not evaluated for this exact historical checkpoint')
    if layout == 'image_macro_patient_disjoint_queries':
        v = value.get('metrics', value)['patient_disjoint_query_supplement']
        metrics(row, 'GMDB_frequent', v['frequent_to_frequent']['macro'])
        metrics(row, 'GMDB_rare', v['rare_to_rare']['macro'])
    else:
        metrics(row, 'GMDB_frequent', value['frequent'])
        metrics(row, 'GMDB_rare', value['rare'])
        for k, v in value.get('fewshot', {}).items():
            row[f'GMDB_{k}shot_top1_pct'] = pct(v.get('macro_top1', v.get('top1')))
    historical.append(row)
    if checkpoint:
        registry.append(dict(strategy='historical', model_id=name, path=checkpoint,
                             exists=Path(checkpoint).exists(), selection=notes,
                             matched_GMDB_test=False, matched_CHOP_test=False))
    return row


add_hist('Original DINOv3', 'gmdb_checkpoint_eval_20261026/results.json', 'original_dinov3',
         'image_macro_patient_disjoint_queries', 'Historical image-level disease macro, not current patient-mean protocol')
add_hist('DINOv3 + craniofacial SSL CPT step5250', 'gmdb_checkpoint_eval_20261026/results.json', 'craniofacial_cpt',
         'image_macro_patient_disjoint_queries', 'Teacher checkpoint; historical image-level protocol',
         str(ROOT / 'gmdb_checkpoint_eval_20261026/checkpoint_snapshot'))
add_hist('ArcFace R100 Glint360K, no GMDB', 'arcface_glint360k_eval_20261026/results.json',
         'arcface_r100_glint360k_no_gmdb', 'image_macro_patient_disjoint_queries',
         'Historical face-only reference; not current RGB112 patient-mean benchmark')

old_supervised = read('gmdb_supervised_transface_20261026/results.json')
old_expert = read('chop_expert_comparison_20261027/results.json')
old_expert_map = {'original': 'original_gmdb', 'identity_cpt': 'cpt_gmdb'}
for key in ('original', 'identity_cpt'):
    case = old_supervised['models'][key]['selected_case']['name']
    ckpt = ROOT / 'gmdb_supervised_transface_20261026/cases' / case / 'best.pt'
    row = add_hist('Early TransFace ' + key + ' + GMDB WCE',
                   'gmdb_supervised_transface_20261026/results.json', key,
                   'image_macro_patient_disjoint_queries',
                   'Old CPT step2500 when used; historical CHOP cohort 314 gallery/264 queries/10 diagnoses, not current 287/238/9', str(ckpt))
    metrics(row, 'historical_CHOP', old_expert['models'][old_expert_map[key]])
    row['current_matched_CHOP_status'] = 'Only old CHOP cohort available; do not mix with current cohort'
for key, name in [('original', 'Early original TransFace'), ('cpt', 'Early CHOP identity CPT step2500')]:
    row = dict(model=name, source=str(ROOT / 'chop_expert_comparison_20261027/results.json'),
               evaluation_protocol='historical_CHOP_patient_LOPO',
               notes='Old 314-patient gallery, 264 query patients, 10 query diagnoses; different CPT checkpoint and cohort',
               current_matched_CHOP_status='Current-cohort result absent for step2500; original has separate matched control')
    metrics(row, 'historical_CHOP', old_expert['models'][key])
    historical.append(row)

for key in ('r100_single', 'r100_TTA', 'unadapted_TTA', 'transface_single', 'transface_TTA'):
    add_hist('Early GM-Arc comparison: ' + key, 'gestaltmatcher_arc_comparison_20261027/results.json', key,
             'image_macro_patient_disjoint_queries',
             'Historical disease feature fine-tuning/TTA; not official author disease weights; protocol differs')

for folder, title in [('gmdb_arcface_finetune_20261029', 'Earlier full ArcFace disease fine-tuning'),
                      ('gmdb_arcface_then_wce_20261030', 'Disease ArcFace then WCE'),
                      ('gmdb_direct_wce_20261030', 'Earlier direct WCE')]:
    d = read(folder + '/results.json')
    for key, arm in [('original', 'original'), ('selected', 'chop_cpt')]:
        selected = d['development_selection'][arm]
        add_hist(title + ': ' + arm, folder + '/results.json', key,
                 notes='Historical patient-macro/fold-mean protocol; selected single seed and best epoch, older quarantine; not matched to current four-seed epoch50 results',
                 checkpoint=selected['path'])

alignment = read('alignment_matched_comparison_20261028/results.json')
row = dict(model='CPT1000 + BiomedBERT alignment, projected visual 256d',
           source=str(ROOT / 'alignment_matched_comparison_20261028/results.json'),
           evaluation_protocol='historical_patient_macro_fold_mean',
           notes='Vision backbone frozen; changed projection space, not corruption of raw 512d encoder',
           current_matched_CHOP_status='not evaluated for this projection')
for task in ('frequent', 'rare'):
    metrics(row, 'GMDB_' + task, alignment['retrieval']['models']['selected'][task])
for k, v in alignment['retrieval']['models']['selected']['fewshot'].items():
    row[f'GMDB_{k}shot_top1_pct'] = pct(v['macro_top1'])
historical.append(row)

r50 = read('gm_arc_r50_reproduction_20261030/results.json')
for key in ('code_keep_bn', 'paper_remove_bn'):
    row = dict(model='R50 paper-method reproduction: ' + key,
               source=str(ROOT / 'gm_arc_r50_reproduction_20261030/results.json'),
               evaluation_protocol='historical_patient_macro_BGR',
               notes='BGR input; backbone BN buffers update; four seeds fixed epoch50; older quarantine; differs from matched RGB/frozen-buffer experiment',
               current_matched_CHOP_status='not evaluated for these exact checkpoints')
    for task in ('frequent', 'rare'):
        metrics(row, 'GMDB_' + task, r50['summary'][key][task], '_mean')
    for k in (1, 5, 10):
        row[f'GMDB_{k}shot_top1_pct'] = pct(mean(v['fewshot'][str(k)]['macro_top1'] for v in r50['models'][key].values()))
    historical.append(row)
    for seed in r50['models'][key]:
        path = ROOT / 'gm_arc_r50_reproduction_20261030' / (key + '_' + seed) / 'last.pt'
        registry.append(dict(strategy='R50_historical_reproduction', model_id=key + '_' + seed,
                             path=str(path), exists=path.is_file(), seed=seed, epoch=50,
                             selection='Fixed final epoch; BGR historic protocol',
                             matched_GMDB_test=False, matched_CHOP_test=False))

add_hist('HF aaronwzl/GestaltMatcher-Arc-GMDB R50 reproduction',
         'gestaltmatcher_hf_evaluation_20261029/results.json', 'selected',
         notes='User-provided reproduction, not author-official. v1.1.2 training manifest unknown; overlap with local v1.1.0 test unresolved. Not eligible for clean superiority claim.',
         checkpoint=str(ROOT / 'gestaltmatcher_hf_evaluation_20261029/assets/s1_glint360k_r50_512d_gmdb_v1.1.2_e50.pt'))

development = []
objective = read('transface_objective_comparison_20261028/results.json')
for arm, observations in objective['arms'].items():
    for obs in observations:
        dev = obs.get('gmdb_development', {})
        r = dict(strategy=arm, step=obs['step'], source=str(ROOT / 'transface_objective_comparison_20261028/results.json'),
                 scope='GMDB development only; not test and not CHOP expert')
        metrics(r, 'GMDB_development_retrieval', dev.get('retrieval', {}))
        development.append(r)
        path = ROOT / 'transface_objective_comparison_20261028' / arm / 'checkpoints' / f"step_{obs['step']:07d}.pt"
        if path.is_file():
            registry.append(dict(strategy='objective_' + arm, model_id=arm + '_step' + str(obs['step']),
                                 path=str(path), exists=True, step=obs['step'],
                                 selection='Saved evaluation point, not newly selected on test',
                                 matched_GMDB_test=False, matched_CHOP_test=False))

seed_rounds = []
for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir() or not folder.name.startswith(('arcface_seed_round_', 'arcface_extra_seed_round_', 'arcface_fill_seed_round_')):
        continue
    file = folder / 'results.json'
    if not file.exists():
        continue
    d = read(file)
    for arm, observations in d.get('arms', {}).items():
        if not observations:
            continue
        latest = max(observations, key=lambda x: x['step'])
        best = max(observations, key=lambda x: (x.get('gmdb_development', {}).get('retrieval', {}).get('top1', -1), -x['step']))
        r = dict(round=folder.name, arm=arm, latest_evaluated_step=latest['step'], best_development_step=best['step'],
                 scope='Development-only repeated-seed sensitivity, no matched test result for this seed')
        metrics(r, 'latest_development', latest.get('gmdb_development', {}).get('retrieval', {}))
        metrics(r, 'best_development', best.get('gmdb_development', {}).get('retrieval', {}))
        seed_rounds.append(r)
        for obs in (latest, best):
            path = folder / arm / 'checkpoints' / f"step_{obs['step']:07d}.pt"
            if path.is_file() and not any(z.get('path') == str(path) for z in registry):
                registry.append(dict(strategy='CHOP_identity_CPT_repeated_seed', model_id=folder.name + '/' + arm + '/step' + str(obs['step']),
                                     path=str(path), exists=True, step=obs['step'],
                                     selection='Latest or development-best saved point; not a matched test competitor',
                                     matched_GMDB_test=False, matched_CHOP_test=False))

multimodal = []


def add_mm(name, v, source, notes):
    r = dict(model=name, source=str(ROOT / source), scope='GMDB development diagnosis; not retrieval test',
             CHOP_expert_status='not evaluated under current matched protocol', notes=notes)
    for k in (1, 5, 10):
        r[f'top{k}_pct'] = pct(v.get(f'macro_top{k}'))
    multimodal.append(r)


for folder, obj, label in [('transface_biomedbert_alignment_20261027/fusion_results.json', 'models', 'CPT2500 + BiomedBERT'),
                           ('alignment_matched_comparison_20261028/results.json', 'diagnosis', 'CPT1000 + BiomedBERT')]:
    d = read(folder)
    for arm, v in d[obj].items():
        add_mm(label + ' ' + arm, v['best'], folder, '321 paired development patients; best epoch selected by development Top5')
qf = 'cranioface_qwen_alignment_20261027/classifier/results.json'
for setting, arms in read(qf)['results'].items():
    for arm, v in arms.items():
        add_mm('Qwen32B classifier ' + setting + ' ' + arm, v, qf, '321 development patients; 216 candidate diagnoses; no clean CHOP expert test')
for qf, label in [('cranioface_qwen_alignment_20261027/diagnosis_results_exhaustive.json', 'Qwen32B generative exhaustive'),
                  ('cranioface_qwen_alignment_20261027/paired_diagnosis_results_exhaustive.json', 'Qwen32B paired-HPO exhaustive')]:
    d = read(qf)
    for arm, v in d['results'].items():
        add_mm(label + ' ' + arm, v, qf, '216 exhaustive disease candidates; token mean log-probability; exploratory development')

# Early SSL runs have internal probes, not a result on the current test cohorts.
missing = []
for folder, title in [('transface_vitl_cpt_20261026', 'TransFace DINO SSL step4000'),
                      ('transface_cpt_low_lr_20261026', 'TransFace low-LR DINO SSL step1750'),
                      ('arcface_cpt_20261026', 'R100 CPT pilot, stopped at step0')]:
    d = read(folder + '/latest_checkpoint.json')
    missing.append(dict(strategy=title, checkpoint=d.get('path', ''),
                        GMDB_current_matched='not evaluated', CHOP_current_matched='not evaluated',
                        note='Internal development probes do not constitute either requested test result'))
    registry.append(dict(strategy='early_SSL_or_pilot', model_id=title, path=d.get('path', ''),
                         exists=Path(d.get('path', '')).is_file(), step=d.get('step'),
                         selection='Latest saved point; no current matched test evaluation',
                         matched_GMDB_test=False, matched_CHOP_test=False))

write_csv('matched_models.csv', matched)
write_csv('matched_fewshot.csv', fewshot)
write_csv('historical_results.csv', historical)
write_csv('objective_development.csv', development)
write_csv('repeated_seed_development.csv', seed_rounds)
write_csv('multimodal_development.csv', multimodal)
write_csv('missing_matched_evaluations.csv', missing)
write_csv('checkpoint_registry.csv', registry)

contrasts = []
for dataset, key, d in [('GMDB', 'paired_top1', gm), ('CHOP_expert', 'paired_retrieval_top1', ex)]:
    for setting, comparisons in d[key].items():
        for comparison, entry in comparisons.items():
            entries = entry if dataset == 'GMDB' else {'retrieval': entry}
            for task, v in entries.items():
                contrasts.append(dict(dataset=dataset, setting=setting, comparison=comparison, task=task,
                                      delta_pp=pct(v['delta']), ci95_low_pp=pct(v['hierarchical_paired_95_CI'][0]),
                                      ci95_high_pp=pct(v['hierarchical_paired_95_CI'][1]),
                                      note='Secondary intervals unadjusted; primary contrast defined in protocol'))
write_csv('paired_uncertainty.csv', contrasts)

report = [
    '# Craniofacial model checkpoint review',
    '',
    'Generated: ' + datetime.now(ZoneInfo('America/New_York')).isoformat(timespec='seconds'),
    '',
    'This report compiles existing aggregate outputs only. No new training, test-based checkpoint replacement, image inspection or patient-record export was performed.',
    '',
    '**Recommendation:** retain the already development-selected CHOP identity-ArcFace CPT TransFace-L step1000 as the general encoder candidate. Keep disease fine-tuning and HPO projection heads as task-specific branches. There is no universal winning checkpoint across the requested endpoints.',
    '',
    '## 1. Models evaluated on both current cohorts',
    '',
    'All values are disease-macro percentages, shown as Top1 / Top5. Fine-tuned rows are means of all four downstream seeds, not ensembles or individual deployable checkpoints. All primary fine-tuning checkpoints are epoch50. “Common” uses the same learning rate across initializations; “tuned” selects one learning rate per initialization using four-seed mean GMDB validation classification Top5, never CHOP.',
    '',
]
rows = []
for r in matched:
    setting = {'frozen_encoder': 'No GMDB tuning', 'common_recipe': 'Common LR', 'equal_budget_tuned': 'Validation-selected LR'}[r['setting']]
    method = {'none': 'None', 'wce_feature_only': 'WCE feature only', 'arcface_full': 'Full ArcFace'}[r['GMDB_supervision']]
    rows.append([r['model'], setting + '; ' + method, str(r['learning_rate']) or '—', pair(r, 'GMDB_frequent'), pair(r, 'GMDB_rare'), pair(r, 'CHOP_expert')])
report += [md_table(['Model', 'GMDB adaptation', 'LR', 'GMDB frequent', 'GMDB rare', 'CHOP expert'], rows), '',
           '- WCE: weighted cross-entropy, frozen encoder body, native 512d feature layer trained. Full ArcFace: disease-supervised backbone adaptation.',
           '- GMDB: 659 frequent query patients/275 diagnoses and 1,052 unique rare query patients/306 diagnoses, after the shared quarantine. Rare results average each patient’s appearances across ten folds before disease macro averaging.',
           '- Rare diagnoses are absent from GMDB disease-supervision labels; this does not establish absence from all identity-CPT patient exposures.',
           '- CHOP: 287 gallery patients/1,683 images/58 total diagnoses; 238 query patients across 9 retrievable diagnoses. The other 49 patients are singleton distractors. Patient self-matches are excluded; view-balanced patient embeddings are used.',
           '- Audits exclude known identity and detected exact/perceptual overlaps. Unknown identity aliases remain unresolved. CHOP is a previously inspected, same-center expert-source cohort; it is not untouched external or prospective validation.',
           '', '## 2. Matched few-shot gallery results', '',
           'Top1 percentages. GMDB uses the same 165 candidate diagnoses and 548 query patients at every shot, with 20 support draws. This is reduced-gallery prototype retrieval after disease supervision, not training the entire model using only k labels. CHOP has only TWO eligible diagnoses and 44 query patients; its Top5/Top10 are trivially 100% and must not be used as evidence of diagnostic breadth.', '']
rows = []
for r in fewshot:
    tag = r['setting'] + ('; ' + r['regime'] if 'regime' in r else '')
    rows.append([r['model'], tag] +
                [f"{r[f'GMDB_{k}shot_top1_pct']:.2f}" for k in (1, 5, 10)] +
                [f"{r[f'CHOP_{k}shot_TWO_CLASSES_ONLY_top1_pct']:.2f}" for k in (1, 5, 10)])
report += [md_table(['Model', 'Setting', 'GMDB 1', 'GMDB 5', 'GMDB 10', 'CHOP 1*', 'CHOP 5*', 'CHOP 10*'], rows), '',
           '## 3. What the paired comparisons support', '',
           '- Common-LR full ArcFace: CPT minus original TransFace, GMDB rare Top1 +1.65 pp (95% CI −0.60 to +4.33); CHOP Top1 +10.59 pp (−0.50 to +29.41). Neither interval establishes a stable positive effect.',
           '- Validation-selected-LR full ArcFace: CPT minus original, GMDB rare −9.52 pp (−12.65 to −6.44); CHOP −6.52 pp (−25.00 to +4.33). Classification-based LR selection did not preserve retrieval transfer.',
           '- Frozen CPT versus frozen original: GMDB frequent 27.85 → 34.12, rare 27.15 → 31.94; CHOP 15.32 → 28.17. These are single fixed initialization checkpoints; the matched report does not supply a paired confidence interval for this frozen comparison.',
           '- The experiment changes learning rate and resulting training trajectories; it identifies an association with the selection policy, not a uniquely proven cause of degradation.',
           '', '## 4. Earlier strategies: retained as historical evidence', '',
           'Do not rank these directly against section 1. Older experiments use image-level macro or different patient/fold aggregation, quarantine, color preprocessing, CPT checkpoint, seed selection, and/or expert cohorts. A dash means no recorded result for that exact model and cohort; it does not mean zero.', '']
rows = [[r['model'], pair(r, 'GMDB_frequent'), pair(r, 'GMDB_rare'), pair(r, 'historical_CHOP'), r['notes']] for r in historical]
report += [md_table(['Strategy', 'GMDB frequent', 'GMDB rare', 'OLD CHOP only', 'Interpretation'], rows), '',
           'The externally downloaded R50 is the user-provided GMDB reproduction, not the original authors’ official disease checkpoint. Its training manifest and overlap with the local test are unresolved; the 55.44/45.46 Top1 values cannot establish a clean ranking against locally audited training.',
           '', '## 5. SSL and repeated identity-CPT seeds', '',
           'The controlled objective study has four arms. The table reports the latest evaluated step, not a test-selected best point. The complete trajectory is in objective_development.csv.', '']
rows = []
for arm, observations in objective['arms'].items():
    v = max(observations, key=lambda x: x['step'])
    m = v['gmdb_development']['retrieval']
    rows.append([arm, v['step'], f"{pct(m['top1']):.2f}", f"{pct(m['top5']):.2f}", 'Not evaluated', 'Not evaluated'])
report += [md_table(['Objective / backbone LR', 'Last evaluated step', 'GMDB dev Top1', 'GMDB dev Top5', 'Current GMDB test', 'Current CHOP'], rows), '',
           f'Additionally, {len(seed_rounds)} repeated-seed arms have existing development summaries, grouped across {len(set(r["round"] for r in seed_rounds))} rounds. They belong to the same identity-CPT strategy, not new methods. See repeated_seed_development.csv for every arm’s latest and development-best values. Those seeds have not all received both matched tests, so no stronger test claim is made.', '',
           md_table(['Earlier run', 'Current GMDB test', 'Current CHOP'], [[r['strategy'], r['GMDB_current_matched'], r['CHOP_current_matched']] for r in missing]), '',
           'The original DINOv3 route has historical GMDB test values in section 4; the controlled TransFace SSL comparison is development-only. These are different initializations and must not be described as one SSL result.',
           '', '## 6. Multimodal diagnosis is a different endpoint', '',
           'These are development-set disease diagnosis results, not the image retrieval tests in sections 1–4. CHOP expert matched multimodal diagnosis is not available. Do not substitute these Top1/Top5 numbers into the retrieval ranking.', '',
           md_table(['Model / input', 'Development Top1', 'Top5', 'Top10'], [[r['model']] + [f"{r[f'top{k}_pct']:.2f}" for k in (1, 5, 10)] for r in multimodal]), '',
           'BiomedBERT CPT1000 fusion improves development Top1 by 4.92 pp and Top5 by 4.07 pp relative to its HPO-only arm. The visual projection reduces historical image retrieval relative to raw 512d features, so the projection should remain a separate multimodal branch.',
           '', '## 7. Checkpoint decision', '',
           '1. General craniofacial encoder candidate: retain CHOP identity-ArcFace CPT step1000, using its native 512d embedding. It was locked using development retrieval before the current tests, and both current datasets show a positive frozen-CPT point estimate versus the original TransFace.',
           '2. Disease-specific retrieval candidate: retain the common-recipe CPT → full ArcFace epoch50 group (LR2e-6, seeds 11/29/47/83) for further validation. It does not demonstrate a reliable improvement over the raw CPT encoder on CHOP and is not an automatic replacement.',
           '3. GMDB rare-transfer reference: original TransFace → full ArcFace LR5e-7. It achieves 35.68% rare Top1 in the validation-selected group, versus 26.16% for validation-selected CPT. Retain it as a strong task-specific comparator.',
           '4. GMDB frequent retrieval reference: R50 → WCE, common LR5e-4, reaches 44.29% frequent Top1. Architecture size alone did not win this endpoint.',
           '5. Multimodal development candidate: CPT1000 + BiomedBERT fusion. Its development result is promising; there is no matched clinical-test result supporting final multimodal checkpoint selection yet.',
           '',
           '**Do not choose a single seed by CHOP/GMDB test performance.** Four-seed means are experiment summaries, not one checkpoint. Keep the existing development-locked step1000 while making the architecture decision. Any new choice guided by the repeatedly inspected CHOP/GMDB tests is exploratory; BCH should remain independent for the subsequent locked model.',
           '',
           'General encoder path:', '',
           '`' + pins['transface_chop_cpt']['path'] + '`', '',
           'Source checkpoint SHA256: `' + pins['transface_chop_cpt']['sha256'] + '`.', '',
           'The encoder checkpoint was preserved; no production selection file or running experiment was changed. checkpoint_registry.csv lists exact files for every matched training case and the principal saved historical/development candidates.',
           '', '## 8. Files', '',
           '- matched_models.csv: 15 model settings; Top1/5/10 and available seed SD on both datasets.',
           '- matched_fewshot.csv: all 15 settings; 1/5/10-shot Top1/5/10 on both datasets.',
           '- historical_results.csv: earlier strategy results with protocol limitations and missing tests.',
           '- objective_development.csv and repeated_seed_development.csv: SSL/ArcFace trajectories and repeated seeds.',
           '- multimodal_development.csv: HPO, facial+HPO and Qwen diagnosis development results.',
           '- checkpoint_registry.csv: exact model paths, stored hashes when available, validation metrics and evaluation coverage. exists checks file presence only, not rehashing every multi-GB checkpoint.',
           '- paired_uncertainty.csv: paired differences and confidence intervals.',
           '- source_manifest.json: hashes of the aggregate result files used to generate this report.',
           '']
(OUT / 'RESULTS.md').write_text('\n'.join(report))
manifest = {str(p): sources[p] for p in sorted(sources)}
(OUT / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
assert len(matched) == len(fewshot) == 15
assert len([r for r in registry if r['strategy'].startswith('matched_')]) == 72
for r in matched:
    for k, v in r.items():
        if k.endswith('_pct') and v != '':
            assert 0 <= v <= 100, (r['model_id'], k)
missing_files = [r['model_id'] for r in registry if not r.get('exists')]
(OUT / 'verification.json').write_text(json.dumps(dict(
    state='passed', matched_settings=15, matched_training_checkpoints=72,
    matched_inference_receipts=len(list((MATCH / 'inference').glob('*.json'))),
    historical_rows=len(historical), objective_development_points=len(development),
    repeated_seed_arms=len(seed_rounds), multimodal_rows=len(multimodal),
    checkpoint_registry_rows=len(registry), checkpoint_files_missing=missing_files,
    patient_records_or_images_loaded=False, running_tasks_modified=False,
    source_files=len(manifest)), indent=2) + '\n')
print((OUT / 'verification.json').read_text())
