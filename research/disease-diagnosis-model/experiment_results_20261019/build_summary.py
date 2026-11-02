"""Export aggregate results and figures only; no patient records in outputs."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
from pathlib import Path
import collections
import csv
import datetime
import json

OUT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(OUT/'.matplotlib_cache'))
BASE = Path('/opt/craniofacial/SyndroDx-python-slurm/pretraining')
V1, V2 = BASE/'dinov3_craniofacial_v1', BASE/'dinov3_craniofacial_v2_official'
WORK = OUT.parent/'training_diagnostics_20261017'


def read(path):
    return json.loads(path.read_text())


def metrics(path, keys, latest_restart=False):
    rows = []
    for line in path.read_text().splitlines():
        try:
            data = json.loads(line)
        except ValueError:
            continue
        rows.append({key:data[key] for key in keys if key in data})
    if latest_restart:
        starts = [i for i, row in enumerate(rows) if row.get('iteration') == 0]
        if starts:
            rows = rows[starts[-1]:]
    return rows


def main():
    os.umask(0o077)
    release = read(V1/'dataset_release/release.json')
    data = {'sampling_weights':release['sampling_weights'],
            'healthy_reference_clinically_verified':release['clinically_verified_healthy_controls'],
            'disease_labels_used_in_ssl':release['labels_used'],
            'patient_split_disjoint':release['patient_split_disjoint'],
            'automated_qc_only':release['automated_qc_only'], 'released_partitions':{}}
    for name in ('train', 'validation', 'test', 'reference_train'):
        spec = release['manifests'][name]
        count, groups, views = 0, set(), collections.Counter()
        with Path(spec['path']).open(newline='') as stream:
            for row in csv.DictReader(stream, delimiter='\t'):
                count += 1
                groups.add(row['group_id'])
                views[row.get('view', 'unspecified')] += 1
        data['released_partitions'][name] = {'images':count, 'groups':len(groups), 'views':dict(views)}
    old = read(V1/'evaluation/report.private.json')
    # This report's writer contains aggregate metrics only. Exclude provenance
    # hashes and any unrelated metadata from the standalone export.
    old = {key:old[key] for key in ('state', 'completed_at_utc', 'models', 'metadata_endpoints',
                                   'test_used_for_hyperparameters', 'clinical_adjudication',
                                   'external_center_validation', 'primary_result')}
    final = read(V2/'validation/step_003200/report.private.json')
    curve = []
    for path in sorted((V2/'validation').glob('step_*/report.private.json')):
        item = read(path)
        curve.append({'step':item['completed_steps'], 'endpoints':item['endpoints']})
    heads = {}
    for name in ('initial', 'warmup200', 'warmup1000'):
        item = read(V2/'diagnostics'/f'head_audit_{name}.json')
        heads[name] = {key:item[key] for key in ('training_patients', 'global_views', 'prototype_count',
                                                'train_only', 'test_used', 'temperatures', 'completed_at_utc')}
    snapshot = {
        'snapshot_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'privacy':'aggregate results only; no patient IDs, images, clinical records, labels, embeddings or weights',
        'data':data, 'v1_historical_test':old,
        'v1_training':metrics(V1/'training/metrics.private.jsonl',
                              ['iteration','total_loss','world_size','global_batch_size'], True),
        'v2_final_validation':final, 'v2_validation_curve':curve,
        'v2_frozen_ridge_probe':read(V2/'diagnostics/frozen_probe_extended_alpha_report.json'),
        'v2_training':metrics(V2/'training/training_metrics.json',
                              ['iteration','total_loss','global_batch_size']),
        'v2_representation_audit':read(V2/'diagnostics/top1_audit_20261017.json'),
        'frozen_head_warmup':{'status':read(WORK/'warmup_pilot/status.json'), 'head_comparisons':heads,
                             'training':metrics(WORK/'warmup_pilot/training/training_metrics.json',
                              ['iteration','total_loss','dino_global_crops_loss','ibot_loss',
                               'dino_teacher_entropy','dino_teacher_information'])},
        'limitations':['Metadata labels are not clinical gold standard.',
                       'V1 test and V2 validation cohorts and probe protocols differ.',
                       'V1 test was inspected; it is not an untouched confirmation set for future tuning.',
                       'V2 disease validation has42 patients,18 labels,10 singleton labels.',
                       'Patient bootstrap intervals and repeated checkpoint looks are exploratory.',
                       'No independent external-center, GMDB or GestaltMatcher-Arc benchmark completed.',
                       'Head information statistics are not disease accuracy.',
                       'The warmup changes both head training rate and freezing protocol; it does not isolate a single cause.']}
    (OUT/'aggregate_results.json').write_text(json.dumps(snapshot, indent=2)+'\n')
    plot(snapshot)
    print('Aggregate-only results JSON, PNG and PDF created.')


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'figure.facecolor':'white', 'axes.facecolor':'white'})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))
    blue, red = '#244c73', '#b45143'
    ax = axes[0, 0]
    curve = result['v2_validation_curve']
    x = [row['step'] for row in curve]
    y = [row['endpoints']['disease_or_disorder']['models']['adapted']['top1_macro']*100 for row in curve]
    original = result['v2_final_validation']['endpoints']['disease_or_disorder']['models']['original_dinov3']['top1_macro']*100
    ax.plot(x, y, color=red, marker='o', markersize=3, label='V2 continual pretraining')
    ax.axhline(original, color=blue, linestyle='--', label='Original DINOv3')
    ax.axvline(2900, color='.55', linestyle=':', linewidth=1, label='Batch increase after step 2900')
    ax.set(title='A. V2 disease retrieval: fixed validation cohort', xlabel='Optimizer steps', ylabel='Macro Top-1 (%)', ylim=(0, 32))
    ax.legend(fontsize=8, loc='lower left')
    ax.text(.98, .97, '42 patients / 18 disease labels', ha='right', va='top', transform=ax.transAxes, fontsize=9)
    ax = axes[0, 1]
    groups, before, after = [], [], []
    for category, short in [('disease_or_disorder','Disease'),('structural_diagnosis','Structural')]:
        for endpoint, short_ep in [('retrieval','retrieval'),('probe','linear probe')]:
            group = (result['v2_final_validation']['endpoints'][category] if endpoint=='retrieval'
                     else result['v2_frozen_ridge_probe']['endpoints'][category])['models']
            a, b = group['original_dinov3'], group['adapted']
            before.append((a['top1_macro'] if endpoint=='retrieval' else a['validation']['top1_macro'])*100)
            after.append((b['top1_macro'] if endpoint=='retrieval' else b['validation']['top1_macro'])*100)
            groups.append(short+'\n'+short_ep)
    positions = np.arange(len(groups))
    for values, offset, color, label in [(before,-.18,blue,'Original'),(after,.18,red,'V2 final')]:
        bars = ax.bar(positions+offset, values, width=.35, color=color, label=label)
        ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=2)
    ax.set(title='B. V2 validation: distinct downstream endpoints', ylabel='Macro Top-1 (%)', ylim=(0,90))
    ax.set_xticks(positions, groups)
    ax.legend(fontsize=8, loc='upper left')
    ax = axes[1, 0]
    rows = result['frozen_head_warmup']['training']
    ax.plot([r['iteration']+1 for r in rows], [r['total_loss'] for r in rows], color=blue)
    ax.set(title='C. Frozen-backbone SSL-head warmup', xlabel='Optimizer steps (logged points)', ylabel='Total SSL loss')
    ax.text(.02,.03,'Backbone unchanged; last loss log at step 991', transform=ax.transAxes, fontsize=9)
    ax = axes[1, 1]
    heads = result['frozen_head_warmup']['head_comparisons']
    positions = [0,200,1000]
    values = [heads[name]['temperatures']['0.07']['sinkhorn']['mutual_information_across_samples']
              for name in ('initial','warmup200','warmup1000')]
    ax.plot(positions, values, marker='o', color=blue)
    for xx, yy in zip(positions, values):
        ax.annotate(f'{yy:.3f}',(xx,yy),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
    ax.set(title='D. Matched head-output diagnostic', xlabel='Warmup steps', ylabel='Across-input information (nats)', ylim=(0,.72))
    ax.text(.03,.93,'Fixed temperature 0.07; same 64 training patients\nDiagnostic signal, not disease accuracy',transform=ax.transAxes,va='top',fontsize=9)
    fig.suptitle('Craniofacial representation experiments — aggregate results', fontsize=14)
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(OUT/'experiment_summary.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT/'experiment_summary.pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
