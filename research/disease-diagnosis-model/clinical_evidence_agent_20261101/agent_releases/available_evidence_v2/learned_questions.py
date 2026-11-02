"""Training-only empirical question model with explicit recording uncertainty.

Probabilities describe recorded positive/negative/unknown outcomes in a research
dataset. They are not clinical penetrance, calibrated disease risk, or a reason
to insert an unobserved finding into a patient record.
"""
import numpy as np
from common import load, sha


CONFIGS = [
    dict(name='fixed', strategy='fixed', strength=5., temperature=1., prior_mix=.1),
    *[dict(name=f'answer_s{s}_t{t}', strategy='answer', strength=float(s), temperature=float(t), prior_mix=.1)
      for s, t in [(5, 1), (5, 2), (20, 1), (20, 2)]],
    *[dict(name=f'information_s5_t{t}', strategy='information', strength=5., temperature=float(t), prior_mix=.1)
      for t in [1, 2]],
    *[dict(name=f'balanced_s{s}_t{t}', strategy='balanced', strength=float(s), temperature=float(t), prior_mix=.1)
      for s, t in [(5, 1), (20, 2)]],
]


def split_policy_patients(labels, seed=20261102):
    """Stratified policy-fit/selection split within the existing training set."""
    rng = np.random.default_rng(seed)
    fit, select = [], []
    for disease in np.unique(labels):
        rows = rng.permutation(np.flatnonzero(labels == disease))
        count = max(1, int(round(len(rows) * .2))) if len(rows) >= 3 else 0
        select.extend(rows[:count]); fit.extend(rows[count:])
    return np.array(sorted(fit), np.int64), np.array(sorted(select), np.int64)


def fit_tables(records, labels, pool, classes):
    labels = np.asarray(labels)
    if len(records) != len(labels) or len(labels) == 0 or labels.min() < 0 or labels.max() >= classes:
        raise ValueError('invalid_policy_training_data')
    index = {term: j for j, term in enumerate(pool)}
    counts = np.zeros((classes, len(pool), 2), np.float64)
    sizes = np.bincount(labels, minlength=classes).astype(np.float64)
    for record, disease in zip(records, labels):
        for term, state in record.items():
            if term in index:
                if state not in (0, 1): raise ValueError('unknown_cannot_be_positive_or_negative')
                counts[disease, index[term], state] += 1
    if np.any(counts.sum(-1) > sizes[:, None]):
        raise ValueError('duplicate_patient_term_count')
    return dict(counts=counts, sizes=sizes, pool=np.asarray(pool, np.int64))


class LearnedQuestions:
    def __init__(self, tables, config, labels, inverse):
        self.config, self.labels, self.inverse = dict(config), list(labels), dict(inverse)
        self.pool = tables['pool'].astype(np.int64)
        self.term_to_col = {int(t): j for j, t in enumerate(self.pool)}
        self.code_to_term = {inverse[int(t)]: int(t) for t in self.pool}
        self.counts, self.sizes = tables['counts'].astype(np.float64), tables['sizes'].astype(np.float64)
        if self.counts.shape != (len(labels), len(self.pool), 2) or len(self.sizes) != len(labels):
            raise ValueError('policy_shape_mismatch')
        totals = self.counts.sum(0)
        n = self.sizes.sum()
        self.prior = (self.sizes + 1) / (n + len(labels))
        # A three-outcome Dirichlet prior includes unknown as a separate outcome.
        global_rates = (totals + .5) / (n + 1.5)
        strength = config['strength']
        self.rates = (self.counts + strength * global_rates[None]) / (self.sizes[:, None, None] + strength)
        if np.any(self.rates.sum(-1) >= 1): raise ValueError('invalid_outcome_probabilities')
        self.rate_log_rate = self.rates * np.log(np.maximum(self.rates, 1e-30))
        self.fixed_order = sorted(range(len(self.pool)), key=lambda j: (-totals[j, 0], inverse[int(self.pool[j])]))

    def utility(self, probabilities):
        p = np.asarray(probabilities, np.float64)
        if p.ndim != 2 or p.shape[1] != len(self.labels) or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError('invalid_disease_weights')
        p = np.power(np.maximum(p, 1e-30), 1 / self.config['temperature'])
        p /= p.sum(-1, keepdims=True)
        p = (1-self.config['prior_mix']) * p + self.config['prior_mix'] * self.prior[None]
        outcomes = np.stack([p @ self.rates[:, :, s] for s in [0, 1]], -1)
        answerability = outcomes.sum(-1)
        # Sum of expected posterior KL only for explicit answers. Unknown receives
        # no diagnostic credit and is never used to revise the disease ranker.
        gain = sum(p @ self.rate_log_rate[:, :, s] - outcomes[:, :, s] * np.log(np.maximum(outcomes[:, :, s], 1e-30)) for s in [0, 1])
        strategy = self.config['strategy']
        if strategy == 'answer': utility = answerability
        elif strategy == 'information': utility = gain
        elif strategy == 'balanced': utility = answerability + .5 * gain
        elif strategy == 'fixed':
            utility = np.zeros_like(answerability)
            utility[:, self.fixed_order] = np.arange(len(self.pool), 0, -1)[None]
        else: raise ValueError('unknown_policy_strategy')
        return utility, answerability, gain

    def choose(self, probabilities, excluded):
        utility, answerability, gain = self.utility(probabilities)
        if len(excluded) != len(utility): raise ValueError('exclusion_batch_mismatch')
        chosen, details = [], []
        for row, terms in enumerate(excluded):
            for term in terms:
                col = self.term_to_col.get(term)
                if col is not None: utility[row, col] = -np.inf
            if not np.isfinite(utility[row]).any():
                chosen.append(None); details.append(None); continue
            col = int(np.argmax(utility[row]))
            chosen.append(int(self.pool[col]))
            details.append({'estimated_recorded_answer_probability': float(answerability[row, col]),
                            'expected_answer_only_information': float(gain[row, col]),
                            'planning_utility': float(utility[row, col])})
        return chosen, details

    def __call__(self, scores, excluded):
        if not scores: return None
        # The production ranker returns the fixed class universe as log scores.
        if set(scores) != set(self.labels): raise ValueError('policy_class_universe_mismatch')
        z = np.array([scores[d] for d in self.labels], np.float64)
        if not np.isfinite(z).all(): raise ValueError('invalid_scores')
        p = np.exp(z-z.max()); p /= p.sum()
        terms, details = self.choose(p[None], [{self.code_to_term[t] for t in excluded if t in self.code_to_term}])
        if terms[0] is None: return None
        return {'tool': 'request_hpo_observation', 'concept_id': self.inverse[terms[0]],
                'selection_method': 'training_fitted_' + self.config['strategy'],
                'planning': details[0], 'calibrated_clinical_probability': False,
                'unknown_means_absent': False, 'policy_provenance': getattr(self, 'provenance', {})}


def load_selector(manifest_path):
    from pathlib import Path
    path = Path(manifest_path)
    manifest = load(path)
    if manifest['state'] != 'locked_research_policy': raise ValueError('policy_not_released')
    artifacts = {}
    for key, pin in manifest['artifacts'].items():
        p = path.parent / pin['path']
        if sha(p) != pin['sha256']: raise ValueError('policy_artifact_changed')
        artifacts[key] = p
    if sha(__file__) != manifest['artifacts']['policy_code']['sha256']:
        raise ValueError('runtime_policy_code_changed')
    with np.load(artifacts['tables'], allow_pickle=False) as data:
        tables = {key: data[key].copy() for key in data.files}
    vocab = load(artifacts['vocabulary'])
    selector = LearnedQuestions(tables, manifest['config'], vocab['labels'], {int(k): v for k, v in vocab['inverse'].items()})
    selector.provenance = {'manifest_sha256': sha(path), 'tables_sha256': manifest['artifacts']['tables']['sha256'],
                           'training_only_fit': True, 'clinical_efficacy_established': False}
    return selector


def current_selector():
    from common import ROOT, PROJECT
    registry = load(ROOT / 'agent_releases/CURRENT.json')
    if sha(registry['manifest']) != registry['sha256']:
        raise ValueError('current_policy_manifest_changed')
    manifest = load(registry['manifest'])
    base = load(PROJECT / 'model_releases/CURRENT.json')
    if sha(base['manifest']) != base['sha256'] or manifest['base_model_manifest_sha256'] != base['sha256']:
        raise ValueError('policy_base_model_mismatch')
    return load_selector(registry['manifest'])
