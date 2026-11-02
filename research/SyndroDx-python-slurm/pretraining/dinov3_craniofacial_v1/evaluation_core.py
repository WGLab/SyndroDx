"""Patient-level metrics; works on synthetic arrays in tests, private arrays on HPC."""
import collections
import re
import numpy as np


def normalized(x):
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('nonfinite_embeddings')
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms < 1e-10):
        raise ValueError('zero_embeddings')
    return x / norms


def aggregate_patients(records, features, split):
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for row, vector in zip(records, features):
        if row['split'] == split:
            grouped[row['group_id']][row['view']].append(vector)
    ids = sorted(grouped)
    # Equal weight per view, then equal weight per patient in every endpoint.
    means = [np.mean([normalized(np.mean(v, axis=0, keepdims=True))[0]
                      for v in grouped[g].values()], axis=0) for g in ids]
    return ids, normalized(means) if means else np.empty((0, features.shape[1]), np.float32)


def retrieval(query, gallery, query_ids, gallery_ids, query_labels, gallery_labels, ks=(1, 5, 10)):
    query, gallery = normalized(query), normalized(gallery)
    if not len(query) or not len(gallery):
        raise ValueError('empty_retrieval_partition')
    if len(set(query_ids)) != len(query_ids) or len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError('patient_aggregation_required')
    hits = {k: np.zeros(len(query), dtype=bool) for k in ks}
    covered = np.zeros(len(query), dtype=bool)
    all_labels = set().union(*gallery_labels)
    for i in range(0, len(query), 128):
        scores = query[i:i + 128] @ gallery.T
        for j in range(len(scores)):
            qi = i + j
            eligible = np.array([g != query_ids[qi] for g in gallery_ids])
            candidates = np.flatnonzero(eligible)
            covered[qi] = bool(query_labels[qi] & all_labels)
            count = min(max(ks), len(candidates))
            if not count:
                continue
            best = candidates[np.argsort(-scores[j, candidates], kind='stable')[:count]]
            for k in ks:
                hits[k][qi] = any(query_labels[qi] & gallery_labels[x] for x in best[:k])
    return hits, covered


def macro_score(values, labels):
    classes = collections.defaultdict(list)
    for value, terms in zip(values, labels):
        for label in terms:
            classes[label].append(float(value))
    return float(np.mean([np.mean(v) for v in classes.values()])) if classes else None


def paired_interval(first, second, labels, repetitions=2000, seed=20261015):
    first, second = np.asarray(first), np.asarray(second)
    if len(first) != len(second) or len(first) != len(labels) or not len(first):
        raise ValueError('paired_patient_results_required')
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(first), size=len(first))
        terms = [labels[i] for i in indices]
        differences.append(macro_score(second[indices], terms) - macro_score(first[indices], terms))
    return dict(delta_macro=macro_score(second, labels) - macro_score(first, labels),
                paired_patient_bootstrap_95_interval=np.quantile(differences, [.025, .975]).tolist(),
                repetitions=repetitions, seed=seed)


def canonical_labels(row):
    mondo = row.get('mondo_id') or row.get('MONDO_ID') or row.get('mondo') or ''
    terms = {'MONDO:' + value.zfill(7) for value in re.findall(r'MONDO[:_](\d+)', mondo, flags=re.I)}
    if terms:
        return terms
    diagnosis = row.get('diagnosis') or row.get('diagnosis_name') or ''
    return {v.strip().casefold() for v in re.split(r'[|;]', diagnosis) if v.strip()}


def collapse_statistics(features, seed=20261015):
    x = normalized(features)
    if len(x) > 512:
        x = x[np.random.default_rng(seed).choice(len(x), 512, replace=False)]
    centered = x - x.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(centered @ centered.T / max(1, len(x) - 1))
    positive = np.maximum(eigenvalues, 0)
    probability = positive / max(float(positive.sum()), 1e-12)
    positive_probability = probability[probability > 0]
    return dict(finite=True, sampled_patients=len(x),
                effective_rank=float(np.exp(-np.sum(positive_probability * np.log(positive_probability)))) if positive.sum() > 1e-12 else 0.,
                total_coordinate_variance=float(positive.sum()),
                mean_off_diagonal_cosine=float(((x.sum(axis=0) ** 2).sum() - len(x)) / max(1, len(x) * (len(x) - 1))))


def ridge_probe(train, validation, test, train_y, validation_y, test_y):
    """Frozen encoder; ridge alpha chosen on validation, never on test."""
    classes = sorted(set(train_y))
    if len(classes) < 2 or not len(validation_y) or not (set(validation_y) & set(classes)):
        return dict(state='not_evaluable', reason='two_train_classes_and_labeled_validation_required')
    vocabulary = {label: i for i, label in enumerate(classes)}
    mean, std = train.mean(axis=0), train.std(axis=0)
    std = np.maximum(std, .01)
    x = (train - mean) / std
    val = (validation - mean) / std
    tst = (test - mean) / std
    x = np.concatenate([x, np.ones((len(x), 1), np.float32)], axis=1)
    val = np.concatenate([val, np.ones((len(val), 1), np.float32)], axis=1)
    tst = np.concatenate([tst, np.ones((len(tst), 1), np.float32)], axis=1)
    counts = collections.Counter(train_y)
    weights = np.array([1 / counts[y] for y in train_y], dtype=np.float32)
    weights *= len(weights) / weights.sum()
    target = np.zeros((len(x), len(classes)), np.float32)
    target[np.arange(len(x)), [vocabulary[y] for y in train_y]] = 1
    gram = x.T @ (weights[:, None] * x)
    rhs = x.T @ (weights[:, None] * target)
    eigenvalues, basis = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0)
    rhs = basis.T @ rhs
    projected_val = val @ basis
    best = None
    for alpha in [.01, .1, 1., 10., 100.]:
        coefficient = rhs / (eigenvalues[:, None] + alpha)
        predictions = np.argmax(projected_val @ coefficient, axis=1)
        success = np.array([classes[i] == y for i, y in zip(predictions, validation_y)])
        score = macro_score(success, [{y} for y in validation_y])
        if best is None or score > best[0]:
            best = (score, alpha, coefficient)
    prediction = np.argmax((tst @ basis) @ best[2], axis=1)
    success = np.array([classes[i] == y for i, y in zip(prediction, test_y)])
    return dict(state='complete', alpha_selected_on_validation=best[1],
                validation_macro=best[0], test_micro=float(success.mean()) if len(success) else None,
                test_macro=macro_score(success, [{y} for y in test_y]),
                test_covered_fraction=float(np.mean([y in vocabulary for y in test_y])) if len(test_y) else None,
                test_success=success)
