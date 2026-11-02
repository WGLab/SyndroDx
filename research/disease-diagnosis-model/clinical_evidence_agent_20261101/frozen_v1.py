"""Offline frozen-v1 ranking adapter; genetics is optional and never imputed.

Input is an audited native512 embedding and eligible HPO observations, not an
image or a note. Patient features, terms and output rankings stay on HPC.
"""
import importlib.util
import numpy as np
import torch
from common import PROJECT, load, sha


class FrozenV1:
    def __init__(self, device='cpu'):
        registry = load(PROJECT / 'model_releases/CURRENT.json')
        self.manifest_path = __import__('pathlib').Path(registry['manifest'])
        if sha(self.manifest_path) != registry['sha256']:
            raise ValueError('release_manifest_hash_mismatch')
        self.manifest = load(self.manifest_path)
        self.release = self.manifest_path.parent
        if self.manifest['default_runtime_seed'] != 11:
            raise ValueError('unexpected_runtime_seed')
        self.pins = {}
        for name in ['code_model.py', 'HPO_term_embeddings', 'HPO_term_index',
                     'disease_vocabulary', 'visual_classifier_class_indices',
                     'evidence_input_receipt.json', 'evidence_protocol.json',
                     'hpo_11', 'reference_11', 'visual_encoder']:
            entry = self.manifest['artifacts'][name]
            p = self.release / entry['path']
            if sha(p) != entry['sha256']:
                raise ValueError('release_artifact_hash_mismatch')
            self.pins[name] = p
        self.device = torch.device(device)
        spec = importlib.util.spec_from_file_location('locked_craniofacial_model', self.pins['code_model.py'])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model_class = module.PhenotypeModel
        self.embeddings = torch.from_numpy(np.load(self.pins['HPO_term_embeddings']))
        term_ids = load(self.pins['HPO_term_index'])['term_ids']
        if not isinstance(term_ids, list) or len(set(term_ids)) != len(term_ids):
            raise ValueError('unexpected_hpo_vocabulary_format')
        self.terms = {term: i + 1 for i, term in enumerate(term_ids)}
        self.labels = load(self.pins['disease_vocabulary'])['labels']
        self.counts = load(self.pins['evidence_input_receipt.json'])['model_counts']
        self.models = {}
        for name, mode in [('hpo_11', 'hpo'), ('reference_11', 'fusion_selected')]:
            model = self.model_class(self.embeddings, len(self.labels), self.counts['public_training_diseases'], mode)
            state = torch.load(self.pins[name], map_location='cpu', weights_only=False)['model']
            if state['assertion.weight'].shape[0] == 2:
                w = state['assertion.weight']
                state['assertion.weight'] = torch.cat([w, w.mean(0, keepdim=True)], 0)
            model.load_state_dict(state, strict=True)
            self.models[name] = model.to(self.device).eval().requires_grad_(False)
            del state
        # Only the original visual classifier weights are needed; no image is read.
        state = torch.load(self.pins['visual_encoder'], map_location='cpu', weights_only=False, mmap=True)['model']
        indices = load(self.pins['visual_classifier_class_indices'])['indices']
        self.visual_head = torch.nn.functional.normalize(state['head.weight'][indices].float().clone(), dim=1).to(self.device)
        del state
        self.visual_temperature = load(self.pins['evidence_protocol.json'])['visual_probability_temperature']

    def pack(self, observations):
        by_term = {}
        for item in observations:
            if item.get('kind') != 'hpo':
                continue
            code, assertion = item.get('concept_id'), item.get('assertion')
            if assertion not in {'present', 'absent'} or code not in self.terms:
                continue
            if code in by_term and by_term[code] != assertion:
                raise ValueError('conflicting_hpo_polarity')
            by_term[code] = assertion
        ordered = [(self.terms[t], state) for assertion, state in [('present', 0), ('absent', 1)]
                   for t in sorted(by_term) if by_term[t] == assertion]
        if len(ordered) > 128:
            raise ValueError('hpo_input_exceeds_frozen_limit')
        ids, states = np.zeros(128, np.int64), np.zeros(128, np.int64)
        for i, (term, state) in enumerate(ordered):
            ids[i], states[i] = term, state
        return ids, states

    @torch.inference_mode()
    def probabilities(self, features, ids, states, mode='face_hpo'):
        if mode not in {'face_hpo', 'hpo_only', 'face_only'}:
            raise ValueError('unsupported_modality_mode')
        f = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        raw_ids, raw_states = np.asarray(ids), np.asarray(states)
        if not np.issubdtype(raw_ids.dtype, np.integer) or not np.issubdtype(raw_states.dtype, np.integer):
            raise ValueError('hpo_arrays_must_have_integer_dtype')
        i = torch.as_tensor(raw_ids, dtype=torch.long, device=self.device)
        s = torch.as_tensor(raw_states, dtype=torch.long, device=self.device)
        if f.ndim != 2 or f.shape[1] != 512 or i.shape != s.shape or i.ndim != 2 or i.shape != (len(f), 128):
            raise ValueError('invalid_adapter_input_shape')
        if not torch.isfinite(f).all() or i.min() < 0 or i.max() >= len(self.embeddings) or s.min() < 0 or s.max() > 2:
            raise ValueError('invalid_adapter_input_values')
        # State 2 is unresolved, not a positive or a negative finding.
        i = i.clone()
        i[s == 2] = 0
        s = s.clone()
        s[i == 0] = 0
        face = f.norm(dim=1) > 0
        hpo = i.ne(0).any(dim=1)
        if mode == 'hpo_only':
            face[:] = False
        elif mode == 'face_only':
            hpo[:] = False
        out = torch.zeros(len(f), len(self.labels), device=self.device)
        routes = ['insufficient_input'] * len(f)
        for name, selected in [('face_hpo', face & hpo), ('hpo_only', ~face & hpo), ('face_only', face & ~hpo)]:
            if not selected.any():
                continue
            if name == 'face_only':
                logits = 64 * torch.nn.functional.normalize(f[selected], dim=1) @ self.visual_head.T / self.visual_temperature
            else:
                model = self.models['reference_11' if name == 'face_hpo' else 'hpo_11']
                logits, _ = model(f[selected] if name == 'face_hpo' else torch.zeros_like(f[selected]), i[selected], s[selected])
            out[selected] = logits.float().softmax(-1)
            for row in torch.where(selected)[0].tolist():
                routes[row] = name
        return out.cpu().numpy(), routes

    def bind(self, native512_feature, mode='face_hpo'):
        """For engine.run: the engine must verify observations before this call."""
        feature = np.zeros(512, np.float32) if native512_feature is None else np.array(native512_feature, dtype=np.float32, copy=True)
        if feature.shape != (512,):
            raise ValueError('invalid_native_feature')
        def rank(accepted_observations):
            ids, states = self.pack(accepted_observations)
            probabilities, routes = self.probabilities(feature[None], ids[None], states[None], mode)
            if routes[0] == 'insufficient_input':
                return {}
            # Planning code softmaxes scores: use log-probabilities, not probabilities.
            return {label: float(np.log(max(float(p), 1e-30))) for label, p in zip(self.labels, probabilities[0])}
        return rank
