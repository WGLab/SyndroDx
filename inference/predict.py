"""Local image/HPO inference. Inputs and individual outputs never enter stdout."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import sys

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
                  HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1')
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from networks import VisualEncoder, PhenotypeModel

ROOT = Path(__file__).resolve().parent


def load(p):
    return json.loads(Path(p).read_text())


def verify():
    pins = load(ROOT/'checksums.json')
    for relative, expected in pins.items():
        h = hashlib.sha256()
        with (ROOT/relative).open('rb') as f:
            for block in iter(lambda: f.read(8*1024*1024), b''):
                h.update(block)
        if h.hexdigest() != expected:
            raise ValueError('package_integrity_check_failed')


def read_cases(path):
    path = Path(path).resolve()
    rows, seen = [], set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) - {'case_id','images','hpo_present','hpo_absent','hpo_unknown'}:
            raise ValueError('unexpected_input_fields')
        key = row.get('case_id')
        if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', key) or key in seen:
            raise ValueError('invalid_or_duplicate_case_id')
        seen.add(key)
        for field in ['images','hpo_present','hpo_absent','hpo_unknown']:
            value = row.setdefault(field, [])
            if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                raise ValueError('expected_list_of_strings')
        for field in ['hpo_present','hpo_absent','hpo_unknown']:
            if not all(re.fullmatch(r'HP:\d{7}', x) for x in row[field]):
                raise ValueError('invalid_HPO_identifier')
        if set(row['hpo_present']) & set(row['hpo_absent']):
            raise ValueError('conflicting_HPO_polarity')
        if set(row['hpo_unknown']) & (set(row['hpo_present']) | set(row['hpo_absent'])):
            raise ValueError('conflicting_HPO_unknown_status')
        row['images'] = [str((path.parent/p).resolve()) for p in row['images']]
        if len(row['images']) != len(set(row['images'])):
            raise ValueError('duplicate_image_path_in_patient')
        if not all(Path(p).is_file() for p in row['images']):
            raise ValueError('image_file_missing')
        rows.append(row)
    if not rows:
        raise ValueError('empty_input_manifest')
    return rows


def image_tensor(path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError('image_decode_failed')
    rgb = cv2.cvtColor(cv2.resize(bgr, (112,112), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.transpose(2,0,1).copy()).float().div_(127.5).sub_(1)


class Predictor:
    def __init__(self, device='cpu', with_images=True):
        self.device = torch.device(device)
        self.config = load(ROOT/'config.json')
        self.diseases = load(ROOT/'assets/diseases.json')
        self.terms = {t:i+1 for i,t in enumerate(load(ROOT/'assets/term_index.json')['term_ids'])}
        self.visual = None
        if with_images:
            self.visual = VisualEncoder()
            self.visual.load_state_dict(torch.load(ROOT/'weights/visual.pt', map_location='cpu', weights_only=True), strict=True)
            self.visual.to(device).eval().requires_grad_(False)
        self.head = F.normalize(torch.load(ROOT/'weights/visual_head.pt', map_location='cpu', weights_only=True).float(), dim=1).to(device)
        self.models = {}
        for name, mode in [('hpo','hpo'),('fusion','fusion_selected')]:
            state = torch.load(ROOT/f'weights/{name}.pt', map_location='cpu', weights_only=True)
            model = PhenotypeModel(state['terms.weight'], self.config['classes'], self.config['public_classes'], mode)
            model.load_state_dict(state, strict=True)
            self.models[name] = model.to(device).eval().requires_grad_(False)

    def pack(self, row):
        pairs, excluded = [], []
        for key, state in [('hpo_present',0),('hpo_absent',1)]:
            for term in sorted(set(row.get(key, []))):
                if term in self.terms:
                    pairs.append((self.terms[term], state))
                else:
                    excluded.append(term)
        if len(pairs) > 128:
            raise ValueError('more_than_128_eligible_HPO_terms')
        ids, states = np.zeros(128, np.int64), np.zeros(128, np.int64)
        for k, (term, state) in enumerate(pairs):
            ids[k], states[k] = term, state
        return ids, states, {'eligible_terms':len(pairs), 'out_of_vocabulary_terms':excluded,
                            'unknown_terms_ignored':len(set(row.get('hpo_unknown', [])))}

    @torch.inference_mode()
    def features(self, row, batch_size):
        if not row['images']:
            return np.zeros(512, np.float32)
        values = []
        for start in range(0, len(row['images']), batch_size):
            x = torch.stack([image_tensor(p) for p in row['images'][start:start+batch_size]]).to(self.device)
            values.append(self.visual(x).cpu())
        feature = F.normalize(torch.cat(values).mean(0), dim=0)
        if not torch.isfinite(feature).all() or feature.norm() == 0:
            raise ValueError('invalid_visual_feature')
        return feature.numpy()

    @torch.inference_mode()
    def probabilities(self, feature, ids, states, mode):
        f = torch.as_tensor(feature[None], device=self.device, dtype=torch.float32)
        i = torch.as_tensor(ids[None], device=self.device, dtype=torch.long)
        s = torch.as_tensor(states[None], device=self.device, dtype=torch.long)
        face = bool(np.linalg.norm(feature) > 0) and mode != 'hpo_only'
        hpo = bool(ids.any()) and mode != 'face_only'
        if not face and not hpo:
            return None, 'insufficient_input'
        if face and not hpo:
            logits = 64 * F.normalize(f, dim=1) @ self.head.T / self.config['visual_probability_temperature']
            route = 'face_only'
        else:
            route = 'face_hpo' if face else 'hpo_only'
            logits, _ = self.models['fusion' if face else 'hpo'](f if face else torch.zeros_like(f), i, s)
        return logits.float().softmax(-1)[0].cpu().numpy(), route

    @torch.inference_mode()
    def hpo_feature(self, ids, states):
        if not ids.any():
            return np.zeros(256, np.float32)
        f = torch.zeros(1,512,device=self.device)
        i = torch.as_tensor(ids[None],device=self.device)
        s = torch.as_tensor(states[None],device=self.device)
        z = self.models['hpo'].represent(f,i,s)
        return F.normalize(z.float(),dim=1)[0].cpu().numpy()


def retrieval(query, gallery, top_k):
    # A visual embedding is required for both sides of this released score rule.
    if not np.linalg.norm(query['visual']):
        return []
    scored = []
    for item in gallery:
        if item['case_id'] == query['case_id'] or not np.linalg.norm(item['visual']):
            continue
        score = float(query['visual'] @ item['visual'])
        route = 'visual_only'
        if np.linalg.norm(query['hpo']) and np.linalg.norm(item['hpo']):
            score = .5*score + .5*float(query['hpo'] @ item['hpo'])
            route = 'visual_HPO_equal_weight'
        scored.append({'gallery_case_id':item['case_id'], 'similarity':score, 'route':route})
    return sorted(scored,key=lambda r:-r['similarity'])[:top_k]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='JSONL: one pseudonymous patient per row')
    p.add_argument('--output',required=True,help='New output directory; never overwrite an existing run')
    p.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    p.add_argument('--mode',default='all',choices=['all','face_hpo','face_only','hpo_only'])
    p.add_argument('--batch-size',type=int,default=8,help='Image batch size within each patient')
    p.add_argument('--top-k',type=int,default=10)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--gallery',help='Optional local gallery JSONL; no gallery is shipped')
    args = p.parse_args()
    if min(args.batch_size,args.top_k,args.threads) < 1:
        raise ValueError('positive_runtime_parameters_required')
    os.umask(0o077)
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('output_directory_already_exists')
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA_unavailable_use_CPU_or_install_CUDA_torch')
    verify()
    rows = read_cases(args.input)
    gallery_rows = read_cases(args.gallery) if args.gallery else []
    predictor = Predictor(args.device, any(r['images'] for r in rows+gallery_rows))
    # Validate all terms before any image inference or output writes.
    for row in rows+gallery_rows:
        predictor.pack(row)
    output.mkdir(parents=True, mode=0o700)
    modes = ['face_only','hpo_only','face_hpo'] if args.mode == 'all' else [args.mode]
    feature_rows, gallery, counts = [], [], collections.Counter()
    for row in gallery_rows:
        ids, states, _ = predictor.pack(row)
        gallery.append({'case_id':row['case_id'], 'visual':predictor.features(row,args.batch_size),
                        'hpo':predictor.hpo_feature(ids,states)})
    with (output/'predictions.jsonl').open('w') as stream:
        for row in rows:
            ids, states, quality = predictor.pack(row)
            v = predictor.features(row,args.batch_size)
            h = predictor.hpo_feature(ids,states)
            feature_rows.append((row['case_id'],v,h))
            record = {'case_id':row['case_id'], 'image_count':len(row['images']), 'HPO_input':quality, 'predictions':{}}
            for mode in modes:
                prob, route = predictor.probabilities(v,ids,states,mode)
                ranking = []
                if prob is not None:
                    for rank, index in enumerate(np.argsort(-prob,kind='stable')[:args.top_k],1):
                        ranking.append({'rank':rank, **predictor.diseases[index], 'model_score':float(prob[index])})
                record['predictions'][mode] = {'route':route, 'ranking':ranking,
                    'scores_in_diseases_json_order':None if prob is None else prob.tolist()}
                counts[mode+':'+route] += 1
            if args.gallery:
                record['retrieval'] = retrieval({'case_id':row['case_id'],'visual':v,'hpo':h},gallery,args.top_k)
            stream.write(json.dumps(record)+'\n'); stream.flush()
    np.savez_compressed(output/'features.npz', case_ids=np.array([r[0] for r in feature_rows]),
        visual=np.stack([r[1] for r in feature_rows]), hpo=np.stack([r[2] for r in feature_rows]))
    summary = {'state':'complete', 'release':predictor.config['release'], 'patients':len(rows),
        'gallery_patients':len(gallery), 'routes':dict(counts), 'precision':'FP32', 'device':args.device,
        'mode':args.mode, 'image_batch_size':args.batch_size, 'torch_version':torch.__version__,
        'numpy_version':np.__version__, 'opencv_version':cv2.__version__,
        'checkpoint_selection_on_BCH':False, 'network_access_used':False}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        # Decoder/framework exceptions may contain local paths; keep stderr aggregate-only.
        known = str(e) if type(e) is ValueError and re.fullmatch(r'[A-Za-z0-9_]+',str(e)) else type(e).__name__
        print('Inference failed: '+known, file=sys.stderr)
        sys.exit(1)
