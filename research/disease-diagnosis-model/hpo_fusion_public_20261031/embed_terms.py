"""Local BiomedBERT encoding of public ontology term names, no patient input."""
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from common import *


def main():
    os.umask(0o077)
    if load(WORK / 'term_embedding_receipt.json').get('state') == 'complete':
        return
    torch.set_num_threads(4)
    texts = load(WORK / 'private/term_texts.json')['texts']
    tokenizer = AutoTokenizer.from_pretrained(TEXT, local_files_only=True)
    model = AutoModel.from_pretrained(TEXT, local_files_only=True, add_pooling_layer=False).cuda().eval().requires_grad_(False)
    values = []
    with torch.inference_mode():
        for start in range(0, len(texts), 256):
            batch = tokenizer(texts[start:start + 256], padding=True, truncation=True, max_length=64, return_tensors='pt').to('cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h = model(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1)
            pooled = (h.float() * mask).sum(1) / mask.sum(1).clamp_min(1)
            values.append(pooled.cpu().numpy())
            save(WORK / 'term_embedding_status.json', dict(state='encoding_public_ontology', completed=min(start + 256, len(texts)), total=len(texts), updated_at_utc=now()))
    result = np.concatenate(values)
    result[0] = 0
    assert np.isfinite(result).all()
    path = WORK / 'features/term_embeddings.npy'
    np.save(path, result)
    save(WORK / 'term_embedding_receipt.json', dict(state='complete', terms=len(result) - 1, width=result.shape[1], sha256=digest(path),
         model='local BiomedBERT abstracts/fulltext; frozen', patient_data_encoded=False, updated_at_utc=now()))


if __name__ == '__main__':
    main()
