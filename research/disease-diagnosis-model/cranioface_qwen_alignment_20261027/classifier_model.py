"""Direct 216-way diagnosis prediction from the decoder's last valid token."""
import torch
from torch import nn
def classification_head(width=5120,classes=216):
    return nn.Sequential(nn.LayerNorm(width),nn.Dropout(.2),nn.Linear(width,classes))
def pooled_hidden(llm,visual,tensors):
    ids,mask,labels,offset=tensors
    embeddings=llm.get_input_embeddings()(ids)
    embeddings=torch.cat((embeddings[:,:offset],visual.to(embeddings.dtype),embeddings[:,offset+visual.shape[1]:]),dim=1)
    positions=mask.long().cumsum(-1)-1;positions.masked_fill_(mask==0,0)
    # Use the text backbone directly: no vocabulary logits or disease-name tokens.
    core=llm.get_base_model() if hasattr(llm,'get_base_model') else llm
    hidden=core.model.language_model(inputs_embeds=embeddings,attention_mask=mask,
        position_ids=positions.unsqueeze(0).expand(3,-1,-1),use_cache=False).last_hidden_state
    last=mask.sum(1)-1
    return hidden[torch.arange(len(ids),device=ids.device),last].float()
