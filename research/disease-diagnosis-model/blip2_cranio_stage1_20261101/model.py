"""Official LAVIS Q-Former with a frozen cached craniofacial visual interface.

The three Stage-1 losses follow blip2_qformer.py. Clinical adaptations are
multi-image masks, identical-caption positives and conservative ITM negatives.
"""
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
from vendor.Qformer import BertConfig,BertLMHeadModel
from common import WORK,load,digest,save,now

class Stage1(nn.Module):
    def __init__(self):
        super().__init__()
        assets=WORK/'assets/bert-base-uncased'
        receipt=load(WORK/'bert_receipt.json')
        for name,info in receipt['files'].items():assert digest(assets/name)==info['sha256']
        self.tokenizer=BertTokenizer.from_pretrained(assets,local_files_only=True)
        self.tokenizer.add_special_tokens({'bos_token':'[DEC]'})
        cfg=BertConfig.from_pretrained(assets,local_files_only=True)
        cfg.encoder_width=768;cfg.add_cross_attention=True;cfg.cross_attention_freq=2;cfg.query_length=32
        self.Qformer,info=BertLMHeadModel.from_pretrained(assets,config=cfg,local_files_only=True,output_loading_info=True)
        # New cross-attention and query-specific FFNs are expected; all other BERT weights must load.
        assert not info.get('mismatched_keys') and not info.get('error_msgs')
        # Safetensors stores the tied decoder/embedding weight once. Verify the
        # alias rather than mistaking its omitted duplicate for random weights.
        assert self.Qformer.cls.predictions.decoder.weight.data_ptr()==self.Qformer.bert.embeddings.word_embeddings.weight.data_ptr()
        assert all(('crossattention' in k or '_query' in k or 'position_ids' in k or k.endswith('predictions.decoder.bias') or k=='cls.predictions.decoder.weight') for k in info['missing_keys'])
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state=self.Qformer.state_dict()
        for name,param in self.Qformer.named_parameters():
            if '_query' in name:param.data.copy_(state[name.replace('_query','')])
        self.query_tokens=nn.Parameter(torch.empty(1,32,768));nn.init.normal_(self.query_tokens,std=cfg.initializer_range)
        self.ln_vision=nn.LayerNorm(768)
        self.vision_proj=nn.Linear(768,256);self.text_proj=nn.Linear(768,256)
        self.itm_head=nn.Linear(768,2);self.temp=nn.Parameter(torch.tensor(.07))
        self.initialization_report=dict(missing_new_parameter_tensors=len(info['missing_keys']),
            unexpected_keys=info['unexpected_keys'],bert_revision=receipt['revision'],
            official_Qformer_sha256=digest(WORK/'vendor/Qformer.py'))

    def visual(self,tokens,mask):
        assert mask.bool().any(1).all()
        x=self.ln_vision(tokens.float())
        out=self.Qformer.bert(query_embeds=self.query_tokens.expand(len(x),-1,-1),
            encoder_hidden_states=x,encoder_attention_mask=mask.long(),use_cache=True,return_dict=True)
        return x,out,F.normalize(self.vision_proj(out.last_hidden_state).float(),dim=-1)

    def text(self,ids,mask):
        out=self.Qformer.bert(ids,attention_mask=mask.long(),return_dict=True)
        return F.normalize(self.text_proj(out.last_hidden_state[:,0]).float(),dim=-1)

    @staticmethod
    def allowed_negatives(present):
        x=present.float();overlap=x@x.T;count=x.sum(1)
        return overlap<torch.minimum(count[:,None],count[None,:])

    def forward(self,tokens,mask,ids,text_mask,groups,present):
        with torch.no_grad():self.temp.clamp_(.001,.5)
        image,out,iv=self.visual(tokens,mask)
        tv=self.text(ids,text_mask)
        with torch.autocast(device_type=tokens.device.type,enabled=False):
            sim=torch.einsum('bqd,nd->bnq',iv.float(),tv.float()).max(-1).values/self.temp.float()
        positives=groups[:,None].eq(groups[None,:]).float()
        target=.9*positives/positives.sum(1,keepdim=True)+.1/len(groups)
        itc=-(F.log_softmax(sim,dim=1)*target).sum(1).mean()/2
        itc=itc-(F.log_softmax(sim.T,dim=1)*target).sum(1).mean()/2
        allowed=self.allowed_negatives(present)
        valid=torch.where(allowed.any(1))[0]
        if len(valid):
            with torch.no_grad():
                a=F.softmax(sim.detach()[valid].masked_fill(~allowed[valid],float('-inf')),dim=1)
                b=F.softmax(sim.detach().T[valid].masked_fill(~allowed[valid],float('-inf')),dim=1)
                nt=torch.multinomial(a,1).squeeze(1);ni=torch.multinomial(b,1).squeeze(1)
            tx=torch.cat([ids,ids[valid],ids[nt]],0)
            tm=torch.cat([text_mask,text_mask[valid],text_mask[nt]],0)
            im=torch.cat([image,image[ni],image[valid]],0)
            imask=torch.cat([mask,mask[ni],mask[valid]],0)
            q=self.query_tokens.expand(len(tx),-1,-1)
            am=torch.cat([torch.ones(q.shape[:2],dtype=tm.dtype,device=tm.device),tm],1)
            paired=self.Qformer.bert(tx,query_embeds=q,attention_mask=am,
                encoder_hidden_states=im,encoder_attention_mask=imask.long(),return_dict=True)
            logits=self.itm_head(paired.last_hidden_state[:,:32]).mean(1).float()
            labels=torch.cat([torch.ones(len(ids),device=ids.device),torch.zeros(2*len(valid),device=ids.device)]).long()
            itm=F.cross_entropy(logits,labels)
        else:itm=itc*0
        decoder=ids.clone();decoder[:,0]=self.tokenizer.bos_token_id
        labels=decoder.masked_fill(~text_mask.bool(),-100)
        am=torch.cat([torch.ones((len(ids),32),dtype=text_mask.dtype,device=ids.device),text_mask],1)
        generated=self.Qformer(decoder,attention_mask=am,past_key_values=out.past_key_values,labels=labels,return_dict=True)
        itg=generated.loss.float()
        return itc+itm+itg,dict(itc=itc.detach(),itm=itm.detach(),itg=itg.detach(),negative_rows=len(valid))
