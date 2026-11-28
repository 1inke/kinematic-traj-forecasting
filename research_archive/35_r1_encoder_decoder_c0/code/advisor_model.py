"""Advisor arms for the sealed F16/h3 baseline."""
import math

import torch
from torch import nn
from torch.nn import functional as F

ARMS = ("B0", "R1", "R2", "R3", "S1", "S2", "S3", "S4", "G1", "G2", "G2-intra")
RANKED = ("S2", "S4", "S4-wide")
DEEP = ("R2", "R2-half", "S2", "S3", "S4", "S4-wide", "G2", "G2-intra")


def zero(linear):
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    return linear


class CrossLayer(nn.Module):
    def __init__(self, self_attention=None):
        super().__init__()
        self.causal = self_attention == "causal"
        self.self_attn = nn.MultiheadAttention(64,4,batch_first=True) if self_attention else None
        self.self_norm = nn.LayerNorm(64) if self_attention else None
        self.cross_norm = nn.LayerNorm(64)
        self.cross = nn.MultiheadAttention(64,4,batch_first=True)
        self.ff_norm = nn.LayerNorm(64)
        self.ff = nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,64))

    def forward(self,q,memory,padding):
        if self.self_attn is not None:
            h = self.self_norm(q)
            mask = torch.ones(q.shape[1],q.shape[1],dtype=torch.bool,device=q.device).triu(1) if self.causal else None
            q = q + self.self_attn(h,h,h,attn_mask=mask,need_weights=False)[0]
        q = q + self.cross(self.cross_norm(q),memory,memory,key_padding_mask=padding,need_weights=False)[0]
        return q + self.ff(self.ff_norm(q))


class AdvisorDecoder(nn.Module):
    def __init__(self, h3, arm):
        super().__init__()
        self.base = h3
        self.arm = arm
        self.valid = None
        self.aux = None
        self.order = None
        self.group_weights = None
        if arm in ("G1","G2","G2-intra"):
            template = h3.anchor_templates.detach()
            fraction = template.new_tensor(h3.anchor_steps).reshape(1,-1,1)/h3.pred_len
            off = template.new_tensor([0.,.08])
            anchors = torch.stack((template,.85*template+fraction*off,1.15*template-fraction*off),1)
            embed = h3.mode_embedding.detach()
            self.base.anchor_templates = nn.Parameter(anchors.reshape(60,3,2))
            others = .02*torch.randn(20,2,64)
            self.base.mode_embedding = nn.Parameter(torch.cat((embed[:,None],others+embed[:,None]),1).reshape(60,64))
            self.base.num_modes = 60
        if arm != "G1":
            self.memory_proj = nn.Linear((h3.context[0].normalized_shape[0]-7)//2,64)
            intra = "causal" if arm in RANKED else ("full" if arm in ("S3","G2") else None)
            self.layer = CrossLayer(intra)
        if arm not in ("R3","G1"):
            self.offset = zero(nn.Linear(64,6))
            self.score = nn.Sequential(nn.LayerNorm(70),nn.Linear(70,64),nn.GELU(),zero(nn.Linear(64,1)))
        if arm in ("R1","R2","R2-half","S2","S3","S4","S4-wide"):
            self.geometric = nn.Linear(6,64)
        if arm in RANKED:
            self.rank = nn.Parameter(torch.empty(20,64))
            nn.init.normal_(self.rank,std=.02)
        if arm in ("G2","G2-intra"):
            self.member_norm = nn.LayerNorm(64)
            self.member_attention = nn.MultiheadAttention(64,4,batch_first=True)
            self.pool_query = nn.Parameter(torch.empty(64))
            nn.init.normal_(self.pool_query,std=.02)

    def memory(self,encoded):
        own, agents = encoded[:,0],encoded[:,:,-1]
        n = encoded.size(0)
        counts = self.valid
        if counts is None:
            raise ValueError("Missing valid-agent counts for cross attention")
        pad = torch.arange(agents.shape[1],device=encoded.device)[None] >= counts.reshape(n,1)
        padding = torch.cat((torch.zeros_like(own[:,:,0],dtype=torch.bool),pad),1)
        return self.memory_proj(torch.cat((own,agents),1)),padding

    def proposal(self,h,scale):
        b = self.base
        delta = .75*torch.tanh(b.proposal_delta(h)).reshape(h.size(0),h.size(1),3,2)
        return scale[:,None,None]*(b.anchor_templates[None]+delta)

    def trajectory(self,anchors,last):
        return last[:,None,None]+torch.einsum("tp,bkpc->bktc",self.base.interpolation,anchors)

    def forward(self,encoded,past):
        b = self.base
        enc, foc = encoded[:,0],past[:,:1]
        c = b.context(torch.cat((enc[:,-1],enc.mean(1),b.motion_features(foc)[:,0]),-1))
        q = b.mode_mlp(c[:,None]+b.mode_embedding[None])
        scale = F.softplus(b.proposal_scale(c))+.25
        last = foc[:,0,-1]
        self.aux, self.order, self.group_weights = None,None,None
        if self.arm == "R3":
            q = self.layer(q,*self.memory(encoded))
        a = self.proposal(q,scale)
        paths = self.trajectory(a,last)
        logits = b.score_head(torch.cat((q,a.flatten(2)),-1)).squeeze(-1)
        if self.arm in ("R3","G1"):
            return paths[:,None],logits[:,None]
        self.aux = paths[:,None],logits[:,None]
        memory,padding = self.memory(encoded)
        if self.arm in ("G2","G2-intra"):
            grouped = q.reshape(-1,3,64)
            k = self.member_norm(grouped)
            m = (grouped+self.member_attention(k,k,k,need_weights=False)[0]).reshape(-1,20,3,64)
            w = torch.softmax(torch.einsum("bgmd,d->bgm",m,self.pool_query)/math.sqrt(64),dim=-1)
            self.group_weights = w
            q = (m*w[...,None]).sum(2)
            a = (a.reshape(-1,20,3,3,2)*w[...,None,None]).sum(2)
            logits = None
        else:
            q = q+self.geometric((a/scale[:,None,None]).flatten(2))
        if self.arm in RANKED:
            self.order = torch.argsort(logits.detach(),dim=1,descending=True,stable=True)
            gather = self.order[...,None].expand_as(q)
            q = q.gather(1,gather)+self.rank[None]
        h = self.layer(q,memory,padding)
        if self.order is not None:
            inverse = self.order.argsort(1)
            h = h.gather(1,inverse[...,None].expand_as(h))
        shift = .30*scale[:,None,None]*torch.tanh(self.offset(h)).reshape(h.size(0),h.size(1),3,2)
        a = a+shift
        score = self.score(torch.cat((h,a.flatten(2)),-1)).squeeze(2)
        if logits is not None:
            score = score+logits
        return self.trajectory(a,last)[:,None],score[:,None]


def set_valid(model,args,kwargs):
    model.decoder.valid = kwargs.get("num_valid")


def attach(model,arm):
    if arm not in ("B0","S1"):
        device = next(model.parameters()).device
        with torch.random.fork_rng(devices=[]):
            model.decoder = AdvisorDecoder(model.decoder.cpu(),arm)
        model.decoder.to(device)
        model.register_forward_pre_hook(set_valid,with_kwargs=True)


def emta_indices(dist,order,threshold=.5):
    steps = torch.arange(1,dist.size(-1)+1,device=dist.device,dtype=dist.dtype)
    match = (dist <= steps*(threshold/dist.size(-1))).all(-1)
    sorted_match = match.gather(1,order)
    first = sorted_match.to(torch.int64).argmax(1)
    fallback = (dist.mean(2)+.25*dist[:,:,-1]).argmin(1)
    return torch.where(match.any(1),order.gather(1,first[:,None])[:,0],fallback),match.any(1)


def advisor_loss(core, model, arm, pred, target, logits, criterion):
    if arm in ("B0","S1","R1","R3","G1"):
        return (*core.loss_value(pred,target,logits,criterion),None)
    decoder = model.decoder
    dist = torch.linalg.vector_norm(pred-target.unsqueeze(-3),dim=-1)
    q = dist.mean(-1)+.25*dist[...,-1]
    match = None
    if arm in ("S4","S4-wide"):
        winner,match = emta_indices(dist[:,0],decoder.order,1. if arm=="S4-wide" else .5)
        winner = winner[:,None]
        index = winner[...,None,None,None].expand(*winner.shape,1,pred.shape[-2],2)
        select = pred.gather(-3,index).squeeze(-3)
        regression = criterion.regression(select,target)
        endpoint = F.smooth_l1_loss(select[...,-1,:],target[...,-1,:])
        short = criterion.short_horizon_steps
        early = F.smooth_l1_loss(select[...,:short,:],target[...,:short,:])
        binary = F.one_hot(winner, pred.shape[-3]).to(logits.dtype)
        raw = F.binary_cross_entropy_with_logits(logits,binary,reduction="none")
        focal = ((1-torch.exp(-raw)).square()*raw*(.25*binary+.75*(1-binary))).mean()
        last = pred[...,-1,:]
        distances = torch.cdist(last,last)
        upper = torch.ones_like(distances,dtype=torch.bool).triu(1)
        diversity = F.relu(criterion.diversity_margin-distances[upper]).mean()
        cover = (torch.softmax(-q.detach()/.10,dim=-1)*q).sum(-1).mean()
        loss = (regression+criterion.classification_weight*focal+criterion.endpoint_weight*endpoint+
                criterion.short_horizon_weight*early+criterion.diversity_weight*diversity+.20*cover)
        detail = dict(regression=regression.detach(),classification=focal.detach())
    else:
        loss,detail = core.loss_value(pred,target,logits,criterion)
    if arm in DEEP:
        a,p = decoder.aux
        extra,_ = core.loss_value(a,target,p,criterion)
        loss = loss+(.5 if arm=="R2-half" else 1.)*extra
    if arm in ("S2","S3","S4","S4-wide"):
        pair = q.unsqueeze(-1)+.20 < q.unsqueeze(-2)
        penalty = F.relu(1-(logits.unsqueeze(-1)-logits.unsqueeze(-2)))
        loss = loss+.10*(penalty*pair).sum()/pair.sum().clamp_min(1)
    return loss,detail,match

