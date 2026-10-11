"""SafetyJev chunk-boundary architecture over the unchanged native Qwen fork."""
import json
from pathlib import Path
import torch
from jev.predictor_judge_model import PredictorJudgeModel
from jev.visual_model import last_valid_hidden
from .chunk_judge import CHUNK_CONTRACT


def _masks(action_mask,history_mask,history_action_mask):
    masks=[m.cpu() for m in (action_mask,history_mask,history_action_mask)]
    am,hm,pm=masks
    if any(m.dtype!=torch.bool or m.ndim!=2 or m.shape[1]!=8 for m in masks) or any(m.shape!=am.shape for m in masks):
        raise ValueError('Expected B,8 boolean masks')
    if not am.all():raise ValueError('Chunk requires eight committed future actions')
    for m in (hm,pm):
        if not torch.equal(m,torch.arange(8)[None]>=8-m.sum(-1)[:,None]):raise ValueError('History must be an adjacent suffix')
    if not hm[:,-1].all() or (pm&~hm).any():raise ValueError('Current image and action-result alignment required')
    # Initial/reset observation may have no preceding executed action.
    if ((hm.sum(-1)-pm.sum(-1))>1).any():raise ValueError('Unpaired history observations')
    return hm,pm


def encode_chunk_inputs(processor,model_config,questions,observations,action_mask,action_dt_s,
                        history_mask,history_action_mask,constraint_context=None):
    from PIL import Image
    hm,pm=_masks(action_mask,history_mask,history_action_mask);b=len(hm)
    cameras=model_config.get('camera_names',['overview','wrist'])
    if len(questions)!=b or any(not isinstance(q,str) or not q.strip() for q in questions):raise ValueError('Nonempty questions required')
    if set(observations)!=set(cameras):raise ValueError('Missing camera')
    if any(v.dtype!=torch.uint8 or v.ndim!=5 or v.shape[:3]!=(b,8,3) for v in observations.values()):raise ValueError('Expected B,8,3,H,W images')
    dt=action_dt_s.float().cpu().reshape(-1)
    if dt.shape!=(b,) or not torch.isfinite(dt).all() or (dt<=0).any():raise ValueError('Invalid action interval')
    contexts=constraint_context or [dict(source='none',text='') for _ in range(b)]
    if len(contexts)!=b or any(c!={'source':'none','text':''} for c in contexts):raise ValueError('Chunk model uses only explicit observed modalities, no oracle context')
    marker=processor.tokenizer.pad_token
    if not marker or any(marker in q for q in questions):raise ValueError('Reserved numeric marker')
    images=[];prompts=[]
    for i,q in enumerate(questions):
        content=[dict(type='text',text=q+'\nPredict whether the queried violation occurs during the next eight committed actions.\n')]
        for j in range(8):
            content.append(dict(type='text',text=f'Executed action at relative step {j-8}: {marker}\n'))
            content.append(dict(type='text',text=f'Post-action robot state at relative step {j-7}: {marker}\n'))
            if hm[i,j]:
                for camera in cameras:
                    im=Image.fromarray(observations[camera][i,j].detach().cpu().permute(1,2,0).numpy())
                    images.append(im)
                    content.extend([dict(type='text',text=f'{camera}, observation at relative step {j-7} (action result; initial view if no action):'),dict(type='image',image=im)])
        content.append(dict(type='text',text=f'Current robot state: {marker}\nFuture committed actions, dt={float(dt[i]):.6g}s: '+' '.join([marker]*8)+'\nAnswer yes or no.'))
        prompts.append(processor.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True,enable_thinking=False))
    encoded=dict(processor(text=prompts,images=images,padding=True,return_tensors='pt'))
    if 'pixel_values' not in encoded:raise ValueError('Native visual inputs required')
    if encoded['input_ids'].shape[1]>model_config['max_length']:raise ValueError('Chunk inputs exceed token budget; do not truncate history silently')
    return encoded


class ChunkJudgeModel(PredictorJudgeModel):
    def __init__(self,backbone,processor,head,model_config):
        if model_config.get('input_contract')!=CHUNK_CONTRACT or model_config.get('history_frames')!=8 or model_config.get('max_actions')!=8:
            raise ValueError('Expected chunk_start_v2 with eight history and future slots')
        super().__init__(backbone,processor,head,model_config)

    @classmethod
    def from_pretrained(cls,model_id,revision,*,history_frames=8,state_dim=16,max_actions=8,input_contract=CHUNK_CONTRACT,**kwargs):
        if input_contract!=CHUNK_CONTRACT or history_frames!=8 or max_actions!=8:raise ValueError('Chunk input contract mismatch')
        from jev.visual_model import VisualDecisionModel
        base=VisualDecisionModel.from_pretrained(model_id,revision,**kwargs)
        cfg={**base.model_config,'method':'native_qwen_chunk_noul','input_contract':CHUNK_CONTRACT,
             'history_frames':8,'state_dim':state_dim,'max_actions':8}
        return cls(base.backbone,base.processor,base.head,cfg)

    @classmethod
    def load(cls,output,*,device='cuda:0',trainable=False):
        from transformers import AutoProcessor
        path=Path(output);cfg=json.loads((path/'model.json').read_text())
        if cfg.get('input_contract')!=CHUNK_CONTRACT or cfg.get('method')!='native_qwen_chunk_noul':
            raise ValueError('Not a chunk-boundary checkpoint')
        model=cls.from_pretrained(cfg['model_id'],cfg['revision'],device=device,dtype=cfg['dtype'],lora_rank=0,
            max_length=cfg['max_length'],min_pixels=cfg['min_pixels'],max_pixels=cfg['max_pixels'],gradient_checkpointing=False,
            history_frames=cfg['history_frames'],state_dim=cfg['state_dim'],max_actions=cfg['max_actions'])
        model.processor=AutoProcessor.from_pretrained(path/'processor');model.processor.tokenizer.padding_side='right'
        if cfg['lora_rank']:
            from peft import PeftModel
            model.backbone.language_model=PeftModel.from_pretrained(model.backbone.language_model,path/'adapter',is_trainable=trainable)
        model.head.load_state_dict(torch.load(path/'head.pt',map_location=device,weights_only=True))
        payload=torch.load(path/'predictor_judge.pt',map_location=device,weights_only=True)
        for name in ('state_projection','action_projection'):getattr(model,name).load_state_dict(payload[name])
        model.set_normalization(**payload['normalization']);model.model_config=cfg;model.set_temperature(cfg.get('temperature',1.))
        if trainable:model._gradient_checkpointing(cfg['gradient_checkpointing']);return model.train()
        return model.eval()

    def forward(self,questions=None,observations=None,robot_state=None,remaining_actions=None,
                action_mask=None,action_dt_s=None,history_mask=None,constraint_context=None,
                executed_actions=None,history_action_mask=None,history_robot_states=None,*,encoded=None):
        _masks(action_mask,history_mask,history_action_mask)
        device=self.head.weight.device;b=len(robot_state)
        state=robot_state.to(device=device,dtype=torch.float32)
        past=executed_actions.to(device=device,dtype=torch.float32)
        if history_robot_states is None:raise ValueError('Historical robot states required by chunk_start_v2')
        states=history_robot_states.to(device=device,dtype=torch.float32)
        future=remaining_actions.to(device=device,dtype=torch.float32)
        pm=history_action_mask.to(device);dt=action_dt_s.to(device=device,dtype=torch.float32).reshape(-1)
        if state.shape!=(b,self.model_config['state_dim']) or past.shape!=(b,8,8) or future.shape!=(b,8,8) or states.shape!=(b,8,self.model_config['state_dim']):raise ValueError('Invalid numeric shape')
        if dt.shape!=(b,) or not torch.isfinite(dt).all() or (dt<=0).any():raise ValueError('Invalid action interval')
        if not torch.isfinite(state).all() or not torch.isfinite(future).all() or not torch.isfinite(past[pm]).all() or not torch.isfinite(states[pm]).all():raise ValueError('Nonfinite valid input')
        if encoded is None:
            encoded=encode_chunk_inputs(self.processor,self.model_config,questions,observations,action_mask,action_dt_s,
                                         history_mask,history_action_mask,constraint_context)
        elif questions is not None or observations is not None or constraint_context is not None:
            raise ValueError('Use raw or prepared inputs, not both')
        encoded={k:v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in encoded.items()}
        ids=encoded.pop('input_ids');attention=encoded['attention_mask'].clone()
        slots=(ids==self.processor.tokenizer.pad_token_id)&attention.bool()
        if not torch.all(slots.sum(-1)==25):raise ValueError('Expected twenty-five numeric slots')
        embeddings=self.backbone.get_input_embeddings()(ids).clone()
        clean=torch.where(pm[:,:,None],past,torch.zeros_like(past))
        normalized=(clean-self.action_mean)/self.action_std
        normalized=torch.where(pm[:,:,None],normalized,torch.zeros_like(normalized))
        def project(actions,times):
            features=torch.cat([actions,times[None,:,None]*dt[:,None,None],dt[:,None,None].expand(-1,8,1)],-1)
            return self.action_projection(features)
        past_tokens=project(normalized,torch.arange(-8,0,device=device))
        past_tokens=torch.where(pm[:,:,None],past_tokens,torch.zeros_like(past_tokens))
        future_tokens=project((future-self.action_mean)/self.action_std,torch.arange(8,device=device))
        state_tokens=self.state_projection((state-self.state_mean)/self.state_std)
        clean_states=torch.where(pm[:,:,None],states,torch.zeros_like(states))
        normalized_states=(clean_states-self.state_mean)/self.state_std
        normalized_states=torch.where(pm[:,:,None],normalized_states,torch.zeros_like(normalized_states))
        history_state_tokens=self.state_projection(normalized_states)
        history_state_tokens=torch.where(pm[:,:,None],history_state_tokens,torch.zeros_like(history_state_tokens))
        for i in range(b):
            ix=slots[i].nonzero().flatten()
            embeddings[i,ix[:16:2]]=past_tokens[i].to(embeddings.dtype)
            embeddings[i,ix[1:16:2]]=history_state_tokens[i].to(embeddings.dtype)
            attention[i,ix[:16]]=pm[i].repeat_interleave(2).to(attention.dtype)
            embeddings[i,ix[16]]=state_tokens[i].to(embeddings.dtype)
            embeddings[i,ix[17:]]=future_tokens[i].to(embeddings.dtype)
        if ids.shape[1]>self.model_config['max_length']:raise ValueError('Chunk input exceeds token budget')
        pos,_=self.backbone.get_rope_index(ids,image_grid_thw=encoded.get('image_grid_thw'),
            attention_mask=attention,mm_token_type_ids=encoded.get('mm_token_type_ids'))
        encoded.update(inputs_embeds=embeddings,attention_mask=attention,position_ids=pos)
        self.last_input_tokens=int(attention.sum());self.last_sequence_length=ids.shape[1]
        output=self.backbone(**encoded,use_cache=False,return_dict=True)
        score=self.head(last_valid_hidden(output.last_hidden_state,attention).float()).squeeze(-1)
        return torch.stack([torch.zeros_like(score),score],-1)


def model_class(input_contract='per_step_v1'):
    if input_contract==CHUNK_CONTRACT:return ChunkJudgeModel
    if input_contract=='per_step_v1':return PredictorJudgeModel
    raise ValueError('Unknown predictor input contract')


def load_judge(output,**kwargs):
    cfg=json.loads((Path(output)/'model.json').read_text())
    return model_class(cfg.get('input_contract','per_step_v1')).load(output,**kwargs)
