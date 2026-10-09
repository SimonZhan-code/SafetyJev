"""Native model interface mechanics with a tiny offline backbone."""
import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/Open-Jev/tests'))
from types import SimpleNamespace
import torch
from torch import nn
from test_predictor_judge_model import Backbone,Processor


class ChunkProcessor(Processor):
    def save_pretrained(self,path):
        Path(path).mkdir()

    def __call__(self,text,images,**kwargs):
        self.images=images;b=len(text)
        return dict(input_ids=torch.tensor([[1]+[0]*25+[2,0]]*b),
            attention_mask=torch.tensor([[1]*27+[0]]*b),
            pixel_values=torch.tensor([sum(im.getpixel((0,0))) for im in images],dtype=torch.float32),
            image_grid_thw=torch.ones(len(images),3,dtype=torch.long),
            mm_token_type_ids=torch.zeros(b,28,dtype=torch.long))


class ChunkModelTests(unittest.TestCase):
    def setUp(self):
        from safetyjev.chunk_judge_model import ChunkJudgeModel
        torch.manual_seed(13)
        self.model=ChunkJudgeModel(Backbone(),ChunkProcessor(),nn.Linear(8,1),
            dict(max_length=256,history_frames=8,state_dim=16,max_actions=8,
                 input_contract='chunk_start_v2',camera_names=['overview','wrist'],lora_rank=0))
        self.x=dict(questions=['Will the jar tilt?'],observations={c:torch.zeros(1,8,3,8,8,dtype=torch.uint8) for c in ('overview','wrist')},
            history_robot_states=torch.ones(1,8,16),robot_state=torch.zeros(1,16),remaining_actions=torch.ones(1,8,8),executed_actions=torch.ones(1,8,8),
            action_mask=torch.ones(1,8,dtype=torch.bool),history_action_mask=torch.ones(1,8,dtype=torch.bool),
            history_mask=torch.ones(1,8,dtype=torch.bool),action_dt_s=torch.tensor([.05]),constraint_context=[dict(source='none',text='')])

    def test_history_changes_prediction_and_gets_gradients(self):
        x=self.x['executed_actions'].clone().requires_grad_()
        result=self.model(**{**self.x,'executed_actions':x})
        self.assertEqual(tuple(result.shape),(1,2))
        result.sum().backward()
        self.assertGreater(x.grad.abs().sum(),0)
        self.assertFalse(torch.allclose(result,self.model(**{**self.x,'executed_actions':x*2})))
        self.assertEqual(len(self.model.processor.images),16)
        text=str(self.model.processor.contents[-1]);self.assertNotIn('NEW violation',text)
        self.assertIn('Executed action',text);self.assertIn('Future committed',text)

    def test_empty_history_has_current_image_and_masked_nan_does_not_leak(self):
        self.x['history_action_mask'][:]=False
        self.x['history_mask'][:]=False;self.x['history_mask'][:,-1]=True
        a=self.model(**self.x)
        self.x['executed_actions'][:]=float('nan')
        self.x['history_robot_states'][:]=float('nan')
        torch.testing.assert_close(a,self.model(**self.x))
        self.assertEqual(len(self.model.processor.images),2)

    def test_worker_encoded_path_agrees_and_wrong_contract_rejected(self):
        from safetyjev.chunk_judge_model import encode_chunk_inputs,ChunkJudgeModel
        visual={k:self.x[k] for k in ('questions','observations','action_mask','action_dt_s','history_mask','history_action_mask','constraint_context')}
        encoded=encode_chunk_inputs(self.model.processor,self.model.model_config,**visual)
        numeric={k:self.x[k] for k in ('robot_state','remaining_actions','executed_actions','history_robot_states','action_mask','action_dt_s','history_mask','history_action_mask')}
        torch.testing.assert_close(self.model(**self.x),self.model(**numeric,encoded=encoded))
        self.x['action_mask'][0,-1]=False
        with self.assertRaises(ValueError):self.model(**self.x)
        with self.assertRaises(ValueError):ChunkJudgeModel(Backbone(),ChunkProcessor(),nn.Linear(8,1),dict(history_frames=3,max_actions=8))

    def test_checkpoint_roundtrip_restores_numeric_weights_and_dispatch(self):
        import copy,tempfile,json
        from unittest.mock import patch
        from safetyjev.chunk_judge_model import ChunkJudgeModel,load_judge
        self.model.model_config.update(method='native_qwen_chunk_noul',model_id='fixture',revision='fixture',dtype='float32',
            min_pixels=64,max_pixels=64,gradient_checkpointing=False)
        self.model.processor.save_pretrained=lambda path:Path(path).mkdir()
        expected=self.model(**self.x)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model';self.model.save(path)
            restored=ChunkJudgeModel(copy.deepcopy(self.model.backbone),ChunkProcessor(),nn.Linear(8,1),self.model.model_config.copy())
            with patch.object(ChunkJudgeModel,'from_pretrained',return_value=restored), \
                 patch('transformers.AutoProcessor.from_pretrained',return_value=ChunkProcessor()):
                actual=load_judge(path,device='cpu')
            torch.testing.assert_close(expected,actual(**self.x))
            self.assertEqual(json.loads((path/'model.json').read_text())['method'],'native_qwen_chunk_noul')

    def test_historical_state_is_used_and_masked_before_projection(self):
        states=self.x['history_robot_states'].clone().requires_grad_()
        score=self.model(**{**self.x,'history_robot_states':states})
        score.sum().backward()
        self.assertGreater(states.grad.abs().sum(),0)
        changed=states.detach().clone();changed[:,0,:]+=3
        self.assertFalse(torch.allclose(score,self.model(**{**self.x,'history_robot_states':changed})))
        self.x['history_action_mask'][:,:4]=False;self.x['history_mask'][:,:4]=False
        expected=self.model(**self.x)
        self.x['history_robot_states'][:,:4]=float('nan')
        torch.testing.assert_close(expected,self.model(**self.x))
        self.x['history_robot_states'][:,4,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):self.model(**self.x)

    def test_old_chunk_checkpoint_contract_is_rejected(self):
        from safetyjev.chunk_judge_model import model_class,ChunkJudgeModel
        with self.assertRaises(ValueError):model_class('chunk_start_v1')
        with self.assertRaises(ValueError):
            ChunkJudgeModel(Backbone(),ChunkProcessor(),nn.Linear(8,1),{**self.model.model_config,'input_contract':'chunk_start_v1'})

    def test_worker_collator_preserves_history_state_and_mask(self):
        from safetyjev.predictor_judge_dataset import PreparedPredictorJudgeCollator,collate_predictor_judge
        inputs={k:(v[0].numpy() if isinstance(v,torch.Tensor) else v[0])
                for k,v in self.x.items() if k!='observations'}
        inputs['observations']={c:v[0].numpy() for c,v in self.x['observations'].items()}
        item=dict(inputs=inputs,target=[0.,1.],sample_id='one',query_id='tilt',valid_steps=8)
        collator=PreparedPredictorJudgeCollator(self.model.model_config);collator.processor=ChunkProcessor()
        prepared=collator([item]);direct=collate_predictor_judge([item])
        self.assertEqual(tuple(prepared['inputs']['history_robot_states'].shape),(1,8,16))
        torch.testing.assert_close(prepared['inputs']['history_robot_states'],direct['inputs']['history_robot_states'])
        torch.testing.assert_close(self.model(**prepared['inputs']),self.model(**direct['inputs']))
        content=str(collator.processor.contents[-1])
        self.assertEqual(content.count('Post-action robot state'),8)
