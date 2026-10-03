"""Deterministic train-only sampling by label, constraint and first-rejection event."""
from collections import Counter
from array import array
import json
import math
import random


class SamplingRows:
    """Read metadata sequentially or by the dataset's compact offsets; never load images."""
    def __init__(self, dataset):self.dataset=dataset
    def __len__(self):return len(self.dataset)
    def __iter__(self):
        with self.dataset.path.open() as stream:
            for line in stream:yield json.loads(line)
    def __getitem__(self,index):return self.dataset.record(index)


def draw_report(rows, order, *, available_labels=None):
    labels=Counter();breakdowns={k:{} for k in ['family','policy','checkpoint_id','constraint_id','valid_steps','episode_safety','negative_stratum']}
    events=set();unique=set();episodes=set()
    for i in order:
        r=rows[i];label='positive' if r['target'][1] else 'negative'
        labels[label]+=1;unique.add(r['id']);episodes.add(r['episode_id'])
        if label=='positive':events.add((r['episode_id'],r['constraint_id'],r['first_violation_step']))
        for field in breakdowns:
            value=str(r.get(field,'unknown'));breakdowns[field].setdefault(value,Counter())[label]+=1
    total=sum(labels.values())
    if available_labels is None:available_labels={int(r['target'][1]) for r in rows}
    return {'draws':total,'labels':dict(labels),'positive_fraction':labels['positive']/total if total else None,
            'unique_samples':len(unique),'repeated_draws':total-len(unique),
            'unique_episodes':len(episodes),'distinct_positive_events':len(events),'by':breakdowns,
            'missing_labels':[name for name,bit in [('positive',1),('negative',0)] if bit not in available_labels]}


def sampling_pools(rows):
    cached=getattr(rows,'_sampling_pools',None)
    if cached is not None:return cached
    pools={0:{},1:{}};positives=0
    for i,r in enumerate(rows):
        if r.get('split')!='train':raise ValueError('Sampler accepts train rows only')
        if r.get('target') not in ([1.,0.],[0.,1.]):raise ValueError('Only binary eligible targets')
        positives+=int(r['target'][1])
        y=int(r['target'][1]);category=(r['family'],r['constraint_id']);node=pools[y].setdefault(category,{})
        if y:
            event=r['first_violation_step']
            if type(event) is not int:raise ValueError('Positive samples need a witnessed event')
            node.setdefault((r['episode_id'],event),array('Q')).append(i)
        else:
            node=node.setdefault(r['episode_safety'],{}).setdefault(r['negative_stratum'],{})
            node.setdefault(r['episode_id'],array('Q')).append(i)
    result=(pools,positives)
    if isinstance(rows,SamplingRows):rows._sampling_pools=result
    return result


class JudgeBatchSampler:
    def __init__(self, rows, batch_size, *, seed, epoch, start_batch=0,
                 samples_per_epoch=None, positive_fraction=.5,rank=0,world_size=1):
        if not rows:raise ValueError('Sampler accepts nonempty train rows only')
        if any(type(v) is not int or v<0 for v in (seed,epoch,start_batch)) or type(batch_size) is not int or batch_size<1:
            raise ValueError('Invalid deterministic batch configuration')
        if not math.isfinite(positive_fraction) or not 0<positive_fraction<1:raise ValueError('Positive fraction must be between zero and one')
        if type(world_size) is not int or world_size<1 or not 0<=rank<world_size:raise ValueError('Invalid rank/world size')
        self.rows=rows;self.batch_size=batch_size;self.start_batch=start_batch;self.rank=rank;self.world_size=world_size
        pools,positives=sampling_pools(rows)
        self.available_labels={k for k,v in pools.items() if v}
        if samples_per_epoch is None:
            samples_per_epoch=max(1,round(positives/positive_fraction)) if pools[0] and pools[1] else len(rows)
        if type(samples_per_epoch) is not int or samples_per_epoch<1:raise ValueError('Positive integer samples_per_epoch required')
        samples_per_epoch=math.ceil(samples_per_epoch/(batch_size*world_size))*batch_size*world_size
        npos=round(samples_per_epoch*positive_fraction) if pools[0] and pools[1] else (samples_per_epoch if pools[1] else 0)
        rng=random.Random(seed+epoch);labels=[1]*npos+[0]*(samples_per_epoch-npos);rng.shuffle(labels)
        def choose(node):return node[rng.choice(sorted(node))]
        self.order=[]
        for label in labels:
            node=choose(pools[label])
            if label:indices=choose(node)
            else:indices=choose(choose(choose(node)))
            self.order.append(rng.choice(indices))
    def __iter__(self):
        width=self.batch_size*self.world_size
        for start in range(self.start_batch*width,len(self.order),width):
            at=start+self.rank*self.batch_size
            yield self.order[at:at+self.batch_size]
    def __len__(self):return max(0,math.ceil(len(self.order)/(self.batch_size*self.world_size))-self.start_batch)
    def report(self):
        return {**draw_report(self.rows,self.order,available_labels=self.available_labels),'scope':'planned full epoch; training may consume only a prefix'}


def consumed_draw_report(rows, *, batch_size, seed, epoch, batch_offset, world_size=1, **settings):
    available_labels=set()
    def indices():
        for e in range(epoch+1):
            sampler=JudgeBatchSampler(rows,batch_size,seed=seed,epoch=e,world_size=world_size,**settings)
            available_labels.update(sampler.available_labels)
            order=sampler.order
            yield from (order if e<epoch else order[:batch_offset*batch_size*world_size])
    return {**draw_report(rows,indices(),available_labels=available_labels),'scope':'consumed through the saved trainer cursor'}
