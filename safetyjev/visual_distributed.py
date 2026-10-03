"""Rank-independent batch semantics and disjoint evaluation partitions."""
import math
import torch


def resolve_accumulation(global_batch, microbatch, world_size):
    if any(type(v) is not int or v<1 for v in (global_batch,microbatch,world_size)):
        raise ValueError('Batch sizes and world size must be positive integers')
    divisor=microbatch*world_size
    if global_batch%divisor:raise ValueError('global_batch_size must divide by microbatch * world_size exactly')
    return global_batch//divisor


def evaluation_indices(size,rank,world_size):
    return range(rank,size,world_size)


class GlobalBatchSampler:
    """Partition global draws, not independent per-rank samplers. Pad training only."""
    def __init__(self,size,batch_size,*,seed,epoch,rank=0,world_size=1,start_batch=0,weights=None):
        if size<1 or batch_size<1 or not 0<=rank<world_size or start_batch<0:raise ValueError('Invalid sampler arguments')
        self.size=size;self.batch_size=batch_size;self.seed=seed;self.epoch=epoch
        self.rank=rank;self.world_size=world_size;self.start_batch=start_batch;self.weights=weights
    def __len__(self):return max(0,math.ceil(self.size/(self.batch_size*self.world_size))-self.start_batch)
    def __iter__(self):
        g=torch.Generator().manual_seed(self.seed+self.epoch)
        width=self.batch_size*self.world_size;total=math.ceil(self.size/width)*width
        if self.weights is None:
            order=torch.randperm(self.size,generator=g)
            if total>len(order):order=order.repeat(math.ceil(total/len(order)))[:total]
        else:order=torch.multinomial(torch.as_tensor(self.weights,dtype=torch.double),total,replacement=True,generator=g)
        for start in range(self.start_batch*width,total,width):
            at=start+self.rank*self.batch_size
            yield order[at:at+self.batch_size].tolist()
