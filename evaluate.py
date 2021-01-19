from model import Model,ModelNtNs
import torch.nn.functional as F
import pickle as pkl
import argparse
import os
import torch
import torch.nn as nn
from library_data import *
from tqdm import trange
import numpy as np
from torch import optim


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='lastfm')
parser.add_argument('--model',default='model')
parser.add_argument('--p',default=0.8)
parser.add_argument('--gpu',default='0')
parser.add_argument('--size',default=3,type=int)
parser.add_argument('--epoch',default=0,type=int)
parser.add_argument('--dim',default=128,type=int)
args = parser.parse_args()

args.datapath = f'./data/{args.dataset}.csv'

args.train_proportion = args.p
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

[user2id, user_sequence_id, user_timediffs_sequence,
user_previous_itemid_sequence, item2id,
item_sequence_id, item_timediffs_sequence,
timestamp_sequence, feature_sequence,
y_true, user_freq] = load_data(args)

num_interactions = len(user_sequence_id)
num_users = len(user2id)
num_items = len(item2id) + 1
num_feats = len(feature_sequence[0])
true_labels_ratio = len(y_true)/(1.0+sum(y_true))

train_end_idx = validation_start_idx = int(num_interactions*
args.train_proportion)
test_start_idx = int(num_interactions*(args.train_proportion+0.1))
test_end_idx = int(num_interactions*(args.train_proportion+0.2))

dim = args.dim
item_max = 3
item_pow = 0.75
user_max = 4
user_pow = 0.75
size = args.size
dev = torch.device('cuda')
user_embs_static = torch.eye(num_users,device=dev)
item_embs_static = torch.diag(torch.tensor([item_max]*num_items,
                                           dtype=torch.float32)).to(dev)
user_timediffs_sequence = torch.tensor(user_timediffs_sequence,
                                       device=dev
                                      ).float().unsqueeze(1)
item_timediffs_sequence = torch.tensor(item_timediffs_sequence,
                                       device=dev
                                      ).float().unsqueeze(1)
feature_sequence = torch.tensor(feature_sequence,
                                device=dev
                               ).float()
prev_ids = []
prev_fqs = []
for prev in user_previous_itemid_sequence:
    tmp_ids = [num_items-1]*size
    tmp_fqs = [0]*size
    cur_ids = list(prev.keys())
    cur_fqs = list(prev.values())
    tmp_ids[size-len(cur_ids):] = cur_ids
    tmp_fqs[size-len(cur_fqs):] = cur_fqs
    prev_ids.append(tmp_ids)
    prev_fqs.append(tmp_fqs)
prev_ids = torch.tensor(prev_ids,device=dev)
prev_fqs = torch.tensor(prev_fqs,device=dev,dtype=torch.float32)
user_fqs = torch.tensor(user_freq,device=dev,dtype=torch.float32)

model = Model(dim,dim,num_users,num_items,num_feats,size)
state = torch.load(f'model/{args.model}_{args.dataset}_{args.size}_{args.epoch}.dat')
model.load_state_dict(state['model'])
model = model.cuda()
model.prev_wgt_stt = F.softmax(torch.arange(1,args.size+1).float().cuda()).unsqueeze(0)
#model.prev_wgt_stt = torch.arange(1,args.size+1).float().cuda().unsqueeze(0)

MSELoss = nn.MSELoss()
optr = optim.Adam(model.parameters(),lr=3e-4,weight_decay=1e-5)
optr.load_state_dict(state['optim'])

user_embs = state['user']
item_embs = state['item']

inte_id_his = []
user_id_his = []
item_id_his = []
start_time = timestamp_sequence[train_end_idx]
time_period = (timestamp_sequence[-1]-timestamp_sequence[0])/500

val_ranks = []
val_recall = 0
val_mrr = 0
val_total = 0
test_ranks = []
test_recall = 0
test_mrr = 0
test_total = 0
loss = 0 

with trange(train_end_idx,test_end_idx) as br:
    for cur in br:
        cur_time = timestamp_sequence[cur]
        user_id = [user_sequence_id[cur]]
        item_id = [item_sequence_id[cur]]
        item_prev_id = prev_ids[[cur]]
        item_prev_fq = prev_fqs[[cur]]
        user_fq = user_fqs[[cur]]
        user_time_diff = user_timediffs_sequence[[cur]]
        item_time_diff = item_timediffs_sequence[[cur]]

        #loss = 0
        user_embs.detach_()
        item_embs.detach_()

        user_emb = user_embs[user_id]
        item_emb = item_embs[item_id]
        prev_emb = item_embs[item_prev_id]
       
       # user_emb = model(user_embs[user_id].view(1,dim),
       #                  time_diff = time_diff.view(1,1),
       #                  mode = 'time'
       #                 )
        item_prev_sum = model(prev_emb,
                              mode='prev',
                              freq=item_prev_fq,
                              item_max=item_max,
                              item_pow=item_pow
                             )
        item_stat = model(mode='stat',
                          freq=item_prev_fq,
                          item_stat=item_embs_static[item_prev_id],
                          item_max=item_max,
                          item_pow=item_pow
                          )
        item_pred_emb = model(torch.cat([user_emb,
                                         item_prev_sum],
                                        dim=1
                                       ).detach(),
                              mode='pred',
                              freq=user_fq,
                              item_stat=item_stat,
                              user_stat=user_embs_static[user_id],
                              user_max=user_max,
                              user_pow=user_pow
                             )
        loss += MSELoss(item_pred_emb,                                                      
                        torch.cat([item_emb,                         
                                   item_embs_static[item_id]],           
                                  dim=1
                                 )                  
                       )
        
        distances = ((torch.cat([item_embs,item_embs_static],dim=1) - item_pred_emb)**2).sum(1).cpu().detach().numpy()
        #distances = ((item_embs_static - item_pred_emb[:,128:])**2).sum(1).cpu().detach().numpy()
        #distances = nn.PairwiseDistance()(item_pred_emb.repeat(num_items,1),torch.cat([item_embs,item_embs_static],dim=1)).squeeze(-1)
        #distances = nn.PairwiseDistance()(item_pred_emb[:,128:].repeat(num_items,1),item_embs_static).squeeze(-1)
        #print(distances)
        true_dis = distances[item_id]
        
        rank = np.where(distances>=true_dis,0,1).sum() + 1
        #rank = np.sum((distances<true_dis).cpu().detach().numpy())+1
        if cur < test_start_idx:
            #val_mrr += 1/rank
            #val_recall += 1 if rank <= 10 else 0
            #val_total += 1
            val_ranks.append(rank)
        else:
            if cur == test_start_idx:
                val_mrr = np.mean([1.0 / r for r in val_ranks])
                val_recall = np.sum(np.array(val_ranks)<=10)*1.0/len(val_ranks)
                #val_mrr /= val_total
                #val_recall /= val_total
                print(f'validation:')
                print(f'mrr {val_mrr:.6f} recall {val_recall:.6f}')
            #test_mrr += 1/rank
            #test_recall += 1 if rank <= 10 else 0
            #test_total += 1
            test_ranks.append(rank)
        
        user_emb_nxt = model(user_emb,
                             torch.cat([item_emb,
                                        user_time_diff,
                                        feature_sequence[[cur]]],
                                       dim = 1
                                      ),
                             mode='user'
                            )

        item_emb_nxt = model(torch.cat([user_emb,
                                        item_time_diff,
                                        feature_sequence[[cur]]],
                                      dim = 1
                                      ),
                             item_embs[item_id],
                             mode='item'
                            )
        item_emb_pre = model(item_emb,
                             prev_emb,
                             mode='addi',
                             freq=item_prev_fq)


        user_embs[user_id] = user_emb_nxt
        item_embs[item_id] = item_emb_nxt
        item_embs[item_prev_id] = item_emb_pre
        
        loss += MSELoss(user_emb_nxt,user_emb.detach())
        loss += MSELoss(item_emb_nxt,item_emb.detach())
        #loss += MSELoss(item_emb_pre,prev_emb.detach())
        
        if cur_time - start_time > time_period:
            loss.backward()
            optr.step()
            optr.zero_grad()
            start_time = cur_time
            loss = 0

#test_mrr /= test_total
#test_recall /= test_total
test_mrr = np.mean([1.0 / r for r in test_ranks])
test_recall = np.sum(np.array(test_ranks)<=10)*1.0/len(test_ranks)
print(f'test:')
print(f'mrr {test_mrr:.6f} recall {test_recall:.6f}')
