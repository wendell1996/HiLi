from library_data import *
from collections import defaultdict
import argparse
import os
from model import Model
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch import optim
import pickle as pkl
from tqdm import trange


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='lastfm')
parser.add_argument('--model',default='model')
parser.add_argument('--size',default=3,type=int)
parser.add_argument('--p',default=0.8)
parser.add_argument('--gpu',default='0')
parser.add_argument('--epoch',default=1,type=int)
parser.add_argument('--dim',default=128,type=int)
args = parser.parse_args()

args.train_proportion = args.p
args.datapath = f'./data/{args.dataset}.csv'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

[user2id, user_sequence_id, user_timediffs_sequence,
user_previous_itemid_sequence, item2id,
item_sequence_id, item_timediffs_sequence,
timestamp_sequence, feature_sequence,
y_true, user_frequence] = load_data(args)

num_interactions = len(user_sequence_id)
num_users = len(user2id)
num_items = len(item2id) + 1
num_feats = len(feature_sequence[0])
true_labels_ratio = len(y_true)/(1.0+sum(y_true))

train_end_idx = validation_start_idx = int(num_interactions *
args.train_proportion)
test_start_idx = int(num_interactions * (args.train_proportion+0.1))
test_end_idx = int(num_interactions * (args.train_proportion+0.2))


dim = args.dim
item_max = 3
item_pow = 0.75
user_max = 4
user_pow = 0.75
size = args.size
dev = torch.device('cuda')
user_embs = F.normalize(torch.rand((1,dim),device=dev)).squeeze()
user_emb_init = user_embs.clone()
user_embs = user_embs.repeat((num_users,1))
item_embs = F.normalize(torch.rand((1,dim),device=dev)).squeeze()
item_emb_init = item_embs.clone()
item_embs = item_embs.repeat((num_items,1))
user_timediffs_sequence = torch.tensor(user_timediffs_sequence,
                                       device=dev
                                      ).float().unsqueeze(1)
item_timediffs_sequence = torch.tensor(item_timediffs_sequence,
                                       device=dev
                                      ).float().unsqueeze(1)
feature_sequence = torch.tensor(feature_sequence,
                                device = dev
                               ).float()
user_embs_static = torch.eye(num_users,device=dev)
item_embs_static = torch.diag(torch.tensor([item_max]*num_items,
                                           dtype=torch.float32)).to(dev)
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
user_fqs = torch.tensor(user_frequence, device=dev,dtype=torch.float32)


model = Model(dim,dim,num_users,num_items,num_feats,size).to(dev)
model.prev_wgt_stt = F.softmax(torch.arange(1,args.size+1).float().cuda()).unsqueeze(0)
MSELoss = nn.MSELoss()
optr = optim.Adam(model.parameters(),lr=3e-4,weight_decay=1e-5)

inte_id_his = defaultdict(list)
user_id_his = defaultdict(list)
item_id_his = defaultdict(list)
user_id_pre = defaultdict(lambda: -1)
item_id_pre = defaultdict(lambda: -1)
start_time = 0
time_period = (timestamp_sequence[-1]-timestamp_sequence[0])/500
inte_cnt = 0

loss = 0
loss_sm = 0

for epoch in range(args.epoch):
    with trange(train_end_idx) as br2:
        for cur in br2:
            user_embs.detach_()
            item_embs.detach_()

            cur_time = timestamp_sequence[cur]
            cur_id = max(user_id_pre[user_sequence_id[cur]],
                         item_id_pre[item_sequence_id[cur]])+1
            user_id_pre[user_sequence_id[cur]] = cur_id
            item_id_pre[item_sequence_id[cur]] = cur_id
            inte_id_his[cur_id].append(cur)
            user_id_his[cur_id].append(user_sequence_id[cur])
            item_id_his[cur_id].append(item_sequence_id[cur])

            if cur_time - start_time <= time_period:
                continue
            start_time = cur_time

            for cur_idx in range(len(inte_id_his)):

                inte_id = inte_id_his[cur_idx]
                user_id = user_id_his[cur_idx]
                item_prev_id = prev_ids[inte_id]
                item_prev_fq = prev_fqs[inte_id]
                user_fq = user_fqs[inte_id]
                item_id = item_id_his[cur_idx]
                user_time_diff = user_timediffs_sequence[inte_id]
                item_time_diff = item_timediffs_sequence[inte_id]

                user_emb = user_embs[user_id]
                item_emb = item_embs[item_id]
                prev_emb = item_embs[item_prev_id]

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
                                                 dim = 1
                                               ).detach(),
                                      mode='pred',
                                      item_stat=item_stat,
                                      user_stat=user_embs_static[user_id],
                                      freq=user_fq,
                                      user_max=user_max,
                                      user_pow=user_pow
                                     )
                loss += MSELoss(item_pred_emb,
                                torch.cat([item_emb,
                                           item_embs_static[item_id]],
                                          dim = 1
                                         ).detach()
                               )
                user_emb_nxt = model(user_emb,
                                     torch.cat([item_emb,
                                                user_time_diff,
                                                feature_sequence[inte_id]],
                                               dim=1
                                              ).detach(),
                                     mode='user'
                                    )
                item_emb_nxt = model(torch.cat([user_emb,
                                                item_time_diff,
                                                feature_sequence[inte_id]],
                                               dim=1
                                              ).detach(),
                                     item_emb,
                                     mode='item'
                                    )
                item_emb_pre = model(item_emb,
                                     prev_emb,
                                     mode='addi',
                                     freq=item_prev_fq,
                                     item_max=item_max,
                                     item_pow=item_pow
                                    )

                user_embs[user_id] = user_emb_nxt
                item_embs[item_id] = item_emb_nxt
                item_embs[item_prev_id] = item_emb_pre

                loss_sm += MSELoss(user_emb_nxt,
                                user_emb)
                loss_sm += MSELoss(item_emb_nxt,
                                item_emb)
                inte_cnt += 1

            loss += loss_sm/inte_cnt
            br2.set_description(f'loss:{loss}')
            loss.backward()
            optr.step()
            optr.zero_grad()
            loss = 0
            loss_sm = 0
            inte_cnt = 0

            inte_id_his = defaultdict(list)
            user_id_his = defaultdict(list)
            item_id_his = defaultdict(list)
            user_id_pre = defaultdict(lambda: -1)
            item_id_pre = defaultdict(lambda: -1)

    state = {'model':model.state_dict(),
             'optim':optr.state_dict(),
             'user':user_embs,
             'item':item_embs}
    torch.save(state,f'model/{args.model}_{args.dataset}_{args.size}_{epoch}.dat')

    inte_id_his = defaultdict(list)
    user_id_his = defaultdict(list)
    item_id_his = defaultdict(list)
    user_id_pre = defaultdict(lambda: -1)
    item_id_pre = defaultdict(lambda: -1)

    start_time = 0
    loss = 0
    loss_sm = 0
    inte_cnt = 0

    user_embs = user_emb_init.repeat((num_users,1))
    item_embs = item_emb_init.repeat((num_items,1))
