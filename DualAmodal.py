import warnings
warnings.filterwarnings('ignore')

import os
# import keras
import numpy as np
from utils import *
from tqdm import *
from evaluate import evaluate
import tensorflow as tf
# import keras.backend as K
# from keras.layers import *
from models import *
import torch.nn.functional as F
import torch
from models_encodercat import Encoder_Model
import random


def set_seed(seed, cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True  
#sess = tf.Session(config=config)  

seed =12306 
np.random.seed(seed)

set_seed(seed)
tf.compat.v1.set_random_seed(seed)

# train_pair,dev_pair,adj_matrix,r_index,r_val,adj_features,rel_features = load_data("data/fb_yago_en/",train_ratio=0.20)
data_dir = 'data/zh_en/'
train_pair,dev_pair,adj_matrix,r_index,r_val,adj_features,rel_features = load_data(data_dir,train_ratio=0.30)
adj_matrix = np.stack(adj_matrix.nonzero(),axis = 1)
rel_matrix,rel_val = np.stack(rel_features.nonzero(),axis = 1),rel_features.data
ent_matrix,ent_val = np.stack(adj_features.nonzero(),axis = 1),adj_features.data




node_size = adj_features.shape[0]
rel_size = rel_features.shape[1]
triple_size = len(adj_matrix)
node_hidden = 128
rel_hidden = 128
batch_size = 1024
dropout_rate = 0.3
lr = 0.005
gamma = 1
depth = 2

device = "cuda:0"
# img_features = load_img_features(node_size, "data/fb_dbk_en/FB15K_DB15K")
# img_features = load_img_features(node_size, "data/fb_dbk_en/FB15K_YAGO15K")
img_features = load_img_features(node_size, data_dir)
img_features = F.normalize(torch.Tensor(img_features).to(device))
print("image feature shape:", img_features.shape)


triple_size = len(adj_matrix)  # not triple size, but number of diff(h, t)
eval_epoch = 3


device = 'cuda'

training_time = 0.
grid_search_time = 0.
time_encode_time = 0.

 
e1 = os.path.join(data_dir,'ent_ids_1')
e2 = os.path.join(data_dir,'ent_ids_2')
left_ents = get_ids(e1)
right_ents = get_ids(e2)

unsup=True
unsup_k=6300



visual_links=[]
used_inds = []

l_img_f = img_features[left_ents] # left images
r_img_f = img_features[right_ents] # right images

img_sim = l_img_f.mm(r_img_f.t())
topk =unsup_k
two_d_indices,vals = get_topk_indices(img_sim, topk*100)
del l_img_f, r_img_f, img_sim



count = 0
for index in range(len(two_d_indices)):
    ind = two_d_indices[index]
    if left_ents[ind[0]] in used_inds: continue
    if right_ents[ind[1]] in used_inds: continue
    used_inds.append(left_ents[ind[0]])
    used_inds.append(right_ents[ind[1]])
    visual_links.append((left_ents[ind[0]], right_ents[ind[1]],vals[index].cpu(),vals[index].cpu()))
    # visual_links.append((left_ents[ind[0]], right_ents[ind[1]]))

    count += 1
    if count == topk: 
        break

for index in range(len(train_pair)):
    ind = train_pair[index]
    visual_links.append((ind[0], ind[1],1,1))


train_pair = np.array(visual_links, dtype=np.float32)

print('begin')
# inputs = [adj_input, index_input, val_input, rel_adj, ent_adj]
inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix]
# model = OverAll(node_size=node_size, node_hidden=node_hidden,
#                 rel_size=rel_size, rel_hidden=rel_hidden,
#                 rel_matrix=rel_matrix, ent_matrix=ent_matrix,
#                 triple_size=triple_size, dropout_rate=dropout_rate,
#                 depth=depth, device=device)
# model = model.to(device)

adj_matrix = torch.from_numpy(np.transpose(adj_matrix))
rel_matrix = torch.from_numpy(np.transpose(rel_matrix))
ent_matrix = torch.from_numpy(np.transpose(ent_matrix))
r_index = torch.from_numpy(np.transpose(r_index))
r_val = torch.from_numpy(r_val)

model = Encoder_Model(node_size=node_size, node_hidden=node_hidden,
                rel_size=rel_size, rel_hidden=rel_hidden,new_node_size=node_size,
                rel_matrix=rel_matrix, ent_matrix=torch.tensor(ent_matrix),r_index=r_index,r_val=r_val,
                adj_matrix=adj_matrix,img_feature = img_features,
                triple_size=triple_size, dropout_rate=dropout_rate,
                depth=depth, device=device)
model = model.to(device)
# opt = torch.optim.RMSprop(model.parameters(), lr=lr)
opt = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=0.0005)
print('model constructed')

evaluater = evaluate(dev_pair)
rest_set_1 = [e1 for e1, e2 in dev_pair]
rest_set_2 = [e2 for e1, e2 in dev_pair]
np.random.shuffle(rest_set_1)
np.random.shuffle(rest_set_2)

model.eval()
with torch.no_grad():
    Lvec, Rvec = model.get_embeddings(dev_pair[:, 0], dev_pair[:, 1],turn=0)
    # output = model(inputs)
    # Lvec, Rvec = get_embedding(dev_pair[:, 0], dev_pair[:, 1], output.cpu())
    evaluater.test(Lvec, Rvec)
epoch = 12

for turn in range(5):

    for i in trange(epoch):
        model.train()
        np.random.shuffle(train_pair)
        for pairs in [train_pair[i * batch_size:(i + 1) * batch_size] for i in
                        range(len(train_pair) // batch_size + 1)]:
            inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix]
            # output = model(inputs)
            # loss = align_loss(pairs, output)
            pairs = torch.from_numpy(pairs).to(device)
            loss  =model(pairs,turn)
            # print(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # if i == epoch - 1:
            # toc = time.time()
        model.eval()
        with torch.no_grad():
            Lvec, Rvec = model.get_embeddings(dev_pair[:, 0], dev_pair[:, 1],turn)
            # output = model(inputs)
            # Lvec, Rvec = get_embedding(dev_pair[:, 0], dev_pair[:, 1], output.cpu())
            evaluater.test(Lvec, Rvec)
            # output2 = output.cpu().numpy()
            # output2 = output2 / (np.linalg.norm(output2, axis=-1, keepdims=True) + 1e-5)
            # dto.saveobj(output2, 'embedding_of_' + save_suffix())
  

        new_pair = []   
    Lvec,Rvec = model.get_embeddings(rest_set_1,rest_set_2,turn)
    A,B,A_score,B_score = evaluater.CSLS_cal(Lvec.detach().numpy(),Rvec.detach().numpy(),False)

    
    for i,j in enumerate(A):
        if  B[j] == i:
            new_pair.append([rest_set_1[j],rest_set_2[i],A_score[i],B_score[j]])
            # new_pair_value.append([A_score[j],B_score[i]])  # 不清楚对不对
    A_score_sorted = np.sort(np.array(new_pair)[:,2])
    print ("highest A_score_sorted:", A_score_sorted[-1].item(), "lowest A_score_sorted:", A_score_sorted[0].item())

    B_score_sorted = np.sort(np.array(new_pair)[:,3])
    print ("highest B_score_sorted:", B_score_sorted[-1].item(), "lowest B_score_sorted:", B_score_sorted[0].item())



    train_pair = np.concatenate([train_pair,np.array(new_pair)],axis = 0)
    # train_pair_score = np.concatenate([train_pair_score,np.array(new_pair_value)],axis = 0)
    for e1,e2,_,_ in new_pair:
        if e1 in rest_set_1:
            rest_set_1.remove(e1) 
        
    for e1,e2,_,_ in new_pair:
        if e2 in rest_set_2:
            rest_set_2.remove(e2)
    epoch = 5








