import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn_layer import NR_GraphAttention,NR_GraphAttentionCross
from tabulate import tabulate
import logging
from torch_scatter import scatter_mean

class MultiModalFusion(nn.Module):
    def __init__(self, modal_num, with_weight=1):
        super().__init__()
        self.modal_num = modal_num
        self.requires_grad = True if with_weight > 0 else False
        self.weight = nn.Parameter(torch.Tensor([1,0.2]),
                                   requires_grad=self.requires_grad)

    def forward(self, embs):
        assert len(embs) == self.modal_num
        weight_norm = F.softmax(self.weight, dim=0)
        embs = [weight_norm[idx] * embs[idx] for idx in range(self.modal_num) if embs[idx] is not None]
        if not self.training:
            print("modal weight: ",weight_norm)
        joint_emb = torch.cat(embs, dim=1)
        # joint_emb = torch.sum(torch.stack(embs, dim=1), dim=1)
        return joint_emb


class Encoder_Model(nn.Module):
    def __init__(self, node_hidden, rel_hidden, triple_size, node_size, new_node_size, rel_size, device,
                 adj_matrix, r_index, r_val, rel_matrix, ent_matrix,img_feature,
                 dropout_rate=0.0, ind_dropout_rate=0.0, gamma=3, lr=0.005, depth=2):
        super(Encoder_Model, self).__init__()
        self.node_hidden = node_hidden
        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.depth = depth
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)
        self.ind_dropout = nn.Dropout(ind_dropout_rate)
        self.gamma = gamma
        self.lr = lr
        self.adj_list = adj_matrix.to(device)
        self.r_index = r_index.to(device)
        self.r_val = r_val.to(device)
        self.rel_adj = rel_matrix.to(device)
        
        self.ent_adj = ent_matrix.to(device)
        self.img_feature = img_feature.to(device)

        self.new_node_size = new_node_size


        self.ent_embedding = nn.Embedding(node_size, node_hidden)
        self.rel_embedding = nn.Embedding(rel_size, rel_hidden)
        self.img_embedding = nn.Embedding(2048, node_hidden)
        torch.nn.init.xavier_uniform_(self.ent_embedding.weight)
        torch.nn.init.xavier_uniform_(self.rel_embedding.weight)
        torch.nn.init.xavier_uniform_(self.img_embedding.weight)

        self.e_encoder = NR_GraphAttention(node_size=self.new_node_size,
                                           rel_size=self.rel_size,
                                           triple_size=self.triple_size,
                                           node_dim=self.node_hidden,
                                           depth=self.depth,
                                           use_bias=True
                                           )
        self.e_encoder_img = NR_GraphAttention(node_size=self.new_node_size,
                                           rel_size=self.rel_size,
                                           triple_size=self.triple_size,
                                           node_dim=self.node_hidden,
                                           depth=self.depth,
                                           use_bias=True
                                           )
        

        self.e_encoder_cross = NR_GraphAttentionCross(node_size=self.new_node_size,
                                           rel_size=self.rel_size,
                                           triple_size=self.triple_size,
                                           node_dim=self.node_hidden,
                                           depth=self.depth,
                                           use_bias=True
                                           )
        self.e_encoder_img_cross = NR_GraphAttentionCross(node_size=self.new_node_size,
                                           rel_size=self.rel_size,
                                           triple_size=self.triple_size,
                                           node_dim=self.node_hidden,
                                           depth=self.depth,
                                           use_bias=True
                                           )
        
        self.r_encoder = NR_GraphAttention(node_size=self.new_node_size,
                                           rel_size=self.rel_size,
                                           triple_size=self.triple_size,
                                           node_dim=self.node_hidden,
                                           depth=self.depth,
                                           use_bias=True
                                           )


        self.fusion = MultiModalFusion(modal_num=2)


    def avg(self, adj, emb, size: int):
        adj = torch.sparse_coo_tensor(indices=adj, values=torch.ones_like(adj[0, :], dtype=torch.float),
                                      size=[self.node_size, size])
        adj = torch.sparse.softmax(adj, dim=1)
        return torch.sparse.mm(adj, emb)

    def gcn_forward(self):
        # [Ne x Ne] · [Ne x dim] = [Ne x dim]
        ent_feature = self.avg(self.ent_adj, self.ent_embedding.weight, self.node_size)
        # [Ne x Nr] · [Nr x dim] = [Ne x dim]
        rel_feature = self.avg(self.rel_adj, self.rel_embedding.weight, self.rel_size)
        img_feature = torch.mm(self.img_feature,self.img_embedding.weight)
        # img_feature = self.img_feature

        opt = [self.rel_embedding.weight, self.adj_list, self.r_index, self.r_val]
        out_rel_feature = self.r_encoder([rel_feature] + opt)

        # out_feature_join,img_rel_feature,ent_rel_feature = self.gcn_forward()



        out_ent_feature = self.e_encoder_cross([ent_feature] + opt+[img_feature,False])
        out_img_feature = self.e_encoder_img_cross([img_feature] + opt+[ent_feature,True])
        # out_img_feature = img_feature
        out_feature_join = torch.cat([out_ent_feature,out_img_feature*0.3,out_rel_feature], dim=-1)
        # out_feature = torch.cat([self.e_encoder([ent_feature] + opt), out_rel_feature], dim=-1)

        # joint_emb  =[out_ent_feature,out_img_feature]
        # # out_rel_feature=F.normalize(out_rel_feature)
        # joint_emb = self.fusion(joint_emb)

        # out_feature_join =torch.cat([joint_emb,out_rel_feature], dim=-1)
        # out_feature_join =joint_emb

        img_rel_feature= torch.cat([out_img_feature,out_rel_feature], dim=-1)
        # img_rel_feature= out_img_feature
        ent_rel_feature= torch.cat([out_ent_feature,out_rel_feature], dim=-1)

        img_rel_feature = self.dropout(img_rel_feature)
        ent_rel_feature = self.dropout(ent_rel_feature)
        out_feature_join = self.dropout(out_feature_join)


        # img_out_feature = torch.cat([self.e_encoder_img([self.img_feature] + opt), out_rel_feature], dim=-1)
        # out_feature = self.dropout(out_feature)
        # img_out_feature = self.dropout(img_out_feature)

        return out_feature_join,img_rel_feature,ent_rel_feature
        # return out_feature,img_out_feature

    def forward(self, train_paris:torch.Tensor):
        out_feature_join,img_rel_feature,ent_rel_feature = self.gcn_forward()

        loss1 = self.align_loss(train_paris, out_feature_join)
        loss2 = self.align_loss(train_paris, img_rel_feature)
        loss3 = self.align_loss(train_paris, ent_rel_feature)
 
        # return loss1 +loss2+loss3
        return loss1


    def align_loss(self, pairs, emb):
        def squared_dist(A, B):
            row_norms_A = torch.sum(torch.square(A), dim=1)
            row_norms_A = torch.reshape(row_norms_A, [-1, 1])
            row_norms_B = torch.sum(torch.square(B), dim=1)
            row_norms_B = torch.reshape(row_norms_B, [1, -1])
            return row_norms_A + row_norms_B - 2 * torch.matmul(A, B.t())

        l, r = pairs[:, 0].long(), pairs[:, 1].long()
        l_emb, r_emb = emb[l], emb[r]

        pos_dis = torch.sum(torch.square(l_emb - r_emb), dim=-1, keepdim=True)
        l_neg_dis = squared_dist(l_emb, emb)
        r_neg_dis = squared_dist(r_emb, emb)

        del l_emb, r_emb

        l_loss = pos_dis - l_neg_dis + self.gamma
        l_loss = l_loss * (1 - F.one_hot(l, num_classes=self.node_size) - F.one_hot(r, num_classes=self.node_size))
        r_loss = pos_dis - r_neg_dis + self.gamma
        r_loss = r_loss * (1 - F.one_hot(l, num_classes=self.node_size) - F.one_hot(r, num_classes=self.node_size))

        del r_neg_dis, l_neg_dis

        r_loss = (r_loss - torch.mean(r_loss, dim=-1, keepdim=True).detach()) / torch.std(r_loss, dim=-1, unbiased=False, keepdim=True).detach()
        l_loss = (l_loss - torch.mean(l_loss, dim=-1, keepdim=True).detach()) / torch.std(l_loss, dim=-1, unbiased=False, keepdim=True).detach()

        lamb, tau = 30, 10
        # print("l_loss:{}, r_loss:{}".format(l_loss,r_loss))
        l_loss = torch.logsumexp(lamb * l_loss + tau, dim=-1)
        r_loss = torch.logsumexp(lamb * r_loss + tau, dim=-1)
        return torch.mean(l_loss + r_loss)



    def get_embeddings(self, index_a, index_b):
        # forward
        out_feature_join,img_rel_feature,ent_rel_feature  = self.gcn_forward()
        out_feature = out_feature_join
        out_feature = out_feature.cpu()

        # get embeddings
        index_a = torch.Tensor(index_a).long()
        index_b = torch.Tensor(index_b).long()
        Lvec = out_feature[index_a]
        Rvec = out_feature[index_b]
        Lvec = Lvec / (torch.linalg.norm(Lvec, dim=-1, keepdim=True) + 1e-5)
        Rvec = Rvec / (torch.linalg.norm(Rvec, dim=-1, keepdim=True) + 1e-5)

        return Lvec, Rvec

    def print_all_model_parameters(self):
        logging.info('\n------------Model Parameters--------------')
        info = []
        head = ["Name", "Element Nums", "Element Bytes", "Total Size (MiB)", "requires_grad"]
        total_size = 0
        total_element_nums = 0
        for name, param in self.named_parameters():
            info.append((name,
                         param.nelement(),
                         param.element_size(),
                         round((param.element_size()*param.nelement())/2**20, 3),
                         param.requires_grad)
                        )
            total_size += (param.element_size()*param.nelement())/2**20
            total_element_nums += param.nelement()
        logging.info(tabulate(info, headers=head, tablefmt="grid"))
        logging.info(f'Total # parameters = {total_element_nums}')
        logging.info(f'Total # size = {round(total_size, 3)} (MiB)')
        logging.info('--------------------------------------------')
        logging.info('')


