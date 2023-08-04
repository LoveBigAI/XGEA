import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn_layer import NR_GraphAttention,NR_GraphAttentionCross,NR_GraphAttentionMu
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
        
        self.e_encoder_mu = NR_GraphAttentionMu(node_size=self.new_node_size,
                                           rel_size=self.rel_size,
                                           triple_size=self.triple_size,
                                           node_dim=self.node_hidden,
                                           depth=self.depth,
                                           use_bias=True
                                           )
        self.e_encoder_img_mu = NR_GraphAttentionMu(node_size=self.new_node_size,
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

    def gcn_forward(self,turn):
        # [Ne x Ne] · [Ne x dim] = [Ne x dim]
        ent_feature = self.avg(self.ent_adj, self.ent_embedding.weight, self.node_size)
        # [Ne x Nr] · [Nr x dim] = [Ne x dim]
        rel_feature = self.avg(self.rel_adj, self.rel_embedding.weight, self.rel_size)
        img_feature = torch.mm(self.img_feature,self.img_embedding.weight)
        # img_feature = self.img_feature

        opt = [self.rel_embedding.weight, self.adj_list, self.r_index, self.r_val]
        out_rel_feature = self.r_encoder([rel_feature] + opt)

        # out_feature_join,img_rel_feature,ent_rel_feature = self.gcn_forward()

        out_ent_feature = self.e_encoder([ent_feature] + opt)
        out_img_feature = self.e_encoder_img([img_feature] + opt)

        out_ent_feature_mu = self.e_encoder_mu([ent_feature] + opt+[img_feature])
        out_img_feature_mu = self.e_encoder_img_mu([img_feature] + opt+[ent_feature])

        # out_ent_feature = self.e_encoder_cross([ent_feature] + opt+[img_feature,False])
        # out_img_feature = self.e_encoder_img_cross([img_feature] + opt+[ent_feature,True])
        # out_img_feature = img_feature
        # if turn>=1:
        # out_feature_join = torch.cat([out_ent_feature,out_img_feature*0.3,out_ent_feature_mu*1,out_img_feature_mu*0.1 ,out_rel_feature], dim=-1)
        # out_feature_join = torch.cat([out_ent_feature,out_img_feature*0.05,out_ent_feature_mu*1,out_img_feature_mu*0.1 ,out_rel_feature], dim=-1)
        # out_feature_join = torch.cat([out_ent_feature,out_ent_feature_mu*1,out_img_feature_mu*0.1 ,out_rel_feature], dim=-1)# 目前最好的
        out_feature_join = torch.cat([out_ent_feature,out_rel_feature], dim=-1)# 目前最好的
        # out_feature_join = torch.cat([out_ent_feature ,out_rel_feature], dim=-1)
        # out_feature_join = torch.cat([out_ent_feature,out_img_feature*0.3 ,out_rel_feature], dim=-1)
        # out_feature_join = torch.cat([out_ent_feature,out_ent_feature_mu*0.1 ,out_img_feature_mu*0.1, out_rel_feature], dim=-1)
        # else :
        #    out_feature_join = torch.cat([out_ent_feature,out_img_feature*0.35,out_rel_feature], dim=-1)
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

    def forward(self, train_paris:torch.Tensor,turn):
        out_feature_join,img_rel_feature,ent_rel_feature = self.gcn_forward(turn)

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

        l_score,r_score = pairs[:, 2],pairs[:, 3].long()
        
        
        n_sigma=0.9

        # sigma=0.9   u=0.56  
        # 最好  Hits@1:  0.8753333333333333   Hits@5:  0.946952380952381   Hits@10:  0.9631428571428572   MRR:  0.9074677604822035  
        # 最差 Hits@1:  0.8731428571428571   Hits@5:  0.9472380952380952   Hits@10:  0.9641904761904762   MRR:  0.9058964547021855
         # Hits@1:  0.8722857142857143   Hits@5:  0.9459047619047619   Hits@10:  0.9633333333333334   MRR:  0.9057415920000347

        # sigma=0.88   u=0.55  
        # 最好 Hits@1:  0.8756190476190476   Hits@5:  0.9460952380952381   Hits@10:  0.9615238095238096   MRR:  0.9071719028850267
        # 最差  Hits@1:  0.8712380952380953   Hits@5:  0.9464761904761905   Hits@10:  0.9622857142857143   MRR:  0.9047243247894088

        # sigma=0.9   u=0.53  
        #最好  Hits@1:  0.8752380952380953   Hits@5:  0.9480952380952381   Hits@10:  0.9635238095238096   MRR:  0.9080578655174485
        #最差 Hits@1:  0.8739047619047619   Hits@5:  0.9473333333333334   Hits@10:  0.9619047619047619   MRR:  0.906856303818705
        # Hits@1:  0.8742857142857143   Hits@5:  0.946952380952381   Hits@10:  0.962952380952381   MRR:  0.9068770034244961
        #Hits@1:  0.871904761904762   Hits@5:  0.9480952380952381   Hits@10:  0.9631428571428572   MRR:  0.9055618830615737
        # Hits@1:  0.8727619047619047   Hits@5:  0.9482857142857143   Hits@10:  0.9637142857142857   MRR:  0.9062328695776046
        # Hits@1:  0.8749523809523809   Hits@5:  0.9482857142857143   Hits@10:  0.9636190476190476   MRR:  0.9074880243404994
        # Hits@1:  0.8736190476190476   Hits@5:  0.9477142857142857   Hits@10:  0.9642857142857143   MRR:  0.9068136879244344


        # sigma=0.9   u=0.51  
        #Hits@1:  0.8734285714285714   Hits@5:  0.946   Hits@10:  0.9627619047619047   MRR:  0.9058746908437892
        #Hits@1:  0.8743809523809524   Hits@5:  0.9471428571428572   Hits@10:  0.9635238095238096   MRR:  0.9068694008956852

        # n_sigma=0.88 u=0.51  
        # Hits@1:  0.8722857142857143   Hits@5:  0.9472380952380952   Hits@10:  0.9624761904761905   MRR:  0.9056926202649038

        # n_sigma=0.9 u=0.51  
        #Hits@1:  0.8718095238095238   Hits@5:  0.9459047619047619   Hits@10:  0.9613333333333334   MRR:  0.905019535548377

        # n_sigma=0.92 u=0.53  
        # Hits@1:  0.872   Hits@5:  0.947047619047619   Hits@10:  0.9628571428571429   MRR:  0.9054179985132265

        # n_sigma=0.88 u=0.53  
        # Hits@1:  0.8737142857142857   Hits@5:  0.9479047619047619   Hits@10:  0.9636190476190476   MRR:  0.9068495902936763
        # Hits@1:  0.874   Hits@5:  0.9460952380952381   Hits@10:  0.9619047619047619   MRR:  0.9064507335950953
        #Hits@1:  0.8706666666666667   Hits@5:  0.9463809523809524   Hits@10:  0.9615238095238096   MRR:  0.9044798084642127

        # n_sigma=0.9 u=0.52  
        # Hits@1:  0.8718095238095238   Hits@5:  0.9476190476190476   Hits@10:  0.963047619047619   MRR:  0.9055502665554463

       # n_sigma=0.88 u=0.52  
        # Hits@1:  0.8727619047619047   Hits@5:  0.9458095238095238   Hits@10:  0.9631428571428572   MRR:  0.9059858818658821

       # n_sigma=1 u=0.45  
    # Hits@1:  0.8703809523809524   Hits@5:  0.9457142857142857   Hits@10:  0.9615238095238096   MRR:  0.9040969012755444
           # n_sigma=1.4 u=0.45  
    # Hits@1:  0.8721904761904762   Hits@5:  0.9461904761904761   Hits@10:  0.9622857142857143   MRR:  0.9052006111839467

           # n_sigma=1.2 u=0.45  
    # Hits@1:  0.8720952380952381   Hits@5:  0.944952380952381   Hits@10:  0.9611428571428572   MRR:  0.9050532288620625

           # n_sigma=1.2 u=0.4 
    # Hits@1:  0.8729523809523809   Hits@5:  0.9461904761904761   Hits@10:  0.9615238095238096   MRR:  0.9056205665436099

        # l_score_mask = torch.exp(-((torch.clamp(l_score - 0.6, max=0.0) ** 2) / (2  / (n_sigma ** 2))))
        # r_score_mask = torch.exp(-((torch.clamp(r_score - 0.6, max=0.0) ** 2) / (2  / (n_sigma ** 2))))

        l_score_mask = torch.exp((l_score - 0.6))/ (n_sigma ** 2)
        r_score_mask = torch.exp((r_score - 0.6) )/ (n_sigma ** 2)

        pos_dis = torch.sum(torch.square(l_emb - r_emb), dim=-1, keepdim=True)
        l_neg_dis = squared_dist(l_emb, emb)
        r_neg_dis = squared_dist(r_emb, emb)

        del l_emb, r_emb


        l_mask_gama = self.gamma*l_score_mask
        r_mask_gama = self.gamma*r_score_mask


        l_loss = pos_dis - l_neg_dis + self.gamma
        # l_loss = pos_dis - l_neg_dis + l_mask_gama.unsqueeze(1)
        l_loss = l_loss * (1 - F.one_hot(l, num_classes=self.node_size) - F.one_hot(r, num_classes=self.node_size))
        r_loss = pos_dis - r_neg_dis + self.gamma
        # r_loss = pos_dis - r_neg_dis + r_mask_gama.unsqueeze(1)
        r_loss = r_loss * (1 - F.one_hot(l, num_classes=self.node_size) - F.one_hot(r, num_classes=self.node_size))

        del r_neg_dis, l_neg_dis

        r_loss = (r_loss - torch.mean(r_loss, dim=-1, keepdim=True).detach()) / torch.std(r_loss, dim=-1, unbiased=False, keepdim=True).detach()
        l_loss = (l_loss - torch.mean(l_loss, dim=-1, keepdim=True).detach()) / torch.std(l_loss, dim=-1, unbiased=False, keepdim=True).detach()

        lamb, tau = 20,8

        # print("l_loss:{}, r_loss:{}".format(l_loss,r_loss))
        l_loss = torch.logsumexp(lamb * l_loss + tau, dim=-1)
        r_loss = torch.logsumexp(lamb * r_loss + tau, dim=-1)
        # l_lamb = (lamb*l_mask_gama).unsqueeze(1)
        # r_lamb = (lamb*r_mask_gama).unsqueeze(1)
        # l_loss = torch.logsumexp(l_lamb * l_loss + tau, dim=-1)
        # r_loss = torch.logsumexp(r_lamb * r_loss + tau, dim=-1)

        # l_loss = l_loss*(l_score_mask)
        # r_loss = r_loss*(r_score_mask)
        # 无 mask 
        # Hits@1:  0.8729523809523809   Hits@5:  0.9460952380952381   Hits@10:  0.9608571428571429   MRR:  0.9060250021812597
        #  Hits@1:  0.8741904761904762   Hits@5:  0.9443809523809524   Hits@10:  0.9600952380952381   MRR:  0.9059946253973302
        #Hits@1:  0.8743809523809524   Hits@5:  0.9442857142857143   Hits@10:  0.9594285714285714   MRR:  0.9057425051569439
        # Hits@1:  0.8728571428571429   Hits@5:  0.944   Hits@10:  0.9598095238095238   MRR:  0.9052039600934775
        # Hits@1:  0.8734285714285714   Hits@5:  0.9454285714285714   Hits@10:  0.9613333333333334   MRR:  0.9057124026543754
        # Hits@1:  0.8746666666666667   Hits@5:  0.9436190476190476   Hits@10:  0.9581904761904761   MRR:  0.9057907475675094


        return torch.mean(l_loss + r_loss)



    def get_embeddings(self, index_a, index_b,turn):
        # forward
        out_feature_join,img_rel_feature,ent_rel_feature  = self.gcn_forward(turn)
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


