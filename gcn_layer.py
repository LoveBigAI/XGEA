import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
# from sklearn.decomposition import PCA


class NR_GraphAttention(nn.Module):
    def __init__(self,
                 node_size,
                 rel_size,
                 triple_size,
                 node_dim,
                 depth=1,
                 attn_heads=1,
                 attn_heads_reduction='concat',
                 use_bias=False):
        super(NR_GraphAttention, self).__init__()

        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.node_dim = node_dim
        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction
        self.activation = torch.nn.Tanh()
        self.use_bias = use_bias
        self.depth = depth
        self.attn_kernels = nn.ParameterList()

        # create parameters
        feature = self.node_dim*(self.depth+1)

        # gate
        self.gate = torch.nn.Linear(feature, feature)
        torch.nn.init.xavier_uniform_(self.gate.weight)
        torch.nn.init.zeros_(self.gate.bias)

        

        # proxy node
        self.proxy = torch.nn.Parameter(data=torch.empty(64, feature, dtype=torch.float32))
        torch.nn.init.xavier_uniform_(self.proxy)

        # attention kernel
        for l in range(self.depth):
            attn_kernel = torch.nn.Parameter(data=torch.empty(self.node_dim, 1, dtype=torch.float32))
            torch.nn.init.xavier_uniform_(attn_kernel)
            self.attn_kernels.append(attn_kernel)

    def forward(self, inputs):
        outputs = []
        features = inputs[0]
        rel_emb = inputs[1]
        adj = inputs[2]
        r_index = inputs[3]
        r_val = inputs[4]

        features = self.activation(features)
        outputs.append(features)

        for l in range(self.depth):
            attention_kernel = self.attn_kernels[l]
            # matrix shape: [N_tri x N_rel]
            tri_rel = torch.sparse_coo_tensor(indices=r_index, values=r_val,
                                              size=[self.triple_size, self.rel_size], dtype=torch.float32)
            # shape: [N_tri x dim]
            tri_rel = torch.sparse.mm(tri_rel, rel_emb)
            # shape: [N_tri x dim]
            neighs = features[adj[1, :].long()]

            tri_rel = F.normalize(tri_rel, dim=1, p=2)
            neighs = neighs - 2*torch.sum(neighs*tri_rel, dim=1, keepdim=True)*tri_rel

            att = torch.squeeze(torch.mm(tri_rel, attention_kernel), dim=-1)
            att = torch.sparse_coo_tensor(indices=adj, values=att, size=[self.node_size, self.node_size])
            att = torch.sparse.softmax(att, dim=1)

            new_features = scatter_sum(src=neighs * torch.unsqueeze(att.coalesce().values(), dim=-1), dim=0,
                                       index=adj[0,:].long())

            features = self.activation(new_features)
            outputs.append(features)

        outputs = torch.cat(outputs, dim=-1)

        proxy_att = torch.mm(F.normalize(outputs, p=2, dim=-1), torch.transpose(F.normalize(self.proxy, p=2, dim=-1), 0, 1))
        proxy_att = F.softmax(proxy_att, dim=-1)
        proxy_feature = outputs - torch.mm(proxy_att, self.proxy)

        gate_rate = torch.sigmoid(self.gate(proxy_feature))

        # final_outputs = gate_rate * outputs + (1-gate_rate) * proxy_feature
        final_outputs = outputs

        return final_outputs
    


class NR_GraphAttentionCross(nn.Module):
    def __init__(self,
                 node_size,
                 rel_size,
                 triple_size,
                 node_dim,
                 depth=1,
                 attn_heads=1,
                 attn_heads_reduction='concat',
                 use_bias=False):
        super(NR_GraphAttentionCross, self).__init__()

        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.node_dim = node_dim
        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction
        self.activation = torch.nn.Tanh()
        self.use_bias = use_bias
        self.depth = depth
        self.attn_kernels = nn.ParameterList()
        self.attn_kernels_ent = nn.ParameterList()

        # create parameters
        feature = self.node_dim*(self.depth+1)

        # gate
        self.gate = torch.nn.Linear(feature, feature)
        torch.nn.init.xavier_uniform_(self.gate.weight)
        torch.nn.init.zeros_(self.gate.bias)

        # proxy node
        self.proxy = torch.nn.Parameter(data=torch.empty(64, feature, dtype=torch.float32))
        torch.nn.init.xavier_uniform_(self.proxy)

        self.concat_dim = torch.nn.Linear(self.node_dim, self.node_dim)
        torch.nn.init.xavier_uniform_(self.concat_dim.weight)
        torch.nn.init.zeros_(self.concat_dim.bias)


        self.cat_embedding = nn.Embedding(self.node_dim*2, self.node_dim)
        torch.nn.init.xavier_uniform_(self.cat_embedding.weight)

        self.trans_embedding = nn.Embedding(self.node_dim, self.node_dim)
        torch.nn.init.xavier_uniform_(self.trans_embedding.weight)

        self.start_train =True




        # attention kernel
        for l in range(self.depth):
            attn_kernel = torch.nn.Parameter(data=torch.empty(self.node_dim, 1, dtype=torch.float32))
            attn_kernel_ent = torch.nn.Parameter(data=torch.empty(self.node_dim, 1, dtype=torch.float32))
            torch.nn.init.xavier_uniform_(attn_kernel)
            torch.nn.init.xavier_uniform_(attn_kernel_ent)
            self.attn_kernels.append(attn_kernel)
            self.attn_kernels_ent.append(attn_kernel_ent)

    def forward(self, inputs):
        outputs = []
        features = inputs[0]
        rel_emb = inputs[1]
        adj = inputs[2]
        r_index = inputs[3]
        r_val = inputs[4]
        features_c = inputs[5]
        Fussion = inputs[6]

        features = self.activation(features)
        outputs.append(features)

        for l in range(self.depth):
            attention_kernel = self.attn_kernels[l]
            attention_kernel_ent = self.attn_kernels_ent[l]
            # matrix shape: [N_tri x N_rel]
            tri_rel = torch.sparse_coo_tensor(indices=r_index, values=r_val,
                                              size=[self.triple_size, self.rel_size], dtype=torch.float32)
            # shape: [N_tri x dim]
            tri_rel = torch.sparse.mm(tri_rel, rel_emb)

            # concat_ent = self.concat_dim(torch.cat([features_c[adj[0, :].long()],features_c[adj[1, :].long()]], dim=-1))
            # concat_ent =torch.abs(features_c[adj[0, :].long()]+features_c[adj[1, :].long()])
            # concat_ent = torch.mm(torch.cat([features_c[adj[0, :].long()],features_c[adj[1, :].long()]], dim=-1),self.cat_embedding.weight)
            
            concat_fea =torch.cat([features_c[adj[0, :].long()],features_c[adj[1, :].long()]], dim=-1).unsqueeze(0)

            U,S,V = torch.pca_lowrank(concat_fea,center=True,q=self.node_dim)
            concat_ent = torch.matmul(concat_fea,V[:,:,:self.node_dim]).squeeze()

            concat_ent = torch.matmul(concat_ent,torch.diag_embed(torch.pow(S.squeeze()+1e-5,-0.5)))
            concat_ent[torch.isinf(concat_ent)]=0

            # shape: [N_tri x dim]
            neighs = features[adj[1, :].long()]

            tri_rel = F.normalize(tri_rel, dim=1, p=2)
            concat_ent = F.normalize(concat_ent, dim=1, p=2)
            neighs_rel = neighs - 2*torch.sum(neighs*tri_rel, dim=1, keepdim=True)*tri_rel
            neighs_ent = neighs - 2*torch.sum(neighs*concat_ent, dim=1, keepdim=True)*concat_ent
            # neighs_ent = torch.mm(neighs,self.trans_embedding.weight)

            att = torch.squeeze(torch.mm(tri_rel, attention_kernel), dim=-1)
            att = torch.sparse_coo_tensor(indices=adj, values=att, size=[self.node_size, self.node_size])
            att = torch.sparse.softmax(att, dim=1)

            att_ent = torch.squeeze(torch.mm(concat_ent, attention_kernel_ent), dim=-1)
            att_ent = torch.sparse_coo_tensor(indices=adj, values=att_ent, size=[self.node_size, self.node_size])
            att_ent = torch.sparse.softmax(att_ent, dim=1)

            # att = att_ent*0.1 +att
            new_features = scatter_sum(src=neighs_rel * torch.unsqueeze(att.coalesce().values(), dim=-1), dim=0,
                                       index=adj[0,:].long())
            

            alpha=0.1
            new_features =new_features+  alpha*scatter_sum(src=neighs_ent * torch.unsqueeze(att_ent.coalesce().values(), dim=-1), dim=0,
                                        index=adj[0,:].long())


            # if not self.start_train and self.training:
            #     new_features =new_features+  alpha*scatter_sum(src=neighs_ent * torch.unsqueeze(att_ent.coalesce().values(), dim=-1), dim=0,
            #                             index=adj[0,:].long())
            # else:
            #     self.start_train =False
            

            features = self.activation(new_features)
            outputs.append(features)

        outputs = torch.cat(outputs, dim=-1)

        proxy_att = torch.mm(F.normalize(outputs, p=2, dim=-1), torch.transpose(F.normalize(self.proxy, p=2, dim=-1), 0, 1))
        proxy_att = F.softmax(proxy_att, dim=-1)
        proxy_feature = outputs - torch.mm(proxy_att, self.proxy)

        gate_rate = torch.sigmoid(self.gate(proxy_feature))

        # final_outputs = gate_rate * outputs + (1-gate_rate) * proxy_feature
        final_outputs = outputs

        return final_outputs
    

class NR_GraphAttentionMu(nn.Module):
    def __init__(self,
                 node_size,
                 rel_size,
                 triple_size,
                 node_dim,
                 depth=1,
                 attn_heads=1,
                 attn_heads_reduction='concat',
                 use_bias=False):
        super(NR_GraphAttentionMu, self).__init__()

        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.node_dim = node_dim
        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction
        self.activation = torch.nn.Tanh()
        self.use_bias = use_bias
        self.depth = depth
        self.attn_kernels = nn.ParameterList()
        self.attn_kernels_ent = nn.ParameterList()

        # create parameters
        feature = self.node_dim*(self.depth+1)

        # gate
        self.gate = torch.nn.Linear(feature, feature)
        torch.nn.init.xavier_uniform_(self.gate.weight)
        torch.nn.init.zeros_(self.gate.bias)

        # proxy node
        self.proxy = torch.nn.Parameter(data=torch.empty(64, feature, dtype=torch.float32))
        torch.nn.init.xavier_uniform_(self.proxy)

        self.concat_dim = torch.nn.Linear(self.node_dim, self.node_dim)
        torch.nn.init.xavier_uniform_(self.concat_dim.weight)
        torch.nn.init.zeros_(self.concat_dim.bias)


        self.cat_embedding = nn.Embedding(self.node_dim*2, self.node_dim)
        torch.nn.init.xavier_uniform_(self.cat_embedding.weight)

        self.trans_embedding = nn.Embedding(self.node_dim, self.node_dim)
        torch.nn.init.xavier_uniform_(self.trans_embedding.weight)

        self.start_train =True




        # attention kernel
        for l in range(self.depth):
            attn_kernel = torch.nn.Parameter(data=torch.empty(self.node_dim, 1, dtype=torch.float32))
            attn_kernel_ent = torch.nn.Parameter(data=torch.empty(self.node_dim, 1, dtype=torch.float32))
            torch.nn.init.xavier_uniform_(attn_kernel)
            torch.nn.init.xavier_uniform_(attn_kernel_ent)
            self.attn_kernels.append(attn_kernel)
            self.attn_kernels_ent.append(attn_kernel_ent)

    def forward(self, inputs):
        outputs = []
        features = inputs[0]
        rel_emb = inputs[1]
        adj = inputs[2]
        r_index = inputs[3]
        r_val = inputs[4]
        features_c = inputs[5]


        features = self.activation(features)
        outputs.append(features)

        for l in range(self.depth):
            # attention_kernel = self.attn_kernels[l]
            attention_kernel_ent = self.attn_kernels_ent[l]
            # matrix shape: [N_tri x N_rel]
            # tri_rel = torch.sparse_coo_tensor(indices=r_index, values=r_val,
            #                                   size=[self.triple_size, self.rel_size], dtype=torch.float32)
            # # shape: [N_tri x dim]
            # tri_rel = torch.sparse.mm(tri_rel, rel_emb)

            # concat_ent = self.concat_dim(torch.cat([features_c[adj[0, :].long()],features_c[adj[1, :].long()]], dim=-1))
            concat_ent =torch.abs(features_c[adj[0, :].long()]+features_c[adj[1, :].long()])
            # concat_ent = torch.mm(torch.cat([features_c[adj[0, :].long()],features_c[adj[1, :].long()]], dim=-1),self.cat_embedding.weight)
            
            # concat_fea =torch.cat([features_c[adj[0, :].long()],features_c[adj[1, :].long()]], dim=-1).unsqueeze(0)

            # U,S,V = torch.pca_lowrank(concat_fea,center=True,q=self.node_dim)
            # concat_ent = torch.matmul(concat_fea,V[:,:,:self.node_dim]).squeeze()

            # concat_ent = torch.matmul(concat_ent,torch.diag_embed(torch.pow(S.squeeze()+1e-5,-0.5)))
            concat_ent[torch.isinf(concat_ent)]=0

            # shape: [N_tri x dim]
            neighs = features[adj[1, :].long()]

            # tri_rel = F.normalize(tri_rel, dim=1, p=2)
            concat_ent = F.normalize(concat_ent, dim=1, p=2)
            # neighs_rel = neighs - 2*torch.sum(neighs*tri_rel, dim=1, keepdim=True)*tri_rel
            neighs_ent = neighs - 2*torch.sum(neighs*concat_ent, dim=1, keepdim=True)*concat_ent
            # neighs_ent = torch.mm(neighs,self.trans_embedding.weight)

            # att = torch.squeeze(torch.mm(tri_rel, attention_kernel), dim=-1)
            # att = torch.sparse_coo_tensor(indices=adj, values=att, size=[self.node_size, self.node_size])
            # att = torch.sparse.softmax(att, dim=1)

            att_ent = torch.squeeze(torch.mm(concat_ent, attention_kernel_ent), dim=-1)
            att_ent = torch.sparse_coo_tensor(indices=adj, values=att_ent, size=[self.node_size, self.node_size])
            att_ent = torch.sparse.softmax(att_ent, dim=1)

            # att = att_ent*0.1 +att
            # new_features = scatter_sum(src=neighs_rel * torch.unsqueeze(att.coalesce().values(), dim=-1), dim=0,
            #                            index=adj[0,:].long())
            

    
            new_features = scatter_sum(src=neighs_ent * torch.unsqueeze(att_ent.coalesce().values(), dim=-1), dim=0,
                                        index=adj[0,:].long())


            # if not self.start_train and self.training:
            #     new_features =new_features+  alpha*scatter_sum(src=neighs_ent * torch.unsqueeze(att_ent.coalesce().values(), dim=-1), dim=0,
            #                             index=adj[0,:].long())
            # else:
            #     self.start_train =False
            

            features = self.activation(new_features)
            features_c = features.clone()
            outputs.append(features)

        outputs = torch.cat(outputs, dim=-1)

        proxy_att = torch.mm(F.normalize(outputs, p=2, dim=-1), torch.transpose(F.normalize(self.proxy, p=2, dim=-1), 0, 1))
        proxy_att = F.softmax(proxy_att, dim=-1)
        proxy_feature = outputs - torch.mm(proxy_att, self.proxy)

        gate_rate = torch.sigmoid(self.gate(proxy_feature))

        # final_outputs = gate_rate * outputs + (1-gate_rate) * proxy_feature
        final_outputs = outputs

        return final_outputs
    



