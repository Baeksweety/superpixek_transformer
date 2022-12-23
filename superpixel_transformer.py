from timm.models.vision_transformer import VisionTransformer
import timm.models.vision_transformer
import skimage.io as io
import argparse
import joblib
import copy
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import skimage.io as io
from timm.models.layers import drop_path, to_2tuple, trunc_normal_,PatchEmbed
from torch_geometric.nn import global_mean_pool,global_max_pool,GlobalAttention,dense_diff_pool,global_add_pool,TopKPooling,ASAPooling,SAGPooling
from torch_geometric.nn import GCNConv,ChebConv,SAGEConv,GraphConv,LEConv,LayerNorm,GATConv
import torch
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
import torch.nn as nn
torch.set_num_threads(8)
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from functools import partial
from block_utils import Block
from torch_geometric.data import Data as geomData

try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message



#设置全局变量？模型输入值为训练所需的wsi的名字
superpixel_graph_path = '/data13/yanhe/miccai/graph_file/tcga_lihc_all/superpixel_num_300'
inter_graph_path = '/data13/yanhe/miccai/super_pixel/graph_file/tcga_lihc_all/superpixel_num_300'
cluster_info_path = '/data13/yanhe/miccai/codebook/cluster_info/tcga_lihc_all/superpixel300_cluster16/all_fold'

class Intra_GCN(nn.Module):
    def __init__(self,in_feats,n_hidden,out_feats,drop_out_ratio=0.2,mpool_method="global_mean_pool"):
        super(Intra_GCN,self).__init__()        
        self.conv1= GCNConv(in_channels=in_feats,out_channels=n_hidden)          
        self.conv2= GCNConv(in_channels=n_hidden,out_channels=out_feats)
        
        self.relu = torch.nn.ReLU() 
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout=nn.Dropout(p=drop_out_ratio)
        self.softmax = nn.Softmax(dim=-1)
        
        if mpool_method == "global_mean_pool":
            self.mpool = global_mean_pool 
        elif mpool_method == "global_max_pool":
            self.mpool = global_max_pool 
        elif mpool_method == "global_att_pool":
            att_net=nn.Sequential(nn.Linear(out_classes, out_classes//2), nn.ReLU(), nn.Linear(out_classes//2, 1))     
            self.mpool = GlobalAttention(att_net)  
        self.norm = LayerNorm(in_feats)      
        self.norm2 = LayerNorm(out_feats)
        self.norm1 = LayerNorm(n_hidden)
        
    def forward(self,data):
        x=data.x
        edge_index = data.edge_index 
        
        x = self.norm(x)
        x = self.conv1(x,edge_index)
        x = self.relu(x)  
        # x = self.sigmoid(x)
        # x = self.norm(x)
        x = self.norm1(x)
        x = self.dropout(x)

        x = self.conv2(x,edge_index)
        x = self.relu(x)  
        # x = self.sigmoid(x)
        # x = self.norm(x)   
        x = self.norm2(x)    
        x = self.dropout(x)
        # print(x)

        batch = torch.zeros(len(x),dtype=torch.long).to(device)
        x = self.mpool(x,batch)
        
        fea = x

        return fea

class Inter_GCN(nn.Module):
    def __init__(self,in_feats,n_hidden,out_feats,drop_out_ratio=0.2,mpool_method="global_mean_pool"):
        super(Inter_GCN,self).__init__()        
        self.conv1= GCNConv(in_channels=in_feats,out_channels=n_hidden)          
        self.conv2= GCNConv(in_channels=n_hidden,out_channels=out_feats)
        
        self.relu = torch.nn.ReLU() 
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout=nn.Dropout(p=drop_out_ratio)
        self.softmax = nn.Softmax(dim=-1)
        
        if mpool_method == "global_mean_pool":
            self.mpool = global_mean_pool 
        elif mpool_method == "global_max_pool":
            self.mpool = global_max_pool 
        elif mpool_method == "global_att_pool":
            att_net=nn.Sequential(nn.Linear(out_classes, out_classes//2), nn.ReLU(), nn.Linear(out_classes//2, 1))     
            self.mpool = GlobalAttention(att_net)        
        self.norm = LayerNorm(in_feats)
        self.norm2 = LayerNorm(out_feats)
        self.norm1 = LayerNorm(n_hidden)
        
    def forward(self,data):
        x=data.x.float()
        edge_index = data.edge_index 
        
        x = self.norm(x)
        x = self.conv1(x,edge_index)
        x = self.relu(x)  
        # x = self.sigmoid(x)
        x = self.norm1(x)
        x = self.dropout(x)

        x = self.conv2(x,edge_index)
        x = self.relu(x)  
        # x = self.sigmoid(x)
        x = self.norm2(x)       
        x = self.dropout(x)

        # batch = torch.zeros(len(x),dtype=torch.long).to(device)
        # x = self.mpool(x,batch)
        
        fea = x
        # print(x.shape)

        return fea

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, num_patches=100,no_embed_class=False,class_token=True, depth=1,drop_path_rate=0.,mlp_ratio=4.,qkv_bias=True,init_values=None,drop_rate=0.,attn_drop_rate=0.,norm_layer=None,act_layer=None,**kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        embed_dim = kwargs['embed_dim']
        self.patch_embed = nn.Linear(embed_dim,embed_dim)
        num_patches = num_patches
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        
        self.num_prefix_tokens = 1 if class_token else 0
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        #可以导出attention score的block
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=kwargs['num_heads'],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)])

    def forward_features(self, x):
        x = self.patch_embed(x)
        #print(x.shape)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return torch.sigmoid(x)

    def get_attention_weights(self):
        return [block.get_attention_weights() for block in self.blocks]

class Superpixel_Vit(nn.Module):
    def __init__(self,in_feats=1024,n_hidden=1024,out_feats=1024,vw_num=16,feat_dim=1024,num_classes=1,depth=1,num_heads = 16):
        super(Superpixel_Vit, self).__init__()

        self.vw_num = vw_num
        self.feat_dim = feat_dim

        #intra-graph
        self.gcn1 = Intra_GCN(in_feats=in_feats,n_hidden=n_hidden,out_feats=out_feats)

        #inter-graph
        self.gcn2 = Inter_GCN(in_feats=in_feats,n_hidden=n_hidden,out_feats=out_feats)
        self.vit = VisionTransformer(num_patches = vw_num,num_classes = num_classes, embed_dim = feat_dim,depth = depth,num_heads = num_heads)

    
    def intra_superpixel_graph(self,superpixel_graph_path:str,slidename):
        slide_path=os.path.join(superpixel_graph_path,slidename)
        superpixels_fea={}
        for superpixel_name in os.listdir(slide_path):
            superpixel_path=os.path.join(slide_path,superpixel_name)
            # print(superpixel_path)
            superpixel_graph=torch.load(superpixel_path)
            superpixel_graph=superpixel_graph.to(device)
            superpixel_fea=self.gcn1(superpixel_graph)
            superpixels_fea[superpixel_name[:-3]]=superpixel_fea
        return superpixels_fea

    def inter_superpixel_graph(self,inter_graph_path,superpixels_feature,slidename):
        graph_path = os.path.join(inter_graph_path,slidename+'.pt')
        g = torch.load(graph_path)
            #封装节点特征
        feat_all = torch.Tensor()
        for index in range(g.ndata['centroid'].shape[0]):
            superpixel_value = index+1
            superpixel_name = 'superpixel_'+str(superpixel_value)
            if superpixel_name in superpixels_feature.keys():
                feat = superpixels_feature[superpixel_name].detach().cpu()
            else:
                print('False!')
            if feat_all.shape[0] == 0:
                feat_all = feat
            else:
                feat_all = torch.cat((feat_all,feat),dim=0)
            #封装边的连接信息
        edge_index = torch.Tensor()
        edge_index = g.edges()[0].unsqueeze(0)
        edge_index = torch.cat((edge_index,g.edges()[1].unsqueeze(0)),dim=0)
        G = geomData(x = feat_all,
                        edge_index = edge_index)
        G = G.to(device)
        slide_fea = self.gcn2(G)
        superpixel_fea={}
        for index in range(slide_fea.shape[0]):
            fea = slide_fea[index].unsqueeze(0)
            superpixel_value = index+1
            # superpixel_name = 'superpixel_'+str(superpixel_value)
            superpixel_fea[superpixel_value] = fea
        return superpixel_fea

    def mean_feature(self,superpixel_features,cluster_info_path,slidename):
        mask=np.zeros((self.vw_num,self.feat_dim))
        mask=torch.tensor(mask).to(device)
        superpixel_cluster_path = os.path.join(cluster_info_path,slidename+'.pth')
        cluster_info = torch.load(superpixel_cluster_path)
        for vw in range(self.vw_num):
            fea_all=torch.Tensor().to(device)
            for superpixel_value in cluster_info.keys(): 
                if cluster_info[superpixel_value]['cluster']==vw:
                    if fea_all.shape[0]==0:
                        fea_all=superpixel_features[superpixel_value]
                    else:
                        fea_all=torch.cat((fea_all,superpixel_features[superpixel_value]),dim=0)
            if fea_all.shape[0]!=0:
                fea_avg=torch.mean(fea_all,axis=0)
#             print('fea_avg shape:{}'.format(fea_avg.shape))
                mask[vw]=fea_avg
        return mask


    def forward(self,slidename):
        #intra-graph
        superpixels_fea = self.intra_superpixel_graph(superpixel_graph_path,slidename)
        #inter-graph
        superpixels_fea = self.inter_superpixel_graph(inter_graph_path,superpixels_fea,slidename)
        #final-fea
        fea = self.mean_feature(superpixels_fea,cluster_info_path,slidename) #【16，1024】大小
        fea = fea.unsqueeze(0)  #[1,16,1024]大小
        # print(fea.shape)
        #输入vit模型
        fea = fea.float()
        out = self.vit(fea)
        return out

    def get_attention_weights(self):
        return self.vit.get_attention_weights()
