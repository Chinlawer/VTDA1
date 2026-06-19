import torch
import torch.nn as nn
from .prune_layer import SubnetLinear


class BatchNormPoint(nn.Module):
    def __init__(self, feat_size):
        super().__init__()
        self.feat_size = feat_size
        self.bn = nn.BatchNorm1d(feat_size)

    def forward(self, x):
        assert len(x.shape) == 3
        s1, s2, s3 = x.shape[0], x.shape[1], x.shape[2]
        assert s3 == self.feat_size
        x = x.reshape(s1 * s2, self.feat_size)
        x = self.bn(x)
        return x.reshape(s1, s2, s3)


class SimplifiedAdapter(nn.Module):
    def __init__(self, num_views=10, in_features=512, sparsity=0.5, bias=False):
        super().__init__()

        self.num_views = num_views
        self.in_features = in_features
        self.adapter_ratio = 0.6
        self.fusion_init = 0.5
        self.dropout = 0.075
        self.sparsity = sparsity
        
        self.fusion_ratio = nn.Parameter(torch.tensor([self.fusion_init] * self.num_views), requires_grad=True)
        
        # self.global_f = nn.Sequential(
        #         BatchNormPoint(self.in_features),
        #         nn.Dropout(self.dropout),
        #         nn.Flatten(),
        #         SubnetLinear(in_features=self.in_features * self.num_views,
        #                   out_features=self.in_features, sparsity=self.sparsity, bias=False),
        #         nn.BatchNorm1d(self.in_features),
        #         nn.ReLU(),
        #         nn.Dropout(self.dropout),
        #         SubnetLinear(in_features=self.in_features, out_features=self.in_features, sparsity=self.sparsity, bias=False))
        
        self.global_f_BN1 = BatchNormPoint(self.in_features)
        self.global_f_DR1 = nn.Dropout(self.dropout)
        self.global_f_Fl = nn.Flatten()
        self.global_f_1 = SubnetLinear(in_features=self.in_features * self.num_views, out_features=self.in_features, sparsity=self.sparsity, bias=False)
        # self.global_f_1 = nn.Linear(in_features=self.in_features * self.num_views, out_features=self.in_features, bias=False)
        self.global_f_BN2 = nn.BatchNorm1d(self.in_features)
        self.global_f_Re = nn.ReLU()
        self.global_f_DR2 = nn.Dropout(self.dropout)
        self.global_f_2 = SubnetLinear(in_features=self.in_features, out_features=self.in_features, sparsity=self.sparsity, bias=False)
        # self.global_f_2 = nn.Linear(in_features=self.in_features, out_features=self.in_features, bias=False)

    def forward(self, feat, weight_mask1, bias_mask1, weight_mask2, bias_mask2, mode='train', current_sparsity=None):
    # def forward(self, feat):
        img_feat = feat.reshape(-1, self.num_views, self.in_features)
        img_point_feat1 = self.global_f_BN1(img_feat * self.fusion_ratio.reshape(1, -1, 1))
        img_point_feat2 = self.global_f_DR1(img_point_feat1)
        img_point_feat3 = self.global_f_Fl(img_point_feat2)
        img_point_feat4 = self.global_f_1(img_point_feat3, weight_mask1, bias_mask1, mode=mode, current_sparsity=current_sparsity)
        # img_point_feat4 = self.global_f_1(img_point_feat3)
        img_point_feat5 = self.global_f_BN2(img_point_feat4)
        img_point_feat6 = self.global_f_Re(img_point_feat5)
        img_point_feat7 = self.global_f_DR2(img_point_feat6)
        img_point_feat = self.global_f_2(img_point_feat7, weight_mask2, bias_mask2, mode=mode, current_sparsity=current_sparsity)
        # img_point_feat = self.global_f_2(img_point_feat7)
        
        # Global feature
        return img_point_feat
    
class SimplifiedAdapter2(nn.Module):
    def __init__(self, num_views=10, in_features=512, Out_features=768):
        super().__init__()

        self.num_views = num_views
        self.in_features = in_features
        self.Out_features = Out_features
        self.adapter_ratio = 0.6
        self.fusion_init = 0.5
        self.dropout = 0.075
        
        self.fusion_ratio = nn.Parameter(torch.tensor([self.fusion_init] * self.num_views), requires_grad=True)
        
        self.global_f = nn.Sequential(
                BatchNormPoint(self.in_features),
                nn.Dropout(self.dropout),
                nn.Flatten(),
                nn.Linear(in_features=self.in_features * self.num_views,
                          out_features=self.in_features),
                nn.BatchNorm1d(self.in_features),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(in_features=self.in_features, out_features=self.Out_features))

    def forward(self, feat):
        img_feat = feat.reshape(-1, self.num_views, self.in_features)
        
        # Global feature
        return self.global_f(img_feat * self.fusion_ratio.reshape(1, -1, 1))
