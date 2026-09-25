import dill
import sys
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
#
from hypergraph import Hypergraph, HGNNConv, NegativeHyperGraphConvDHG, CosineRepulsionHyperGraphConvDHG, FeatureDispersionHyperGraphConvDHG

class HGDR(nn.Module):
    def __init__(self, voc_size, ddi_adj_path, EHR_path, device=torch.device('cpu:0'), args=None):
        super().__init__()

        self.args = args
        self.voc_size = voc_size

        self.emb_dim = args.dim
        self.device = device

        with open(ddi_adj_path, 'rb') as f: # currently not used (ddi loss)
            ddi_adj = dill.load(f)
            self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        self.hg_d, self.hg_p, self.hg_m = self.construct_HG(EHR_path)
        #
        if self.args.hgmtype == 0:
            conv_class = NegativeHyperGraphConvDHG
        elif self.args.hgmtype == 1:
            conv_class = CosineRepulsionHyperGraphConvDHG
        elif self.args.hgmtype == 2:
            conv_class = FeatureDispersionHyperGraphConvDHG
        elif self.args.hgmtype == 3:
            conv_class = HGNNConv
        else:
            raise ValueError("hgmtype must be 0 (RHGNN), 1 (cosine), 2 (dispersion), or 3 (regular HGNN)")
        #
        if self.args.use_hgd:
            self.d_hgconv1 = HGNNConv(in_channels=self.emb_dim, out_channels=self.emb_dim, bias=True, use_bn=False, drop_rate=0.3, is_last=False)
        #
        if self.args.use_hgp:
            self.p_hgconv1 = HGNNConv(in_channels=self.emb_dim, out_channels=self.emb_dim, bias=True, use_bn=False, drop_rate=0.3, is_last=False)
        #
        if self.args.use_hgm:
            if self.args.hgmtype == 0:
                self.m_hgconv1 = conv_class(in_channels=self.emb_dim, out_channels=self.emb_dim)
            elif self.args.hgmtype in [1, 2]:
                self.m_hgconv1 = conv_class(self.emb_dim, self.emb_dim, self.args.hmw)
            elif self.args.hgmtype == 3:
                self.m_hgconv1 = HGNNConv(in_channels=self.emb_dim, out_channels=self.emb_dim, bias=True, use_bn=False, drop_rate=0.3, is_last=False)

        #
        #
        if self.args.use_hgd:
            self.d_hgconv = HGNNConv(in_channels=self.emb_dim, out_channels=self.emb_dim, bias=True, use_bn=False, drop_rate=0.3, is_last=True)
        #
        if self.args.use_hgp:
            self.p_hgconv = HGNNConv(in_channels=self.emb_dim, out_channels=self.emb_dim, bias=True, use_bn=False, drop_rate=0.3, is_last=True)
        #
        if self.args.use_hgm:
            if self.args.hgmtype == 0:
                self.m_hgconv = conv_class(in_channels=self.emb_dim, out_channels=self.emb_dim)
            elif self.args.hgmtype in [1, 2]:
                self.m_hgconv = conv_class(self.emb_dim, self.emb_dim, self.args.hmw)
            elif self.args.hgmtype == 3:
                self.m_hgconv = HGNNConv(in_channels=self.emb_dim, out_channels=self.emb_dim, bias=True, use_bn=False, drop_rate=0.3, is_last=True)

        # embeddings
        self.embeddings = nn.ModuleList(
            [nn.Embedding(self.voc_size[i]+1, self.emb_dim, padding_idx=self.voc_size[i]) for i in range(3)]) # D,P,M
        #
        self.dropout = nn.Dropout(0.3)
        
        # encode historical visits
        self.encoders = nn.ModuleList([nn.GRU(self.emb_dim, self.emb_dim, batch_first=True) for _ in range(2)])
        #
        self.query = nn.Sequential(
                nn.ReLU(), # nn.LeakyReLU(0.1),
                nn.Linear(2 * self.emb_dim, self.emb_dim)
        )
        # final mapping
        self.final_map = nn.Sequential(
            nn.Linear(self.voc_size[2], self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.voc_size[2]),
            nn.LayerNorm(self.voc_size[2])
        )

    def im2edges(self, im_np):
        e2v = []
        num_edges = im_np.shape[1]
        for j in range(num_edges):
            # 提取第 j 列（超边 j）中值 > 0 的行索引（节点索引）
            nodes_in_edge = np.where(im_np[:, j] > 0)[0].tolist()
            # 确保超边非空（DHG 不允许空超边）
            if len(nodes_in_edge) > 0:
                e2v.append(nodes_in_edge)
        return e2v

    def construct_HG(self,EHR_path):
        with open(EHR_path, 'rb') as f:
            all_mats = dill.load(f)
        # all_mats = (im_col_dC, im_col_pC, im_col_mC, im_col_DDI)
        # 0 1 2 3

        # HG for diag
        if self.args.use_hgd:
            im_D = all_mats[0][0:self.voc_size[0], :]
            d_e2v = self.im2edges(im_D)
            num_d_v = im_D.shape[0]
            print(f"诊断节点数 num_d_v: {num_d_v}, 诊断超边数: {len(d_e2v)}")
            hg_d = Hypergraph(num_v=num_d_v, e_list=d_e2v, device=self.device)
        else:
            hg_d = 0
        
        # HG for pro
        if self.args.use_hgp:
            im_P = all_mats[1][self.voc_size[0]:self.voc_size[0] + self.voc_size[1], :]
            p_e2v = self.im2edges(im_P)
            num_p_v = im_P.shape[0]
            print(f"治疗节点数 num_p_v: {num_p_v}, 治疗超边数: {len(p_e2v)}")
            hg_p = Hypergraph(num_v=num_p_v, e_list=p_e2v, device=self.device)
        else:
            hg_p = 0
        
        # HG for med
        if self.args.use_hgm:
            im_M = all_mats[2][self.voc_size[0] + self.voc_size[1]:, :]
            im_DDI = all_mats[3][self.voc_size[0] + self.voc_size[1]:, :]
            if self.args.use_hgm_ddi:
                im_Med = np.hstack([im_M, im_DDI])
            else:
                im_Med = im_M
            m_e2v = self.im2edges(im_Med)
            num_m_v = im_Med.shape[0]
            print(f"联合节点数 num_m_v: {num_m_v}, 联合超边数: {len(m_e2v)}")
            hg_m = Hypergraph(num_v=num_m_v, e_list=m_e2v, device=self.device)
        else:
            hg_m = 0
        
        #
        return hg_d, hg_p, hg_m

    def propagate(self):
        #
        if self.args.use_hgd:
            diag_feature = self.dropout(self.embeddings[0].weight[:self.voc_size[0]])
            diag_f1 = self.dropout(self.d_hgconv1(diag_feature, self.hg_d))
            diag_f = self.dropout(self.d_hgconv(diag_f1, self.hg_d)) / 4 + diag_f1 / 3 + diag_feature * 1 / 2
            # append the padding row back so returned features include the last item
            pad_d = self.embeddings[0].weight[self.voc_size[0]:self.voc_size[0]+1]  # (1, D)
            diag_f = torch.cat([diag_f, pad_d], dim=0)
        else:
            diag_f = self.embeddings[0].weight
        #
        if self.args.use_hgp:
            pro_feature = self.dropout(self.embeddings[1].weight[:self.voc_size[1]])
            pro_f1 = self.dropout(self.p_hgconv1(pro_feature, self.hg_p))
            pro_f = self.dropout(self.p_hgconv(pro_f1, self.hg_p)) / 4 + pro_f1 / 3 + pro_feature * 1 / 2
            # append the padding row back so returned features include the last item
            pad_p = self.embeddings[1].weight[self.voc_size[1]:self.voc_size[1]+1]  # (1, D)
            pro_f = torch.cat([pro_f, pad_p], dim=0)
        else:
            pro_f = self.embeddings[1].weight
        #
        if self.args.use_hgm:
            med_feature = self.dropout(self.embeddings[2].weight[:self.voc_size[2]])
            med_f1 = self.dropout(self.m_hgconv1(med_feature, self.hg_m))
            med_f = self.dropout(self.m_hgconv(med_f1, self.hg_m)) / 4 + med_f1 / 3 + med_feature * 1 / 2
            #med_f = self.dropout(self.m_hgconv(med_f1, self.hg_m))
            # append the padding row back so returned features include the last item
            pad_m = self.embeddings[2].weight[self.voc_size[2]:self.voc_size[2]+1]  # (1, D)
            med_f = torch.cat([med_f, pad_m], dim=0)
        else:
            med_f = self.embeddings[2].weight        
        #
        return diag_f, pro_f, med_f

    def forward(self, diags, pros, meds, visit_len):
        diag_feature, pro_feature, med_feature = self.propagate()

        # aggregate features using embedding-like lookup (diag_feature/pro_feature already include padding row)
        # expects diags/pros to be LongTensors of shape (B, 2, L) with padding index == self.voc_size[i]
        diag_embs = F.embedding(diags.long(), diag_feature)   # (B, 2, L, D)
        pro_embs  = F.embedding(pros.long(), pro_feature)     # (B, 2, L, D)

        # sum codes per visit (sum over the code dimension L) and apply dropout
        diag_emb_seq = self.dropout(diag_embs.sum(dim=2))     # (B, 2, D)
        pro_emb_seq  = self.dropout(pro_embs.sum(dim=2))  
        #
        o1, h1 = self.encoders[0](diag_emb_seq)
        o2, h2 = self.encoders[1](pro_emb_seq)  # o2 with shape (B, M, D)
        # NOTE: select by len
        # o1, o2 with shape (B, D)
        o1 = torch.stack([o1[i,visit_len[i]-1, :] for i in range(visit_len.shape[0])]) #每个batch挑一个o1或者o2
        o2 = torch.stack([o2[i,visit_len[i]-1, :] for i in range(visit_len.shape[0])])
        #
        patient_representations = torch.cat([o1, o2], dim=-1)  # (B, dim*2)
        #
        query = self.query(patient_representations)
        #
        norm_of_query = torch.norm(query, 2, 1, keepdim=True) + 1e-12  # 防止范数接近0
        normed_query = query / norm_of_query
        #
        valid_med_feature = med_feature[:self.voc_size[2], :]  # exclude padding row
        valid_med_feature = valid_med_feature / (valid_med_feature.norm(p=2, dim=1, keepdim=True) + 1e-12)
        # compute inner product
        innerP = normed_query @ valid_med_feature.T

        ###############################################################
        ###############################################################
        # # make use of the medications in the first visit if and only if there are two visits in total
        # embed with base medication embeddings
        med_embs = F.embedding(meds.long(), med_feature)  # (B, 2, Lm, D)
        med_emb_seq = self.dropout(med_embs.sum(dim=2))  # (B, 2, D)
        # take first visit (previous visit when visit_len==2)
        med_prev = med_emb_seq[:, 0, :]  # (B, D)
        # mask out samples without a previous visit 
        # i.e., for those samples with visit_len=1, their first visits are also the current visits, 
        # so we do not use the medications from them as prior knowledge because they are not the targets to predict
        mask = (visit_len.to(med_prev.device) >= 2).unsqueeze(1).float()
        med_prev = med_prev * mask
        # normalize and compute prior scores over meds
        med_prev_norm = (med_prev / (med_prev.norm(p=2, dim=1, keepdim=True) + 1e-12)) * mask  # (B, D)
        med_prior_scores = med_prev_norm @ valid_med_feature.T  # (B, Nmed)
        # combine prior with model inner product; alpha controls contribution
        # Only mix in med_prior_scores for samples that have a previous visit (mask==1).
        # mask has shape (B, 1). Expand to (B, Nmed) so we can apply per-sample weighting.
        mask_exp = mask.expand_as(med_prior_scores)  # (B, Nmed)
        # compute per-sample mixture weight: for samples with mask=1 use hmw, else 0
        per_sample_alpha = mask_exp * 0.5
        # convex combination per sample: (1 - alpha)*innerP + alpha*prior
        innerP = (1.0 - per_sample_alpha) * innerP + per_sample_alpha * med_prior_scores
        ###############################################################
        ###############################################################
        #
        result = self.final_map(innerP)
        #
        if self.args.ddi: 
            neg_pred_prob = torch.sigmoid(result)
            tmp_left = neg_pred_prob.unsqueeze(2)  # (B, Nmed, 1)
            tmp_right = neg_pred_prob.unsqueeze(1)  # (B, 1, Nmed)
            neg_pred_prob = torch.matmul(tmp_left, tmp_right)  # (N, Nmed, Nmed)
            batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum() # ddi loss
        else:
            batch_neg = 0   # ddi_loss

        return result, batch_neg #, se_loss

    def evaluate(self, diags, pros, meds, visit_len):
        # Use HG-smoothed features (propagate) for evaluation as in forward
        diag_feature, pro_feature, med_feature = self.propagate()

        # embedding-like lookup using hg-smoothed features (these include padding row)
        diag_embs = F.embedding(diags.long(), diag_feature)   # (B, 2, L, D)
        pro_embs  = F.embedding(pros.long(), pro_feature)     # (B, 2, L, D)

        # sum codes per visit and apply dropout
        diag_emb_seq = self.dropout(diag_embs.sum(dim=2))     # (B, 2, D)
        pro_emb_seq  = self.dropout(pro_embs.sum(dim=2))
        #
        o1, h1 = self.encoders[0](diag_emb_seq)
        o2, h2 = self.encoders[1](pro_emb_seq)
        # select last valid output per sample
        o1 = torch.stack([o1[i, visit_len[i]-1, :] for i in range(visit_len.shape[0])])
        o2 = torch.stack([o2[i, visit_len[i]-1, :] for i in range(visit_len.shape[0])])

        patient_representations = torch.cat([o1, o2], dim=-1)  # (B, dim*2)
        query = self.query(patient_representations)  # (B, dim)

        norm_of_query = torch.norm(query, 2, 1, keepdim=True) + 1e-12
        normed_query = query / norm_of_query

        valid_med_feature = med_feature[:self.voc_size[2], :]  # exclude padding row
        valid_med_feature = valid_med_feature / (valid_med_feature.norm(p=2, dim=1, keepdim=True) + 1e-12)
        innerP = normed_query @ valid_med_feature.T

        ###############################################################
        ###############################################################
        # # make use of the medications in the first visit if and only if there are two visits in total
        # embed with base medication embeddings
        med_embs = F.embedding(meds.long(), med_feature)  # (B, 2, Lm, D)
        med_emb_seq = self.dropout(med_embs.sum(dim=2))  # (B, 2, D)
        # take first visit (previous visit when visit_len==2)
        med_prev = med_emb_seq[:, 0, :]  # (B, D)
        # mask out samples without a previous visit 
        # i.e., for those samples with visit_len=1, their first visits are also the current visits, 
        # so we do not use the medications from them as prior knowledge because they are not the targets to predict
        mask = (visit_len.to(med_prev.device) >= 2).unsqueeze(1).float()
        med_prev = med_prev * mask
        # normalize and compute prior scores over meds
        med_prev_norm = (med_prev / (med_prev.norm(p=2, dim=1, keepdim=True) + 1e-12)) * mask  # (B, D)
        med_prior_scores = med_prev_norm @ valid_med_feature.T  # (B, Nmed)
        # combine prior with model inner product; alpha controls contribution
        # Only mix in med_prior_scores for samples that have a previous visit (mask==1).
        mask_exp = mask.expand_as(med_prior_scores)  # (B, Nmed)
        per_sample_alpha = mask_exp * 0.5
        innerP = (1.0 - per_sample_alpha) * innerP + per_sample_alpha * med_prior_scores
        ###############################################################
        ###############################################################

        result = self.final_map(innerP)
        return result
