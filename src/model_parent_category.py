import dill
import sys
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
#
# Parent-category baseline: no hypergraph operators are used.

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

        # ------------------------------------------------------------------
        # Simple parent-category baseline
        # ------------------------------------------------------------------
        # This baseline uses the SAME taxonomy information as TDH-DR but does
        # not perform hypergraph message passing. Each entity receives a
        # learnable embedding for its parent category, and the final entity
        # representation is a fixed convex combination of its own embedding
        # and its parent-category embedding.
        #
        # x_i^* = (1 - w) * x_i + w * p_parent(i)
        #
        # With w=0.5 (default), both components contribute equally.
        # ------------------------------------------------------------------
        (
            parent_index_d,
            parent_index_p,
            parent_index_m,
            num_parent_d,
            num_parent_p,
            num_parent_m,
        ) = self.construct_parent_maps(EHR_path)

        self.register_buffer(
            "parent_index_d",
            torch.as_tensor(parent_index_d, dtype=torch.long, device=self.device),
        )
        self.register_buffer(
            "parent_index_p",
            torch.as_tensor(parent_index_p, dtype=torch.long, device=self.device),
        )
        self.register_buffer(
            "parent_index_m",
            torch.as_tensor(parent_index_m, dtype=torch.long, device=self.device),
        )

        self.num_parent_categories = (
            num_parent_d,
            num_parent_p,
            num_parent_m,
        )

        self.parent_weight = float(getattr(self.args, "parent_weight", 0.5))
        if not 0.0 <= self.parent_weight <= 1.0:
            raise ValueError("parent_weight must be in [0, 1].")

        # embeddings
        self.embeddings = nn.ModuleList(
            [nn.Embedding(self.voc_size[i]+1, self.emb_dim, padding_idx=self.voc_size[i]) for i in range(3)]) # D,P,M
        #
        # Learnable embeddings for parent categories (D, P, M).
        self.parent_embeddings = nn.ModuleList([
            nn.Embedding(self.num_parent_categories[0], self.emb_dim),
            nn.Embedding(self.num_parent_categories[1], self.emb_dim),
            nn.Embedding(self.num_parent_categories[2], self.emb_dim),
        ])
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

    @staticmethod
    def _single_parent_index(category_matrix, entity_name):
        """
        Convert a binary entity-by-category incidence matrix into one parent
        index per entity.

        The current preprocessing assigns every diagnosis, procedure, and
        medication code to exactly one coarse parent category. This baseline
        intentionally relies on that one-parent assumption.
        """
        category_matrix = np.asarray(category_matrix)
        membership_count = (category_matrix > 0).sum(axis=1)

        if not np.all(membership_count == 1):
            bad_rows = np.where(membership_count != 1)[0]
            raise ValueError(
                f"{entity_name}: expected exactly one parent category per "
                f"entity, but {len(bad_rows)} entities violate this assumption. "
                f"Example indices: {bad_rows[:10].tolist()}"
            )

        parent_index = np.argmax(category_matrix > 0, axis=1).astype(np.int64)
        return parent_index, int(category_matrix.shape[1])

    def construct_parent_maps(self, EHR_path):
        """
        Load HG_matrix.pkl only as a source of taxonomy category membership.
        No hypergraph object is constructed.

        all_mats = (im_col_dC, im_col_pC, im_col_mC, im_col_DDI)
        """
        with open(EHR_path, "rb") as f:
            all_mats = dill.load(f)

        # Same slices as the full TDH-DR model.
        im_D = all_mats[0][0:self.voc_size[0], :]
        im_P = all_mats[1][
            self.voc_size[0]:self.voc_size[0] + self.voc_size[1], :
        ]
        im_M = all_mats[2][
            self.voc_size[0] + self.voc_size[1]:
            self.voc_size[0] + self.voc_size[1] + self.voc_size[2],
            :
        ]

        parent_index_d, num_parent_d = self._single_parent_index(
            im_D, "Diagnosis"
        )
        parent_index_p, num_parent_p = self._single_parent_index(
            im_P, "Procedure"
        )
        parent_index_m, num_parent_m = self._single_parent_index(
            im_M, "Medication"
        )

        print(
            f"Parent-category baseline - Diagnosis: "
            f"{self.voc_size[0]} entities, {num_parent_d} parent categories"
        )
        print(
            f"Parent-category baseline - Procedure: "
            f"{self.voc_size[1]} entities, {num_parent_p} parent categories"
        )
        print(
            f"Parent-category baseline - Medication: "
            f"{self.voc_size[2]} entities, {num_parent_m} parent categories"
        )

        return (
            parent_index_d,
            parent_index_p,
            parent_index_m,
            num_parent_d,
            num_parent_p,
            num_parent_m,
        )

    def propagate(self):
        """
        Construct taxonomy-aware entity representations without hypergraph
        message passing.

        Each entity combines:
          (1) its own trainable embedding, and
          (2) the trainable embedding of its single parent category.

        The padding row is appended unchanged, matching the interface expected
        by the original forward/evaluate code.
        """
        w = self.parent_weight

        # Diagnosis
        diag_base = self.dropout(
            self.embeddings[0].weight[:self.voc_size[0]]
        )
        diag_parent = self.dropout(
            self.parent_embeddings[0](self.parent_index_d)
        )
        diag_f = (1.0 - w) * diag_base + w * diag_parent
        pad_d = self.embeddings[0].weight[
            self.voc_size[0]:self.voc_size[0] + 1
        ]
        diag_f = torch.cat([diag_f, pad_d], dim=0)

        # Procedure
        pro_base = self.dropout(
            self.embeddings[1].weight[:self.voc_size[1]]
        )
        pro_parent = self.dropout(
            self.parent_embeddings[1](self.parent_index_p)
        )
        pro_f = (1.0 - w) * pro_base + w * pro_parent
        pad_p = self.embeddings[1].weight[
            self.voc_size[1]:self.voc_size[1] + 1
        ]
        pro_f = torch.cat([pro_f, pad_p], dim=0)

        # Medication
        med_base = self.dropout(
            self.embeddings[2].weight[:self.voc_size[2]]
        )
        med_parent = self.dropout(
            self.parent_embeddings[2](self.parent_index_m)
        )
        med_f = (1.0 - w) * med_base + w * med_parent
        pad_m = self.embeddings[2].weight[
            self.voc_size[2]:self.voc_size[2] + 1
        ]
        med_f = torch.cat([med_f, pad_m], dim=0)

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
