import dill
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hypergraph import Hypergraph, HGNNConv


class HGDR(nn.Module):
    """
    TDH-DR ablation: w/o explicit drug representation (w/o DrugRep).

    Key changes relative to the full model:
    1. No medication embedding matrix is created.
    2. No medication hypergraph / RHGNN is constructed.
    3. The health query is NOT matched to medication embeddings by dot product.
    4. Previous-visit medications are retained only as a binary multi-hot vector.
    5. [health query ; previous-medication multi-hot] is fed directly to an MLP
       that outputs medication logits.

    Diagnosis/procedure hypergraphs, longitudinal encoders, prediction losses,
    DDI-aware loss, data split, and training protocol remain unchanged.
    """

    def __init__(
        self,
        voc_size,
        ddi_adj_path,
        EHR_path,
        device=torch.device("cpu:0"),
        args=None,
    ):
        super().__init__()

        self.args = args
        self.voc_size = voc_size
        self.emb_dim = args.dim
        self.device = device

        # This ablation intentionally removes all medication-side embeddings/HG.
        if getattr(self.args, "use_hgm", False) or getattr(
            self.args, "use_hgm_ddi", False
        ):
            raise ValueError(
                "w/o DrugRep ablation must NOT use --use_hgm or --use_hgm_ddi. "
                "Medication embeddings and medication hypergraph propagation are "
                "intentionally removed."
            )

        with open(ddi_adj_path, "rb") as f:
            ddi_adj = dill.load(f)
            self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        # Only diagnosis and procedure hypergraphs are retained.
        self.hg_d, self.hg_p = self.construct_HG(EHR_path)

        if self.args.use_hgd:
            self.d_hgconv1 = HGNNConv(
                in_channels=self.emb_dim,
                out_channels=self.emb_dim,
                bias=True,
                use_bn=False,
                drop_rate=0.3,
                is_last=False,
            )
            self.d_hgconv = HGNNConv(
                in_channels=self.emb_dim,
                out_channels=self.emb_dim,
                bias=True,
                use_bn=False,
                drop_rate=0.3,
                is_last=True,
            )

        if self.args.use_hgp:
            self.p_hgconv1 = HGNNConv(
                in_channels=self.emb_dim,
                out_channels=self.emb_dim,
                bias=True,
                use_bn=False,
                drop_rate=0.3,
                is_last=False,
            )
            self.p_hgconv = HGNNConv(
                in_channels=self.emb_dim,
                out_channels=self.emb_dim,
                bias=True,
                use_bn=False,
                drop_rate=0.3,
                is_last=True,
            )

        # Only diagnosis and procedure have explicit embedding matrices.
        # There is deliberately NO medication embedding table.
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    self.voc_size[i] + 1,
                    self.emb_dim,
                    padding_idx=self.voc_size[i],
                )
                for i in range(2)
            ]
        )

        self.dropout = nn.Dropout(0.3)

        # Same longitudinal diagnosis/procedure encoders as the full model.
        self.encoders = nn.ModuleList(
            [
                nn.GRU(self.emb_dim, self.emb_dim, batch_first=True)
                for _ in range(2)
            ]
        )

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2 * self.emb_dim, self.emb_dim),
        )

        # Direct multi-label prediction head.
        #
        # Input:
        #   health query:                (B, D)
        #   previous-medication binary:  (B, N_med)
        #
        # No medication feature matrix participates in this prediction.
        self.direct_med_head = nn.Sequential(
            nn.Linear(
                self.emb_dim + self.voc_size[2],
                self.emb_dim,
            ),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.voc_size[2]),
            nn.LayerNorm(self.voc_size[2]),
        )

        print(
            "w/o DrugRep: medication embedding/RHGNN disabled; "
            "previous medications are represented only by binary multi-hot vectors."
        )

    def im2edges(self, im_np):
        e2v = []
        num_edges = im_np.shape[1]
        for j in range(num_edges):
            nodes_in_edge = np.where(im_np[:, j] > 0)[0].tolist()
            if len(nodes_in_edge) > 0:
                e2v.append(nodes_in_edge)
        return e2v

    def construct_HG(self, EHR_path):
        """
        Construct only the diagnosis and procedure hypergraphs.

        The medication taxonomy matrix is intentionally not used in this
        ablation because medications no longer have explicit node embeddings.
        """
        with open(EHR_path, "rb") as f:
            all_mats = dill.load(f)

        # Diagnosis hypergraph
        if self.args.use_hgd:
            im_D = all_mats[0][0 : self.voc_size[0], :]
            d_e2v = self.im2edges(im_D)
            num_d_v = im_D.shape[0]
            print(
                f"诊断节点数 num_d_v: {num_d_v}, "
                f"诊断超边数: {len(d_e2v)}"
            )
            hg_d = Hypergraph(
                num_v=num_d_v,
                e_list=d_e2v,
                device=self.device,
            )
        else:
            hg_d = 0

        # Procedure hypergraph
        if self.args.use_hgp:
            im_P = all_mats[1][
                self.voc_size[0] :
                self.voc_size[0] + self.voc_size[1],
                :
            ]
            p_e2v = self.im2edges(im_P)
            num_p_v = im_P.shape[0]
            print(
                f"治疗节点数 num_p_v: {num_p_v}, "
                f"治疗超边数: {len(p_e2v)}"
            )
            hg_p = Hypergraph(
                num_v=num_p_v,
                e_list=p_e2v,
                device=self.device,
            )
        else:
            hg_p = 0

        return hg_d, hg_p

    def propagate(self):
        """Propagate diagnosis/procedure embeddings only."""

        if self.args.use_hgd:
            diag_feature = self.dropout(
                self.embeddings[0].weight[: self.voc_size[0]]
            )
            diag_f1 = self.dropout(
                self.d_hgconv1(diag_feature, self.hg_d)
            )
            diag_f = (
                self.dropout(self.d_hgconv(diag_f1, self.hg_d)) / 4
                + diag_f1 / 3
                + diag_feature / 2
            )
            pad_d = self.embeddings[0].weight[
                self.voc_size[0] : self.voc_size[0] + 1
            ]
            diag_f = torch.cat([diag_f, pad_d], dim=0)
        else:
            diag_f = self.embeddings[0].weight

        if self.args.use_hgp:
            pro_feature = self.dropout(
                self.embeddings[1].weight[: self.voc_size[1]]
            )
            pro_f1 = self.dropout(
                self.p_hgconv1(pro_feature, self.hg_p)
            )
            pro_f = (
                self.dropout(self.p_hgconv(pro_f1, self.hg_p)) / 4
                + pro_f1 / 3
                + pro_feature / 2
            )
            pad_p = self.embeddings[1].weight[
                self.voc_size[1] : self.voc_size[1] + 1
            ]
            pro_f = torch.cat([pro_f, pad_p], dim=0)
        else:
            pro_f = self.embeddings[1].weight

        return diag_f, pro_f

    def previous_medication_multihot(self, meds, visit_len):
        """
        Encode previous-visit medication identities as a binary 0/1 vector.

        `meds` has shape (B, 2, Lm), and padding medication index is N_med.
        For samples with visit_len == 1, meds[:, 0, :] corresponds to the
        current visit, so it MUST be masked out to avoid target leakage.
        """
        n_med = self.voc_size[2]
        prev_idx = meds[:, 0, :].long()  # (B, Lm)

        valid = (prev_idx >= 0) & (prev_idx < n_med)
        safe_idx = prev_idx.clamp(min=0, max=n_med - 1)

        prev_multihot = torch.zeros(
            prev_idx.size(0),
            n_med,
            dtype=torch.float32,
            device=meds.device,
        )

        # scatter_add followed by clamp handles any accidental duplicate code.
        prev_multihot.scatter_add_(
            1,
            safe_idx,
            valid.float(),
        )
        prev_multihot.clamp_(0.0, 1.0)

        # Only patients with two valid visits have true previous medications.
        history_mask = (
            visit_len.to(meds.device) >= 2
        ).unsqueeze(1).float()

        prev_multihot = prev_multihot * history_mask
        return prev_multihot

    def predict_logits(self, diags, pros, meds, visit_len):
        """
        Shared prediction path for training and evaluation.
        """
        diag_feature, pro_feature = self.propagate()

        diag_embs = F.embedding(
            diags.long(),
            diag_feature,
        )
        pro_embs = F.embedding(
            pros.long(),
            pro_feature,
        )

        diag_emb_seq = self.dropout(
            diag_embs.sum(dim=2)
        )
        pro_emb_seq = self.dropout(
            pro_embs.sum(dim=2)
        )

        o1, _ = self.encoders[0](diag_emb_seq)
        o2, _ = self.encoders[1](pro_emb_seq)

        o1 = torch.stack(
            [
                o1[i, visit_len[i] - 1, :]
                for i in range(visit_len.shape[0])
            ]
        )
        o2 = torch.stack(
            [
                o2[i, visit_len[i] - 1, :]
                for i in range(visit_len.shape[0])
            ]
        )

        patient_representations = torch.cat(
            [o1, o2],
            dim=-1,
        )

        # Same health-query module as the full TDH-DR model.
        query = self.query(patient_representations)  # (B, D)

        # Medication history is now only a binary vector.
        prev_med_multihot = self.previous_medication_multihot(
            meds,
            visit_len,
        )  # (B, N_med)

        # No drug embeddings and no query-drug dot product.
        direct_input = torch.cat(
            [query, prev_med_multihot],
            dim=-1,
        )

        result = self.direct_med_head(direct_input)
        return result

    def forward(self, diags, pros, meds, visit_len):
        result = self.predict_logits(
            diags,
            pros,
            meds,
            visit_len,
        )

        # Keep the original DDI-aware loss exactly unchanged.
        if self.args.ddi:
            neg_pred_prob = torch.sigmoid(result)
            tmp_left = neg_pred_prob.unsqueeze(2)
            tmp_right = neg_pred_prob.unsqueeze(1)
            neg_pred_prob = torch.matmul(
                tmp_left,
                tmp_right,
            )
            batch_neg = (
                0.0005
                * neg_pred_prob.mul(
                    self.tensor_ddi_adj
                ).sum()
            )
        else:
            batch_neg = 0

        return result, batch_neg

    def evaluate(self, diags, pros, meds, visit_len):
        return self.predict_logits(
            diags,
            pros,
            meds,
            visit_len,
        )
