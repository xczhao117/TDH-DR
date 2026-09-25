import os
import torch
import dill
from torch.utils.data import Dataset


class EHR_Train_EVAL_Dataset(Dataset):
    """
    generate dataset from preprocessed pkl file
    Args:
    - `path`: the path of data file
    - `vocab_size`: the vocabulary size of diagnosis, procedure, and medication dictionaries
    """

    def __init__(self, dataset, vocab_size):
        self.dataset = dataset
        self.vocab_size = vocab_size
        self.data = self.process_data()

    def __len__(self):
        return self.data[0].shape[0] # 样本数 = 诊断张量的第一个维度（所有返回张量维度一致）

    def __getitem__(self, index):
        cur_diag = self.data[0][index]
        cur_pro = self.data[1][index]
        cur_med = self.data[2][index]
        cur_med_bce = self.data[3][index]
        cur_med_ml = self.data[4][index]
        cur_len = self.data[5][index]
        return cur_diag, cur_pro, cur_med, cur_med_bce, cur_med_ml, cur_len

    def process_data(self, MaxVisit=2):
        # The train/validation/test split is performed by the experiment scripts.
        # This class receives one already-split patient list and converts it to tensors.
        dataset = self.dataset

        #def get_inputs(self, dataset, MaxVisit=2):
        # 将list的数据形式转换为tensor形式的
        # use the pad index to make th same length tensor
        diag_list, pro_list, med_all_list, med_list, med_ml_list, len_list = [], [], [], [], [], []
        max_visit = min(max([len(cur) for cur in dataset]), MaxVisit) # finnaly max_visit=2
        ml_diag = max([len(dataset[i][j][0]) for i in range(len(dataset)) for j in range(len(dataset[i]))]) # max number of diagnoses in one visit
        ml_pro = max([len(dataset[i][j][1]) for i in range(len(dataset)) for j in range(len(dataset[i]))]) # max number of procedures in one visit
        ml_med = max([len(dataset[i][j][2]) for i in range(len(dataset)) for j in range(len(dataset[i]))]) # max number of medications in one visit
        # [v1, v2, v3] -> [v1], [v1, v2], [v2, v3]
        
        zxc_med_cnt = 0
        zxc_smp_cnt = 0
        
        for p in dataset:
            cur_diag = torch.full((max_visit, ml_diag), self.vocab_size[0])
            cur_pro = torch.full((max_visit, ml_pro), self.vocab_size[1])
            cur_all_med = torch.full((max_visit, ml_med), self.vocab_size[2])
            for ad_idx in range(len(p)):
                d_list, p_list, m_list = p[ad_idx]
                if ad_idx >= max_visit:
                    cur_diag[:-1] = cur_diag[1:]
                    cur_pro[:-1] = cur_pro[1:]
                    cur_all_med[:-1] = cur_all_med[1:]
                    cur_diag[-1] = self.vocab_size[0]
                    cur_pro[-1] = self.vocab_size[1]
                    cur_all_med[-1] = self.vocab_size[2]
                    cur_diag[-1, :len(d_list)] = torch.LongTensor(d_list)
                    cur_pro[-1, :len(p_list)] = torch.LongTensor(p_list)
                    cur_all_med[-1, :len(m_list)] = torch.LongTensor(m_list)
                    # visit len mask
                    len_list.append(max_visit)
                else:
                    cur_diag[ad_idx, :len(d_list)] = torch.LongTensor(d_list) 
                    cur_pro[ad_idx, :len(p_list)] = torch.LongTensor(p_list)
                    cur_all_med[ad_idx, :len(m_list)] = torch.LongTensor(m_list)
                    # visit len mask
                    len_list.append(ad_idx + 1)

                diag_list.append(cur_diag.long().clone())
                pro_list.append(cur_pro.long().clone())
                med_all_list.append(cur_all_med.long().clone())
                # bce target
                cur_med = torch.zeros(self.vocab_size[2])
                cur_med[m_list] = 1
                med_list.append(cur_med)
                # multi-label margin target
                cur_med_ml = torch.full((self.vocab_size[2],), -1)
                cur_med_ml[:len(m_list)] = torch.LongTensor(m_list)
                med_ml_list.append(cur_med_ml)
                zxc_med_cnt += len(m_list)
                zxc_smp_cnt += 1
        print('avg. Drug Number: ', zxc_med_cnt/zxc_smp_cnt)

        diag_tensor = torch.stack(diag_list)
        pro_tensor = torch.stack(pro_list)
        med_tensor = torch.stack(med_all_list)
        med_tensor_bce_target = torch.stack(med_list)
        med_tensor_ml_target = torch.stack(med_ml_list)
        len_tensor = torch.LongTensor(len_list)
    
        return diag_tensor, pro_tensor, med_tensor, med_tensor_bce_target, med_tensor_ml_target, len_tensor


class EHR_Test_Dataset(Dataset):
    """
    generate dataset from preprocessed pkl file
    Args:
    - `test_data`: the loaded test data
    - `vocab_size`: the vocabulary size of diagnosis, procedure, and medication dictionaries
    """

    def __init__(self, test_data, vocab_size):
        self.vocab_size = vocab_size
        self.data = self.process_test_data(test_data)        

    def __len__(self):
        return self.data[0].shape[0] # 样本数 = 诊断张量的第一个维度（所有返回张量维度一致）

    def __getitem__(self, index):
        cur_diag = self.data[0][index]
        cur_pro = self.data[1][index]
        cur_med = self.data[2][index]
        cur_med_bce = self.data[3][index]
        cur_med_ml = self.data[4][index]
        cur_len = self.data[5][index]
        return cur_diag, cur_pro, cur_med, cur_med_bce, cur_med_ml, cur_len

    def process_test_data(self, dataset, MaxVisit=2):
        #def get_inputs(self, dataset, MaxVisit=2):
        # 将list的数据形式转换为tensor形式的
        # use the pad index to make th same length tensor
        diag_list, pro_list, med_all_list, med_list, med_ml_list, len_list = [], [], [], [], [], []
        max_visit = min(max([len(cur) for cur in dataset]), MaxVisit) # finnaly max_visit=2
        ml_diag = max([len(dataset[i][j][0]) for i in range(len(dataset)) for j in range(len(dataset[i]))]) # max number of diagnoses in one visit
        ml_pro = max([len(dataset[i][j][1]) for i in range(len(dataset)) for j in range(len(dataset[i]))]) # max number of procedures in one visit
        ml_med = max([len(dataset[i][j][2]) for i in range(len(dataset)) for j in range(len(dataset[i]))]) # max number of medications in one visit
        # [v1, v2, v3] -> [v1], [v1, v2], [v2, v3]
        
        zxc_med_cnt = 0
        zxc_smp_cnt = 0
        
        for p in dataset:
            cur_diag = torch.full((max_visit, ml_diag), self.vocab_size[0])
            cur_pro = torch.full((max_visit, ml_pro), self.vocab_size[1])
            cur_all_med = torch.full((max_visit, ml_med), self.vocab_size[2])
            for ad_idx in range(len(p)):
                d_list, p_list, m_list = p[ad_idx]
                if ad_idx >= max_visit:
                    cur_diag[:-1] = cur_diag[1:]
                    cur_pro[:-1] = cur_pro[1:]
                    cur_all_med[:-1] = cur_all_med[1:]
                    cur_diag[-1] = self.vocab_size[0]
                    cur_pro[-1] = self.vocab_size[1]
                    cur_all_med[-1] = self.vocab_size[2]
                    cur_diag[-1, :len(d_list)] = torch.LongTensor(d_list)
                    cur_pro[-1, :len(p_list)] = torch.LongTensor(p_list)
                    cur_all_med[-1, :len(m_list)] = torch.LongTensor(m_list)
                    # visit len mask
                    len_list.append(max_visit)
                else:
                    cur_diag[ad_idx, :len(d_list)] = torch.LongTensor(d_list) 
                    cur_pro[ad_idx, :len(p_list)] = torch.LongTensor(p_list)
                    cur_all_med[ad_idx, :len(m_list)] = torch.LongTensor(m_list)
                    # visit len mask
                    len_list.append(ad_idx + 1)

                diag_list.append(cur_diag.long().clone())
                pro_list.append(cur_pro.long().clone())
                med_all_list.append(cur_all_med.long().clone())
                # bce target
                cur_med = torch.zeros(self.vocab_size[2])
                cur_med[m_list] = 1
                med_list.append(cur_med)
                # multi-label margin target
                cur_med_ml = torch.full((self.vocab_size[2],), -1)
                cur_med_ml[:len(m_list)] = torch.LongTensor(m_list)
                med_ml_list.append(cur_med_ml)
                zxc_med_cnt += len(m_list)
                zxc_smp_cnt += 1
        print('avg. Drug Number: ', zxc_med_cnt/zxc_smp_cnt)

        diag_tensor = torch.stack(diag_list)
        pro_tensor = torch.stack(pro_list)
        med_tensor = torch.stack(med_all_list)
        med_tensor_bce_target = torch.stack(med_list)
        med_tensor_ml_target = torch.stack(med_ml_list)
        len_tensor = torch.LongTensor(len_list)
    
        return diag_tensor, pro_tensor, med_tensor, med_tensor_bce_target, med_tensor_ml_target, len_tensor
