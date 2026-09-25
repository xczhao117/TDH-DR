import pandas as pd
from datetime import datetime
import dill
import numpy as np
from collections import defaultdict, Counter
import os
from paths import RAW_DATA_DIR, PROCESSED_DATA_DIR
import math

import scipy.sparse as sp
import torch

np.random.seed(1029)

# files can be downloaded from https://mimic.physionet.org/gettingstarted/dbsetup/
# please change into your own MIMIC folder
datadir = str(RAW_DATA_DIR) + os.sep
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
med_file = datadir + "mimic-iii/PRESCRIPTIONS.csv"
diag_file = datadir + "mimic-iii/DIAGNOSES_ICD.csv"
procedure_file = datadir + "mimic-iii/PROCEDURES_ICD.csv"
admission_file = datadir + "mimic-iii/ADMISSIONS.csv"

med_structure_file = datadir + 'idx2SMILES.pkl'

# drug code mapping files
ndc2atc_file = datadir + 'ndc2atc_level4.csv' 
cid_atc = datadir + 'drug-atc.csv'
ndc_rxnorm_file = datadir + 'ndc2rxnorm_mapping.txt'

# ddi information
ddi_file = datadir + 'drug-DDI.csv'


##### process medications #####
# load med data
def med_process(med_file):
    med_pd = pd.read_csv(med_file, dtype={'NDC':'category'})

    med_pd.drop(columns=['ROW_ID','DRUG_TYPE','DRUG_NAME_POE','DRUG_NAME_GENERIC',
                        'FORMULARY_DRUG_CD','PROD_STRENGTH','DOSE_VAL_RX',
                        'DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP', 'GSN', 'FORM_UNIT_DISP',
                        'ROUTE','ENDDATE','DRUG'], axis=1, inplace=True)
    med_pd.drop(index = med_pd[med_pd['NDC'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
    med_pd['STARTDATE'] = pd.to_datetime(med_pd['STARTDATE'], format='%Y-%m-%d %H:%M:%S')    
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)

    med_pd = med_pd.drop(columns=['ICUSTAY_ID'])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    return med_pd

# medication mapping
def ndc2atc4(med_pd):
    with open(ndc_rxnorm_file, 'r') as f:
        ndc2rxnorm = eval(f.read())
    # print("med_pd", med_pd)
    med_pd['RXCUI'] = med_pd['NDC'].map(ndc2rxnorm)
    # print("wa", med_pd['RXCUI'])
    # print(med_pd)
    med_pd.dropna(inplace=True)

    rxnorm2atc = pd.read_csv(ndc2atc_file)
    # print(rxnorm2atc)
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR','MONTH','NDC'])
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)
    med_pd.drop(index = med_pd[med_pd['RXCUI'].isin([''])].index, axis=0, inplace=True)
    
    med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(rxnorm2atc, on=['RXCUI'])
    med_pd.drop(columns=['NDC', 'RXCUI'], inplace=True)
    med_pd = med_pd.rename(columns={'ATC5':'NDC'})
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: x[:4])

    # A07B,A09A,A11G,A12C,C01A,H01B,H01C,J01F. 
    # remove these medicines that cannot be processed by fragnet.
    med_pd.drop(index = med_pd[med_pd['NDC'].isin(['A07B','A09A','A11G','A12C','C01A','H01B','H01C','J01F'])].index, axis=0, inplace=True)

    med_pd = med_pd.drop_duplicates()    
    med_pd = med_pd.reset_index(drop=True)
    # print("med_pd_new", med_pd)
    return med_pd

# visit >= 2
def process_visit_lg2(med_pd):
    a = med_pd[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
    a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x:len(x))
    a = a[a['HADM_ID_Len'] > 1]
    return a 

# most common medications
def filter_300_most_med(med_pd):
    med_count = med_pd.groupby(by=['NDC']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    med_pd = med_pd[med_pd['NDC'].isin(med_count.loc[:299, 'NDC'])]
    
    return med_pd.reset_index(drop=True)

##### process diagnosis #####
def diag_process(diag_file):
    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=['SEQ_NUM','ROW_ID'],inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['SUBJECT_ID','HADM_ID'], inplace=True)
    diag_pd = diag_pd.reset_index(drop=True)

    # sample
    # diag_pd = diag_pd.sample(frac=0.8, random_state=5, axis=0) # to test robustness

    def filter_2000_most_diag(diag_pd):
        diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
        diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(diag_count.loc[:1999, 'ICD9_CODE'])]
        
        return diag_pd.reset_index(drop=True)

    diag_pd = filter_2000_most_diag(diag_pd)

    return diag_pd

##### process procedure #####
def procedure_process(procedure_file):
    pro_pd = pd.read_csv(procedure_file, dtype={'ICD9_CODE':'category'})
    pro_pd.drop(columns=['ROW_ID'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], inplace=True)
    pro_pd.drop(columns=['SEQ_NUM'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    return pro_pd

def filter_1000_most_pro(pro_pd):
    pro_count = pro_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    pro_pd = pro_pd[pro_pd['ICD9_CODE'].isin(pro_count.loc[:1000, 'ICD9_CODE'])]
    
    return pro_pd.reset_index(drop=True) 

###### combine three tables #####
def combine_process(med_pd, diag_pd, pro_pd):

    med_pd_key = med_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    diag_pd_key = diag_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    pro_pd_key = pro_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()

    combined_key = med_pd_key.merge(diag_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined_key = combined_key.merge(pro_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    diag_pd = diag_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    med_pd = med_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    pro_pd = pro_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    # flatten and merge
    diag_pd = diag_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index()  
    med_pd = med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['NDC'].unique().reset_index()
    pro_pd = pro_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index().rename(columns={'ICD9_CODE':'PRO_CODE'})  
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: list(x))
    pro_pd['PRO_CODE'] = pro_pd['PRO_CODE'].map(lambda x: list(x))
    data = diag_pd.merge(med_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data = data.merge(pro_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    #     data['ICD9_CODE_Len'] = data['ICD9_CODE'].map(lambda x: len(x))
    data['NDC_Len'] = data['NDC'].map(lambda x: len(x))

    return data

def add_adm_time(data, admission_file):
    adm_data = pd.read_csv(admission_file)
    adm_data = adm_data[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME']]
    adm_data.dropna(inplace=True)
    data = data.merge(adm_data, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data['ADMITTIME'] = data['ADMITTIME'].map(
        lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    return data

def statistics(data):
    print('#patients ', data['SUBJECT_ID'].unique().shape)
    print('#clinical events ', len(data))
    
    diag = data['ICD9_CODE'].values
    med = data['NDC'].values
    pro = data['PRO_CODE'].values
    
    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])
    unique_pro = set([j for i in pro for j in list(i)])
    
    print('#diagnosis ', len(unique_diag))
    print('#med ', len(unique_med))
    print('#procedure', len(unique_pro))
    
    avg_diag, avg_med, avg_pro, max_diag, max_med, max_pro, cnt, max_visit, avg_visit = [0 for i in range(9)]

    for subject_id in data['SUBJECT_ID'].unique():
        item_data = data[data['SUBJECT_ID'] == subject_id]
        x, y, z = [], [], []
        visit_cnt = 0
        for index, row in item_data.iterrows():
            visit_cnt += 1
            cnt += 1
            x.extend(list(row['ICD9_CODE']))
            y.extend(list(row['NDC']))
            z.extend(list(row['PRO_CODE']))
        x, y, z = set(x), set(y), set(z)
        avg_diag += len(x)
        avg_med += len(y)
        avg_pro += len(z)
        avg_visit += visit_cnt
        if len(x) > max_diag:
            max_diag = len(x)
        if len(y) > max_med:
            max_med = len(y) 
        if len(z) > max_pro:
            max_pro = len(z)
        if visit_cnt > max_visit:
            max_visit = visit_cnt
        
    print('#avg of diagnoses ', avg_diag/ cnt)
    print('#avg of medicines ', avg_med/ cnt)
    print('#avg of procedures ', avg_pro/ cnt)
    print('#avg of vists ', avg_visit/ len(data['SUBJECT_ID'].unique()))
    
    print('#max of diagnoses ', max_diag)
    print('#max of medicines ', max_med)
    print('#max of procedures ', max_pro)
    print('#max of visit ', max_visit)

##### indexing file and final record
class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)
                
# create voc set
def create_str_token_mapping(df):
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()
    
    for index, row in df.iterrows():
        diag_voc.add_sentence(row['ICD9_CODE'])
        med_voc.add_sentence(row['NDC'])
        pro_voc.add_sentence(row['PRO_CODE'])
    
    dill.dump(obj={'diag_voc':diag_voc, 'med_voc':med_voc ,'pro_voc':pro_voc}, file=open(str(PROCESSED_DATA_DIR / 'voc_final.pkl'),'wb'))
    return diag_voc, med_voc, pro_voc

# create final records
def create_patient_record(df, diag_voc, med_voc, pro_voc):
    records = [] # (patient, code_kind:3, codes)  code_kind:diag, proc, med
    for subject_id in df['SUBJECT_ID'].unique():
        item_df = df[df['SUBJECT_ID'] == subject_id]
        item_df = item_df.sort_values(by=['ADMITTIME'])
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([diag_voc.word2idx[i] for i in row['ICD9_CODE']])
            admission.append([pro_voc.word2idx[i] for i in row['PRO_CODE']])
            admission.append([med_voc.word2idx[i] for i in row['NDC']])
            # add medicine molecular structure information (by fragnet) into patient data
            '''
            mol_list = []
            for one_atc3 in row['NDC']:
                # lode medicine information for this ATC3 code
                mols = dill.load(open(str(PROCESSED_DATA_DIR / f"{one_atc3}.pkl"), 'rb'))
                for moldata in mols:
                    # print(moldata)
                    mol_list.append(moldata)
            admission.append(mol_list)
            '''
            patient.append(admission)
        records.append(patient) 
    dill.dump(obj=records, file=open(str(PROCESSED_DATA_DIR / 'records_final.pkl'), 'wb'))
    return records
        


# get ddi matrix
def get_ddi_matrix(records, med_voc, ddi_file):

    TOPK = 40 # topk drug-drug interaction
    cid2atc_dic = defaultdict(set)
    med_voc_size = len(med_voc.idx2word)
    med_unique_word = [med_voc.idx2word[i] for i in range(med_voc_size)]
    atc3_atc4_dic = defaultdict(set)
    for item in med_unique_word:
        atc3_atc4_dic[item[:4]].add(item)
    
    with open(cid_atc, 'r') as f:
        for line in f:
            line_ls = line[:-1].split(',')
            cid = line_ls[0]
            atcs = line_ls[1:]
            for atc in atcs:
                if len(atc3_atc4_dic[atc[:4]]) != 0:
                    cid2atc_dic[cid].add(atc[:4])
            
    # ddi load
    ddi_df = pd.read_csv(ddi_file)
    # fliter sever side effect 
    ddi_most_pd = ddi_df.groupby(by=['Polypharmacy Side Effect', 'Side Effect Name']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    ddi_most_pd = ddi_most_pd.iloc[-TOPK:,:]
    # ddi_most_pd = pd.DataFrame(columns=['Side Effect Name'], data=['as','asd','as'])
    fliter_ddi_df = ddi_df.merge(ddi_most_pd[['Side Effect Name']], how='inner', on=['Side Effect Name'])
    ddi_df = fliter_ddi_df[['STITCH 1','STITCH 2']].drop_duplicates().reset_index(drop=True)

    # ddi adj
    ddi_adj = np.zeros((med_voc_size,med_voc_size))
    for index, row in ddi_df.iterrows():
        # ddi
        cid1 = row['STITCH 1']
        cid2 = row['STITCH 2']
        
        # cid -> atc_level3
        for atc_i in cid2atc_dic[cid1]:
            for atc_j in cid2atc_dic[cid2]:
                
                # atc_level3 -> atc_level4
                for i in atc3_atc4_dic[atc_i]:
                    for j in atc3_atc4_dic[atc_j]:
                        if med_voc.word2idx[i] != med_voc.word2idx[j]:
                            ddi_adj[med_voc.word2idx[i], med_voc.word2idx[j]] = 1
                            ddi_adj[med_voc.word2idx[j], med_voc.word2idx[i]] = 1
    dill.dump(ddi_adj, open(str(PROCESSED_DATA_DIR / 'ddi_A_final.pkl'), 'wb'))

    return ddi_adj

def ddi_rate_score(record, path):
    # ddi rate
    if isinstance(path, str):
        ddi_A = dill.load(open(path, 'rb'))
    all_cnt = 0
    dd_cnt = 0
    for patient in record:
        for adm in patient:
            med_code_set = adm[-1] # D,P,M,mol
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        dd_cnt += 1
    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt



def process_step1():


    # for med
    med_pd = med_process(med_file)
    med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)    
    med_pd = med_pd.merge(med_pd_lg2[['SUBJECT_ID']], on='SUBJECT_ID', how='inner').reset_index(drop=True) 

    med_pd = ndc2atc4(med_pd)
    # print("pos1", med_pd)
    NDCList = dill.load(open(med_structure_file, 'rb'))
    # print("NDCList", NDCList)
    med_pd = med_pd[med_pd.NDC.isin(list(NDCList.keys()))]
    # print("pos2", med_pd)
    med_pd = filter_300_most_med(med_pd)

    print ('complete medication processing')

    # for diagnosis
    diag_pd = diag_process(diag_file)

    print ('complete diagnosis processing')

    # for procedure
    pro_pd = procedure_process(procedure_file)
    # pro_pd = filter_1000_most_pro(pro_pd)

    print ('complete procedure processing')

    # combine
    data = combine_process(med_pd, diag_pd, pro_pd)
    data = add_adm_time(data, admission_file)
    statistics(data)
    # print("data", data)

    print ('complete combining')


    # ddi_matrix
    diag_voc, med_voc, pro_voc = create_str_token_mapping(data)
    records = create_patient_record(data, diag_voc, med_voc, pro_voc)
    ddi_adj = get_ddi_matrix(records, med_voc, ddi_file)
    ddi_rate = ddi_rate_score(records, str(PROCESSED_DATA_DIR / 'ddi_A_final.pkl'))
    print("ddi_rate", ddi_rate) # ddi_rate 0.09674941837943381


def construct_HG():
    voc = dill.load(file=open(str(PROCESSED_DATA_DIR / 'voc_final.pkl'),'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    #print(voc_size) # (1958, 1430, 123)
    
    ## DC
    # ICD-9 的层级分类规则（章→节→类目→亚目→细目）
    # 普通代码：3 位类目为前 3 位数字（如 4239→423，5119→511）
    # V码：类目为 V + 前 2 位数字（如 V1259→V12，V1046→V10）
    # E码：类目为 E + 前 3 位数字（如 E8788→E878，E8790→E879）
    diag_dict = dict()
    for idx in range(0, voc_size[0]):
        key = diag_voc.idx2word[idx]
        # print(key)
        #if key in diag_dict:
        #    print(key, 'already in dict, skipped')
        #    continue
        if key.startswith('E'):
            diag_dict[key] = key[:4]
        else:
            diag_dict[key] = key[:3]

    unique_values = list(set(list(diag_dict.values())))
    #print(len(list(diag_dict.keys())), len(unique_values)) # 1958 554 // 1958种诊断，它们属于554个类目
    unique_dC_num = len(unique_values)

    im_diag_class = np.zeros((voc_size[0], unique_dC_num))#, dtype=np.int16)
    for didx in range(0, voc_size[0]):
        key = diag_voc.idx2word[didx]
        for cidx in range(0, unique_dC_num):
            if diag_dict[key] == unique_values[cidx]:
                im_diag_class[didx, cidx] = 1
    #print(im_diag_class)
    #print(im_diag_class.sum(axis=0))
    
    ## PC
    # 要对这些 ICD-9-CM-3 手术代码分类，核心依据是其前 2 位数字--这是 ICD-9-CM-3 的 “大类标识”，
    # 代表手术所属的解剖系统或操作类型（后 2 位为大类下的细分操作）
    pro_dict = dict()
    for idx in range(0, voc_size[1]):
        key = pro_voc.idx2word[idx]
        # print(key)
        pro_dict[key] = key[:2]

    unique_values = list(set(list(pro_dict.values())))
    #print(len(list(pro_dict.keys())), len(unique_values)) # 1430 92 // 1430种手术，它们属于92个类目
    unique_pC_num = len(unique_values)

    im_pro_class = np.zeros((voc_size[1], unique_pC_num))#, dtype=np.int16)
    for pidx in range(0, voc_size[1]):
        key = pro_voc.idx2word[pidx]
        for cidx in range(0, unique_pC_num):
            if pro_dict[key] == unique_values[cidx]:
                im_pro_class[pidx, cidx] = 1
    #print(im_pro_class)
    #print(im_pro_class.sum(axis=0))

    ## MC
    # ATC 代码共 7 位，前 2 位（格式为 “字母 + 数字”，如 A01、B02）为ATC2 层级，
    # 代表药物的治疗大类（对应特定疾病系统或治疗目的）；后 5 位（如 A01A、B02B）为 ATC3 及以下细分层级，代表具体药物或剂型。
    med_dict = dict()
    for idx in range(0, voc_size[2]):
        key = med_voc.idx2word[idx]
        # print(key)
        med_dict[key] = key[:3]

    unique_values = list(set(list(med_dict.values())))
    #print(len(list(med_dict.keys())), len(unique_values)) # 123 68 // 123种药物，它们属于68个类目
    unique_mC_num = len(unique_values)

    im_med_class = np.zeros((voc_size[2], unique_mC_num))#, dtype=np.int16)
    for midx in range(0, voc_size[2]):
        key = med_voc.idx2word[midx]
        for cidx in range(0, unique_mC_num):
            if med_dict[key] == unique_values[cidx]:
                im_med_class[midx, cidx] = 1
    #print(im_med_class)
    #print(im_med_class.sum(axis=0))

    ## MM-DDI
    ddi_adj = dill.load(open(str(PROCESSED_DATA_DIR / 'ddi_A_final.pkl'), 'rb'))
    # print(ddi_adj.shape) # (123, 123)
    # print(ddi_adj)
    im_MM_ddi = 1 - ddi_adj # im_mm_ddi indicates the drug pairs that have no adverse DDIs
    # print(im_MM_ddi) 

    # Dataset splitting is intentionally performed in the experiment scripts.

    # dC (diagnosis class) column
    #print(im_diag_class.shape) # (1958, 554)
    im_pdc0 = np.zeros((voc_size[1], unique_dC_num))
    im_mdc0 = np.zeros((voc_size[2], unique_dC_num))
    im_col_dC = np.vstack((im_diag_class, im_pdc0, im_mdc0))
    #print(im_col_dC.shape) # (3511, 554)

    # pC (procedure class) column
    #print(im_pro_class.shape) # (1430, 92)
    im_dpc0 = np.zeros((voc_size[0], unique_pC_num))
    im_mpc0 = np.zeros((voc_size[2], unique_pC_num))
    im_col_pC = np.vstack((im_dpc0, im_pro_class, im_mpc0))
    #print(im_col_pC.shape) # (3511, 92)

    # mC (medication class) column
    #print(im_med_class.shape) # (123, 68)
    im_dmc0 = np.zeros((voc_size[0], unique_mC_num))
    im_pmc0 = np.zeros((voc_size[1], unique_mC_num))
    im_col_mC = np.vstack((im_dmc0, im_pmc0, im_med_class))
    #print(im_col_mC.shape) # (3511, 68)
    
    ## 1-DDI matrix
    im_d_ddi0 = np.zeros((voc_size[0], voc_size[2]))
    im_p_ddi0 = np.zeros((voc_size[1], voc_size[2]))
    im_col_DDI = np.vstack((im_d_ddi0, im_p_ddi0, im_MM_ddi))

    #
    all_mats = (im_col_dC, im_col_pC, im_col_mC, im_col_DDI)
    with open(str(PROCESSED_DATA_DIR / 'HG_matrix.pkl'), 'wb') as f:
        dill.dump(all_mats, f)


if __name__ == '__main__':
    process_step1()
    construct_HG()
