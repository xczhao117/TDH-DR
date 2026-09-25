import sys
import time
import math
import dill
import numpy as np
import argparse
from collections import defaultdict
from torch.optim import Adam, SGD
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import HGDR
from dataset import EHR_Train_EVAL_Dataset, EHR_Test_Dataset
from util import ddi_rate_score, multi_label_metric
from paths import PROCESSED_DATA_DIR, checkpoint_dir

# 补充实验：将药物的RHGNN改为regular HGNN，进行消融性实验(应该与主方法一样epoch=13（从0开始）, 实际是epoch=14)

parser = argparse.ArgumentParser()
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--model_name', type=str, default='model_name', help="model name")
parser.add_argument('--resume_path', type=str, default='resume_path', help='resume path')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--dim', type=int, default=64, help='dimension')
parser.add_argument('--datadir', type=str, default=str(PROCESSED_DATA_DIR), help='datadir')
parser.add_argument('--cuda', type=int, default=0, help='use cuda')
parser.add_argument('--seed', type=int, default=1029, help='random seed')
parser.add_argument('--epoch', type=int, default=120, help='# of epoches')
parser.add_argument('--bs', type=int, default=16, help='batch size')
parser.add_argument('--load', action='store_true', default=False, help='load resume file')
parser.add_argument('--ddi', action='store_true', default=False, help='calculate ddi') # default设置为False，才有可能让这个选项为false
parser.add_argument('--target_ddi', type=float, default=0.005, help='target ddi')
parser.add_argument('--alpha', type=float, default=0.95, help='weight for loss_bce and loss_multi')
parser.add_argument('--beta', type=float, default=0.9, help='weight for loss_acc and loss_ddi')

parser.add_argument('--use_hgd', action='store_true', default=False, help='apply HG model of diags')
parser.add_argument('--use_hgp', action='store_true', default=False, help='apply HG model of pros')
parser.add_argument('--use_hgm', action='store_true', default=False, help='apply HG model of drugs')
parser.add_argument('--use_hgm_ddi', action='store_true', default=False, help='using DDI for HG model of drugs')

#parser.add_argument('--ddi2', action='store_true', default=False, help='compute side-effects loss for med embeddings based on DDI matrix')

parser.add_argument('--hgmtype', type=int, default=3, choices=[3], help='medication HG type; this experiment fixes it to 3=regular HGNN')
parser.add_argument('--hmw', type=float, default=0.3, help='weight for hgmt1&2')

args = parser.parse_args()
print(args)

run_dir = checkpoint_dir(args.model_name)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.cuda > -1:
    torch.cuda.manual_seed(args.seed)


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def eval(model: HGDR, eval_data_loader, ddi_adj_path, device):
    model.eval()

    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0

    y_gt_list, y_pred_list, y_prob_list = [], [], []

    with torch.no_grad():
        for data_tensors in eval_data_loader:
            cur_diag, cur_pro, cur_med, cur_med_bce, _, cur_len = data_tensors
            # move current batch to device
            cur_diag = cur_diag.to(device)
            cur_pro = cur_pro.to(device)
            cur_med = cur_med.to(device)
            cur_med_bce_target = cur_med_bce.to(device)
            cur_len = cur_len.to(device)
            
            # run model (forward)
            result = model.evaluate(cur_diag, cur_pro, cur_med, cur_len) # (B, voc_size[2]), probability for each medicine
            #
            result = torch.sigmoid(result).detach().cpu().numpy()
            preds = np.zeros_like(result)
            preds[result>=0.5] = 1
            preds[result<0.5] = 0
            visit_cnt += cur_med_bce_target.shape[0]
            med_cnt += preds.sum()
            cur_med_bce_target = cur_med_bce_target.detach().cpu().numpy()
            #-----------metric_obj.feed_data(cur_med_target, preds, result)
            y_gt_list.append(cur_med_bce_target)
            y_pred_list.append(preds)
            y_prob_list.append(result)

    #-----------metric_obj.set_data()
    gt = np.vstack(y_gt_list)
    pred = np.vstack(y_pred_list)
    prob = np.vstack(y_prob_list)
    #
    ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(gt, pred, prob)
    # ddi rate
    list_pred = []
    list_target = []
    for i in range(pred.shape[0]):
        idx = np.nonzero(pred[i])[0].tolist()
        list_pred.append(idx)
    for i in range(gt.shape[0]):
        idx = np.nonzero(gt[i])[0].tolist()
        list_target.append(idx)
    ddi_rate = ddi_rate_score([list_pred], ddi_adj_path)

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt



def main():
    records_path = os.path.join(args.datadir, 'records_final.pkl')
    with open(records_path, 'rb') as f:
        records = dill.load(f)

    # Patient-level split: 2/3 train, 1/6 test, 1/6 validation.
    # Keep the original experiment split logic and patient ordering.
    split_point = int(len(records) * 2 / 3)
    data_train = records[:split_point]
    eval_len = int(len(records[split_point:]) / 2)
    data_test = records[split_point:split_point + eval_len]
    data_eval = records[split_point + eval_len:]
    ddi_adj_path = os.path.join(args.datadir, 'ddi_A_final.pkl')
    EHR_path = os.path.join(args.datadir, 'HG_matrix.pkl')
    #
    voc_path = os.path.join(args.datadir, 'voc_final.pkl')
    with open(voc_path, 'rb') as f:
        voc = dill.load(f)
        diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
        voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
    #
    device = torch.device('cuda:'+str(args.cuda) if args.cuda > -1 else 'cpu')
    #
    model = HGDR(voc_size, ddi_adj_path, EHR_path, device=device, args=args)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epoch_begin = 0
    ##
    with open(ddi_adj_path, 'rb') as f: # currently not used (ddi loss)
        ddi_adj = dill.load(f)
    ddi_adj_path = ddi_adj # change path to the ddi matrix to save I/O time
    #
    args.resume_path = str(run_dir / 'best.model')
    # load model from save checkpoint
    if args.Test or args.load:
        checkpoint = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_begin = checkpoint['epoch'] + 1
        print(f"Load {args.resume_path} finish...")

    if args.Test:
        model.to(device=device)
        
        tic = time.time()
        print('--------------------Begin Testing--------------------')
        ddi_list, ja_list, prauc_list, f1_list, med_list = [], [], [], [], []
        tic, result, sample_size = time.time(), [], round(len(data_test) * 0.8)
        # np.random.seed(0)
        for _ in range(10):
            test_sample_indices = np.random.choice(len(data_test), sample_size, replace=True)
            test_sample = [data_test[i] for i in test_sample_indices]
            cur_test_data = EHR_Test_Dataset(test_sample, voc_size)
            test_loader = DataLoader(
                cur_test_data, 128, False, num_workers=16, pin_memory=True
            )
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, test_loader, ddi_adj_path, device)
            result.append([ddi_rate, ja, avg_f1, prauc, avg_med])
        result = np.array(result)
        mean, std = result.mean(axis=0), result.std(axis=0)
        metric_list = ['ddi_rate', 'ja', 'avg_f1', 'prauc', 'med']
        outstring = ''.join([
            "{}:\t{:.4f} $\\pm$ {:.4f} & \n".format(metric_list[idx], m, s)
            for idx, (m, s) in enumerate(zip(mean, std))
        ])
        print(outstring)
        print ('test time: {}'.format(time.time() - tic))
        return 
        

    # start train iterations
    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    # dataloader for training and evaluation datasets
    train_data = EHR_Train_EVAL_Dataset(data_train, voc_size)
    eval_data = EHR_Train_EVAL_Dataset(data_eval, voc_size)
    #
    train_loader = DataLoader(
        train_data, args.bs, True, num_workers=8, pin_memory=True
    )
    eval_loader = DataLoader(
        eval_data, 128, False, num_workers=16, pin_memory=True
    )
    #
    EPOCH = args.epoch
    for epoch in range(epoch_begin, EPOCH):
        tic = time.time()
        print ('\nepoch {} --------------------------'.format(epoch))
        
        model.train()
        step = 0
        trian_visit_num = len(train_data)
        
        epoch_loss_ddi = []
        #epoch_loss_se = []
        epoch_loss_bce = []
        epoch_loss_multi = []
        epoch_loss_total = []

        for cur_batch in train_loader:
            cur_diag, cur_pro, cur_med, cur_med_bce, cur_med_ml, cur_len = cur_batch
            # move current batch to device
            cur_diag = cur_diag.to(device)
            cur_pro = cur_pro.to(device)
            cur_med = cur_med.to(device)
            cur_med_bce_target = cur_med_bce.to(device)
            cur_med_ml_target = cur_med_ml.to(device)
            cur_len = cur_len.to(device)
            #
            # run model (forward)
            #result, loss_ddi, loss_se = model(cur_diag, cur_pro, cur_med, cur_len) # (B, voc_size[2]), probability for each medicine
            result, loss_ddi = model(cur_diag, cur_pro, cur_med, cur_len) # (B, voc_size[2]), probability for each medicine
            # loss
            loss_bce = F.binary_cross_entropy_with_logits(result, cur_med_bce_target)
            loss_multi = F.multilabel_margin_loss(torch.sigmoid(result), cur_med_ml_target)
            # final loss
            loss_pred = args.alpha * loss_bce + (1 - args.alpha) * loss_multi
            #
            if args.ddi:
                labellist = []
                for i in range(cur_med_ml_target.shape[0]):
                    cur = torch.nonzero(cur_med_ml_target[i])[:, 0].tolist()
                    labellist.append(cur)
                cur_ddi_rate = ddi_rate_score([labellist], ddi_adj_path)
                if cur_ddi_rate > args.target_ddi:   # 如果当前ddi率大于目标ddi率，则加入ddi loss
                    loss = args.beta * loss_pred + (1 - args.beta) * loss_ddi
                else:
                    loss = loss_pred
            else:
                loss = loss_pred
            #
            #loss = loss + loss_se if args.ddi2 else loss
            epoch_loss_ddi.append(loss_ddi)
            #epoch_loss_se.append(loss_se)
            epoch_loss_bce.append(loss_bce)
            epoch_loss_multi.append(loss_multi)
            epoch_loss_total.append(loss)
            # BP
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += cur_diag.shape[0]
            llprint('\rtraining step: {} / {}'.format(step, trian_visit_num))
        #'''
        print()
        tic2 = time.time()        
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, eval_loader, ddi_adj_path, device)
        print('training time: {}, eval time: {}'.format(tic2 - tic, time.time() - tic2))
        print('evaluation result: ddi_rate:{}, ja:{}, PRAUC:{}, avg_f1:{}, avg_med:{}'.format(ddi_rate, ja, prauc, avg_f1, avg_med))
        
        # record metrics and losses
        history['ddi_rate'].append(ddi_rate)
        history['ja'].append(ja)        
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)
        history['med'].append(avg_med)

        history['loss_ddi'].append(sum(epoch_loss_ddi)/len(epoch_loss_ddi))
        #history['loss_se'].append(sum(epoch_loss_se)/len(epoch_loss_se))
        history['loss_bce'].append(sum(epoch_loss_bce)/len(epoch_loss_bce))
        history['loss_multi'].append(sum(epoch_loss_multi)/len(epoch_loss_multi))
        history['loss_total'].append(sum(epoch_loss_total)/len(epoch_loss_total))

        #print("average losses of the last epoch (overall): total:{:.4f}, bce:{:.4f}, multi:{:.4f}, ddi:{:.4f}, se:{:.4f}".format(
        print("average losses of the last epoch (overall): total:{:.4f}, bce:{:.4f}, multi:{:.4f}, ddi:{:.4f}".format(
            history['loss_total'][-1],
            history['loss_bce'][-1],
            history['loss_multi'][-1],
            history['loss_ddi'][-1]#,
            #history['loss_se'][-1]
        ))

        print("averages of various evaluation metrics over the last five epoches:")
        if epoch >= 5:
            print ('ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}'.format(
                np.mean(history['ddi_rate'][-5:]),
                np.mean(history['med'][-5:]),
                np.mean(history['ja'][-5:]),
                np.mean(history['avg_f1'][-5:]),
                np.mean(history['prauc'][-5:])
                ))
        #'''
        savefile = str(run_dir / ('Epoch_%d.model' % epoch))
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch}, open(savefile, 'wb'))
        #'''
        if best_ja < ja:
            best_epoch = epoch
            best_ja = ja
            savefile = str(run_dir / 'best.model')
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch}, open(savefile, 'wb'))

        print ('best_epoch: {}'.format(best_epoch))

    dill.dump(history, open(str(run_dir / 'history_{}.pkl'.format(args.model_name)), 'wb'))

if __name__ == '__main__':
    main()
