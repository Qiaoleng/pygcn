import time
import argparse
import numpy as np

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import models
import smote_utils
from utils2 import load_data, accuracy
import data_load
import random
import ipdb
import copy
import os
import xlsxwriter

# biggnn
parser2 = argparse.ArgumentParser()  # 创建一个解析器
parser2.add_argument('--batch_size', type=int, default=4, help='batch_size')
parser2.add_argument('--hidden_size', type=int, default=66, help='隐藏层的大小，小型网络一般在64， 128， 256')
parser2.add_argument('--graph_direction', type=str, default='all', help='边的方向')
parser2.add_argument('--message_function', type=str, default='no_edge',
                     help='message_function传递函数')  # 分为edge_mm、edge_network、edge_pair、no_edge
parser2.add_argument('--graph_hops', type=int, default=3, help='图神经网络的跳数')
parser2.add_argument('--word_dropout', type=float, default=0.5, help='丢弃词向量，一般在0~1之间')

parser = smote_utils.get_parser()
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:",device)

config = parser2.parse_args()

# from torch.utils.tensorboard import SummaryWriter

foward = "/HOME/scz0657/run/qy/GraphSmote-main/data/"
for j in range(1, 17):
    if (j == 1):
        file = foward+"camel/camel-1.4.0/"
        name = "camel-1.4.0(1)"
    if (j == 2):
        file = foward+"groovy/groovy-1.5.7/"
        name = "groovy-1.5.7(1)"
    if (j == 3):
        file = foward+"hive/hive-0.9.0/"
        name = "hive-0.9.0(1)"
    if (j == 4):
        file = foward+"jruby/jruby-1.1/"
        name = "jruby-1.1(1)"
    if (j == 5):
        file = foward+"lucene/lucene-2.3.0/"
        name = "lucene-2.3.0(1)"
    if (j == 6):
        file = foward+"activemq/activemq-5.0.0/"
        name = "activemq-5.0.0(1)"
    if (j == 7):
        file = foward+"camel/camel-2.9.0/"
        name = "camel-2.9.0(1)"
    if (j == 8):
        file = foward+"camel/camel-2.10.0/"
        name = "camel-2.10.0(1)"
    if (j == 9):
        file = foward+"camel/camel-2.11.0/"
        name = "camel-2.11.0(1)"
    if (j == 10):
        file = foward+"activemq/activemq-5.1.0/"
        name = "activemq-5.1.0(1)"
    if (j == 11):
        file = foward+"activemq/activemq-5.3.0/"
        name = "activemq-5.3.0(1)"
    if (j == 12):
        file = foward+"activemq/activemq-5.2.0/"
        name = "activemq-5.2.0(1)"
    if (j == 13):
        file = foward+"activemq/activemq-5.8.0/"
        name = "activemq-5.8.0(1)"
    if (j == 14):
        file = foward+"jruby/jruby-1.4.0/"
        name = "jruby-1.4.0(1)"
    if (j == 15):
        file = foward+"jruby/jruby-1.5.0/"
        name = "jruby-1.5.0(1)"
    if (j == 16):
        file = foward+"lucene/lucene-2.9.0/"
        name = "lucene-2.9.0(1)"
    # 选文件
    choose = "file_dependencies"
    choose2 = "final_weight_edge"

    # Training setting
    parser = smote_utils.get_parser()

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    '''
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    '''

    # Load data
    # try:
    adj, features, labels, idx_train, idx_val, idx_test, adj_in, adj_out = load_data(file, name, choose)
    adj2, features2, labels2, idx_train2, idx_val2, idx_test2, adj_in2, adj_out2 = load_data(file, name, choose2)
    
        
    #except RuntimeError as e:
        #print(e)
        #break
    class_sample_num = 20
    im_class_num = 2
    config.hidden_size = features.size(-1)

    # for artificial imbalanced setting: only the last im_class_num classes are imbalanced
    c_train_num = []
    for i in range(labels.max().item() + 1):
        if args.imbalance and i > labels.max().item() - im_class_num:  # only imbalance the last classes
            c_train_num.append(int(class_sample_num * args.im_ratio))

        else:
            c_train_num.append(class_sample_num)

    # get train, validatio, test data split
    if args.dataset == 'BlogCatalog':
        idx_train1, idx_val1, idx_test1, class_num_mat = smote_utils.split_genuine(labels)
    elif args.dataset == 'cora':
        idx_train1, idx_val1, idx_test1, class_num_mat = smote_utils.split_arti(labels, c_train_num)
    elif args.dataset == 'twitter':
        idx_train1, idx_val1, idx_test1, class_num_mat = smote_utils.split_genuine(labels)

    # method_1: oversampling in input domain
    if args.setting == 'upsampling':
        adj, adj_in, adj_out, features, labels, idx_train1 = smote_utils.src_upsample(adj, adj_in, adj_out, features,
                                                                                labels, idx_train,
                                                                                portion=args.up_scale,
                                                                                im_class_num=im_class_num)
        adj2, adj_in2, adj_out2, features2, labels2, idx_train1 = smote_utils.src_upsample(adj2, adj_in2, adj_out2, features2,
                                                                            labels2, idx_train2,
                                                                            portion=args.up_scale,
                                                                            im_class_num=im_class_num)
        
    if args.setting == 'smote':
        adj, adj_in, adj_out, features, labels, idx_train1 = smote_utils.src_smote(adj,  adj_in, adj_out,features, labels, idx_train, portion=args.up_scale,im_class_num=im_class_num)
        adj2, adj_in2, adj_out2, features2, labels2, idx_train2 = smote_utils.src_smote(adj2,  adj_in2, adj_out2,features2, labels2, idx_train2, portion=args.up_scale,im_class_num=im_class_num)
        print(features.shape, features2.shape)

    # Model and optimizer
    # if oversampling in the embedding space is required, model need to be changed
    if args.setting != 'embed_up':
        if args.model == 'sage':
            encoder = models.Sage_En(nfeat=features.shape[1],
                                    nhid=args.nhid,
                                    nembed=args.nhid,
                                    dropout=args.dropout)
            classifier = models.Sage_Classifier(nembed=args.nhid,
                                                nhid=args.nhid,
                                                nclass=labels.max().item() + 1,
                                                dropout=args.dropout)
        elif args.model == 'gcn':
            encoder = models.GCN_En(nfeat=features.shape[1],
                                    nhid=args.nhid,
                                    nembed=args.nhid,
                                    dropout=args.dropout)
            classifier = models.GCN_Classifier(nembed=args.nhid,
                                            nhid=args.nhid,
                                            nclass=labels.max().item() + 1,
                                            dropout=args.dropout)
        elif args.model == 'gat':
            encoder = models.GAT_En(nfeat=features.shape[1],
                                    nhid=args.nhid,
                                    nembed=args.nhid,
                                    dropout=args.dropout)
            classifier = models.GAT_Classifier(nembed=args.nhid,
                                            nhid=args.nhid,
                                            nclass=labels.max().item() + 1,
                                            dropout=args.dropout)
        elif args.model == 'BiGGNN':
            encoder = models.BiGGNN(config)
            classifier = models.Classifier(nembed=config.hidden_size,
                                           nhid=config.hidden_size,
                                           nclass=labels.max().item() + 1,
                                           dropout=args.dropout)
    else:
        if args.model == 'sage':
            encoder = models.Sage_En2(nfeat=features.shape[1],
                                    nhid=args.nhid,
                                    nembed=args.nhid,
                                    dropout=args.dropout)
            classifier = models.Classifier(nembed=args.nhid,
                                        nhid=args.nhid,
                                        nclass=labels.max().item() + 1,
                                        dropout=args.dropout)
        elif args.model == 'gcn':
            encoder = models.GCN_En2(nfeat=features.shape[1],
                                    nhid=args.nhid,
                                    nembed=args.nhid,
                                    dropout=args.dropout)
            classifier = models.Classifier(nembed=args.nhid,
                                        nhid=args.nhid,
                                        nclass=labels.max().item() + 1,
                                        dropout=args.dropout)
        elif args.model == 'GAT':
            encoder = models.GAT_En2(nfeat=features.shape[1],
                                    nhid=args.nhid,
                                    nembed=args.nhid,
                                    dropout=args.dropout)
            classifier = models.Classifier(nembed=args.nhid,
                                        nhid=args.nhid,
                                        nclass=labels.max().item() + 1,
                                        dropout=args.dropout)

    # 新图生成
    decoder = models.Decoder(nembed=config.hidden_size,
                            dropout=args.dropout)

    optimizer_en = optim.Adam(encoder.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    optimizer_cls = optim.Adam(classifier.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    optimizer_de = optim.Adam(decoder.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    fusion = models.EmbedFusion(input_size=66, hidden_size=args.nhid, num_layers=3)
    contrastive_loss = models.ContrastiveLoss(tau=0.1)


    if args.cuda:
        encoder = encoder.cuda()
        classifier = classifier.cuda()
        decoder = decoder.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        adj_in = adj_in.cuda()
        adj_out = adj_out.cuda()
        features2 = features2.cuda()
        adj2 = adj2.cuda()
        labels2 = labels2.cuda()
        idx_train2 = idx_train2.cuda()
        idx_val2 = idx_val2.cuda()
        idx_test2 = idx_test2.cuda()
        adj_in2 = adj_in2.cuda()
        adj_out2 = adj_out2.cuda()
        fusion = fusion.cuda()
        contrastive_loss = contrastive_loss.cuda()


    def train(epoch, adj_in, adj_out):
        t = time.time()
        encoder.train()
        classifier.train()
        decoder.train()
        optimizer_en.zero_grad()
        optimizer_cls.zero_grad()
        optimizer_de.zero_grad()

        # 生成嵌入
        edge_vec = []
        embed = encoder(features, edge_vec, adj_in.to_dense(), adj_out.to_dense(), config)
        embed2 = encoder(features2, edge_vec, adj_in2.to_dense(), adj_out2.to_dense(), config)
        embed = embed[0, :, :]
        embed2 = embed2[0, :, :]

        embed_con = add_attention(embed, embed2).cuda()
        output_con = classifier(embed_con, adj)
        loss_con_train = F.cross_entropy(output_con[idx_train], labels[idx_train])
        acc_con_train = smote_utils.accuracy(output_con[idx_train], labels[idx_train])
        loss_con_train.backward()

        # 添加对比学习损失
        loss_contrastive = contrastive_loss(embed, embed2)
        loss_contrastive.backward()

        optimizer_en.step()

        loss_val = F.cross_entropy(output_con[idx_val], labels[idx_val])
        loss_contrastive_val = contrastive_loss(embed[idx_val], embed2[idx_val])
        acc_val = smote_utils.accuracy(output_con[idx_val], labels[idx_val])

        # ipdb.set_trace()
        smote_utils.print_class_acc(output_con[idx_val], labels[idx_val], class_num_mat[:, 1])

        print('Epoch: {:05d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_con_train.item()),
              'acc_train: {:.4f}'.format(acc_con_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))
        return loss_val, loss_contrastive_val, acc_val


    def test(fileload, epoch=0):
        encoder.eval()
        classifier.eval()
        decoder.eval()
        edge_vec = []
        embed = encoder(features, edge_vec, adj_in.to_dense(), adj_out.to_dense(), config)
        embed2 = encoder(features2, edge_vec, adj_in2.to_dense(), adj_out2.to_dense(), config)
        embed = embed[0, :, :]
        embed2 = embed2[0, :, :]
        embed_con = add_attention(embed, embed2).cuda()
        output = classifier(embed_con, adj)
        output_norm = F.softmax(output, dim=1)
        output_int = output.argsort()[:, :2]

        loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
        acc_test = smote_utils.accuracy(output[idx_test], labels[idx_test])
        auc = roc_auc_score(labels[idx_test], output_norm[idx_test][:, 1])

        recall = recall_score(labels[idx_test], output_int[idx_test][:, 1])
        brier = brier_score_loss(labels[idx_test], output_norm[idx_test][:, 1])
        # pop_RF = CE_score(labels_array[idx_test], output_int[idx_test][:, 1])
        pf = fp_rate(labels[idx_test], output_int[idx_test][:, 1])
        precision = precision_score(labels[idx_test], output_int[idx_test][:, 1])

        df = pd.read_excel(fileload)
        numbers = [loss_test.item(), acc_test.item(), auc.item(), recall.item(), brier.item(), pf, precision]
        df.loc[len(df)] = numbers
        df.to_excel(fileload, index=False)

        print_string = "Test set results:" + \
                    "loss= {:.4f}".format(loss_test.item()) + ", " + \
                    "accuracy= {:.4f}".format(acc_test.item()) + ", " + \
                    "auc= {:.4f}".format(auc.item()) + ", " + \
                    "recall= {:.4f}".format(recall.item()) + ", " + \
                    "brier= {:.4f}".format(brier.item()) + ", " + \
                    "fp={:.4f}".format(pf) + ", " + \
                    "precision={:.4f}".format(precision)
        return loss_test, acc_test

        # # 使用 'a' 模式打开文件，这样文件会被追加到末尾
        # with open(file + "score2.txt", "a") as ff:
        #     # 写入指定的 string
        #     ff.write(name + "[" + choose + "]:\n")
        #     ff.write(print_string)
        #     # 在文件末尾写入一个换行符
        #     ff.write("\n")

        print("Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()),
            "auc= {:.4f}".format(auc.item()),
            "recall= {:.4f}".format(recall.item()),
            "brier= {:.4f}".format(brier.item()),
            "pf= {:.4f}".format(pf),
            "precision= {:.4f}".format(precision.item())
            )

        smote_utils.print_class_acc(output[idx_test], labels[idx_test], class_num_mat[:, 2], pre='test')

        '''
        if epoch==40:
            torch
        '''


    def fp_rate(labels, output_int):
        # 初始化 TP、FN、FP 和 TN 的值为 0
        FP = 0
        TN = 0

        # 遍历每一行，统计 TP、FN、FP 和 TN 的值
        for i in range(len(output_int)):
            if output_int[i] == 1 and labels[i] == 0:
                FP += 1
            elif output_int[i] == 0 and labels[i] == 0:
                TN += 1

        # 计算 FP/(TN+FP)
        fp_rate = FP / (TN + FP)
        return fp_rate


    def save_model(epoch):
        saved_content = {}

        saved_content['encoder'] = encoder.state_dict()
        saved_content['decoder'] = decoder.state_dict()
        saved_content['classifier'] = classifier.state_dict()

        torch.save(saved_content,
                'checkpoint/{}/{}_{}_{}_{}.pth'.format(args.dataset, args.setting, epoch, args.opt_new_G,
                                                        args.im_ratio))

        return


    def load_model(filename):
        loaded_content = torch.load('checkpoint/{}/{}.pth'.format(args.dataset, filename),
                                    map_location=lambda storage, loc: storage)

        encoder.load_state_dict(loaded_content['encoder'])
        decoder.load_state_dict(loaded_content['decoder'])
        classifier.load_state_dict(loaded_content['classifier'])

        print("successfully loaded: " + filename)

        return
    
    def add_attention(embed, embed2):
        embed_add = np.concatenate((embed[np.newaxis, :, :], embed2[np.newaxis, :, :]), axis=0)
        embed_add = torch.from_numpy(embed_add)
        q = k = v = embed_add
        alpha = torch.bmm(q, k.transpose(-1, -2))
        alpha = F.softmax(alpha, dim=-1)
        out = torch.bmm(alpha, v)
        out1, out2 = torch.split(out, 1, dim=0)
        out_avg = (out1 + out2) / 2
        out_avg = torch.squeeze(out_avg)
        return out_avg


    # Train model
    if args.load is not None:
        load_model(args.load)

    t_total = time.time()
    # 创建输出表格
    name3= "att_merge"
    if not os.path.exists(foward + "score/" + name3):
            os.mkdir(foward + "score/"+ name3)
    fileload = foward + "score/" + name3+"/" + choose + "_" + name + ".xlsx"
    df = pd.DataFrame(columns=['loss', 'accuracy', 'auc', 'recall', 'brier', 'pf', 'precision'])
    df.to_excel(fileload, index=False)
    # 训练/测试
    # try:
    writer = SummaryWriter(log_dir='/HOME/scz0657/run/qy/GraphSmote-main/data/draw')
    for epoch in range(args.epochs):
        loss_val, loss_contra_val, acc_val = train(epoch, adj_in, adj_out)
        writer.add_scalar(tag="loss_val", # 可以暂时理解为图像的名字
                      scalar_value=loss_val,  # 纵坐标的值
                      global_step=epoch  # 当前是第几次迭代，可以理解为横坐标的值
                      )
        time.sleep(2 * random.uniform(0.5, 1.5))
        writer.add_scalar(tag="loss_contra_val", # 可以暂时理解为图像的名字
                      scalar_value=loss_contra_val,  # 纵坐标的值
                      global_step=epoch  # 当前是第几次迭代，可以理解为横坐标的值
                      )
        time.sleep(2 * random.uniform(0.5, 1.5))
        writer.add_scalar(tag="acc_val", # 可以暂时理解为图像的名字
                      scalar_value=acc_val,  # 纵坐标的值
                      global_step=epoch  # 当前是第几次迭代，可以理解为横坐标的值
                      )
        time.sleep(2 * random.uniform(0.5, 1.5))
        if epoch % 10 == 0:
            loss_test, acc_test = test(fileload, epoch)
            writer.add_scalar(tag="loss_test", # 可以暂时理解为图像的名字
                      scalar_value=loss_test,  # 纵坐标的值
                      global_step=epoch  # 当前是第几次迭代，可以理解为横坐标的值
                      )
            writer.add_scalar(tag="acc_test", # 可以暂时理解为图像的名字
                      scalar_value=acc_test,  # 纵坐标的值
                      global_step=epoch  # 当前是第几次迭代，可以理解为横坐标的值
                      )

        if epoch % 100 == 0:
            save_model(epoch)
    # except RuntimeError as e:
    #     print(e)
    #     break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing

    test(fileload)
