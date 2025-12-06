'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-01-31 12:21:35
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-04-18 19:12:57
FilePath: /mycode/paper1/ood_utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
import torch.nn.functional as F
import numpy as np
from torch.nn.modules import Module
from tqdm import tqdm
import time
import sklearn.covariance
from torch.autograd import Variable
import math


def ctrans_feature_list(model,x):
    feature=[]
    x=model.patch_embed(x)
    x = model.pos_drop(x)
    #x = model.layers(x)#特征在这里
    feature.append(x)
    for layer in model.layers:
        x = layer(x)
        feature.append(x)
    #x = model.norm(x)  # B L C
    #x = model.head(x)
    return feature
def soft_output(model,data,temperature,magnitude,net_type,criterion):#,batch_output
    batch_output = model(data)
    outputs = batch_output / temperature
    labels = outputs.data.max(1)[1]
    #labels = labels.requires_grad_(True)
    model.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    
    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(data.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2#变为正负1
    if net_type == 'densenet':
        #gradient.index_select将第0列的元素除以(63.0/255.0)，
        #index_copy_将上一步得到的结果复制回原始梯度张量的第0列，表示直接修改原始张量而不创建新的张量
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
    elif net_type == 'resnet' or net_type== 'retccl':
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
    else:
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))

    tempInputs = torch.add(data.data, gradient, alpha=-magnitude)
    outputs = model(tempInputs)
    outputs = outputs / temperature
    soft_out = F.softmax(outputs, dim=1)
    soft_out, _ = torch.max(soft_out, dim=1)
    return soft_out
    
def test_probability_file(model,test_loader,T_list,M_list,net_type,outputfile,oodname='',id_flag=True,dataset_name=''):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total = 0
    np.random.seed(0)
    
    if id_flag == True:   
        temp_file_name_test = '%s/Probability_test_In.txt'%(outputfile)
    else:
        temp_file_name_test = '%s/Probability_test_Out_%s.txt'%(outputfile,oodname)
    f = open(temp_file_name_test, 'w')
    
    t0=time.time()
    T_M=[(T_list[0], M_list[0]),(T_list[1], M_list[1])]
    #对数据进行多次提取
    item_t=0
    for temperature,magnitude in T_M:
        item_t+=1
        item=0
        print('第%d个参数'%(item_t))
        for data, _ in test_loader:
            
            '''
            if item%10000==1:
                print('第%d组数据，100组数据总共花费了%.2f秒'%(item,time.time()-t0))
                t0=time.time()'''
            
            #data = data.cuda()
            data = data.to('cuda')
            data.requires_grad = True
            
            #print("Current memory allocated1:", torch.cuda.memory_allocated())
            out=soft_output(model,data,temperature,magnitude,net_type,criterion)
                
                
            #print("Current memory allocated2:", torch.cuda.memory_allocated())
            if item==0:
                soft_out=out.detach().cpu().numpy()
            else:
                soft_out=np.concatenate((soft_out,out.detach().cpu().numpy()),axis=0)
            '''
            if item%2500==1:
                print('第%d组数据，数据前面花费了%.2f秒，后面花费了%.3f秒'%(item,t1,time.time()-t0-t1))
                t0=time.time()'''
            item+=1
        t1=time.time()-t0
        print('test datastet第%d组数据，花费了%.2f秒'%(item,t1))
        t0=time.time()
        if item_t==1:
            matrix_soft=soft_out.reshape((soft_out.size,1))
        else:
            matrix_soft=np.concatenate((matrix_soft,soft_out.reshape((soft_out.size,1))),axis=1)
            #matrix_soft.append(out.numpy())
    for i in range(matrix_soft.shape[0]):
        for col in range(matrix_soft.shape[1]):
            f.write("{} ".format(matrix_soft[i,col]))
        f.write('\n')
    f.close()



def probability_file(model,test_loader,T_list,M_list,net_type,outputfile,oodname='',id_flag=True,dataset_name=''):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total = 0
    np.random.seed(12)
    
    if id_flag == True:
        temp_file_name_val = '%s/Probability_val_In.txt'%(outputfile)
        temp_file_name_test = '%s/Probability_test_In.txt'%(outputfile)
    else:
        temp_file_name_val = '%s/Probability_val_Out_%s.txt'%(outputfile,oodname)
        temp_file_name_test = '%s/Probability_test_Out_%s.txt'%(outputfile,oodname)
        
    g = open(temp_file_name_val, 'w')
    f = open(temp_file_name_test, 'w')
    
    t0=time.time()
    T_M=[(a, b) for a in T_list for b in M_list]
    #对数据进行多次提取
    item_t=0
    for temperature,magnitude in T_M:
        item_t+=1
        item=0
        print('第%d个参数'%(item_t))
        for data, _ in test_loader:
            
            '''
            if item%10000==1:
                print('第%d组数据，100组数据总共花费了%.2f秒'%(item,time.time()-t0))
                t0=time.time()'''
            
            #data = data.cuda()
            data = data.to('cuda')
            data.requires_grad = True
            
            #print("Current memory allocated1:", torch.cuda.memory_allocated())
            out=soft_output(model,data,temperature,magnitude,net_type,criterion)
                
                
            #print("Current memory allocated2:", torch.cuda.memory_allocated())
            if item==0:
                soft_out=out.detach().cpu().numpy()
            else:
                soft_out=np.concatenate((soft_out,out.detach().cpu().numpy()),axis=0)
            '''
            if item%2500==1:
                print('第%d组数据，数据前面花费了%.2f秒，后面花费了%.3f秒'%(item,t1,time.time()-t0-t1))
                t0=time.time()'''
            item+=1
        t1=time.time()-t0
        print('第%d组数据，花费了%.2f秒'%(item,t1))
        t0=time.time()
        if item_t==1:
            matrix_soft=soft_out.reshape((soft_out.size,1))
        else:
            matrix_soft=np.concatenate((matrix_soft,soft_out.reshape((soft_out.size,1))),axis=1)
            #matrix_soft.append(out.numpy())

    if dataset_name=='imagenet':
        matrix_soft_shape = matrix_soft.shape[0]
        random_numbers = np.arange(matrix_soft_shape)
    else:
        random_numbers = np.random.choice(range(matrix_soft.shape[0]), size=1000, replace=False)



    for i in range(matrix_soft.shape[0]):
        if i in random_numbers:
            for col in range(matrix_soft.shape[1]):
                g.write("{} ".format(matrix_soft[i,col]))
            g.write('\n')
        else:
            for col in range(matrix_soft.shape[1]):
                f.write("{} ".format(matrix_soft[i,col]))
            f.write('\n')
    f.close()
    g.close()

def get_curve(dir_name,oodname, stypes = ['Baseline', 'Gaussian_LDA']):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    fpr_at_tpr95 = dict()
    for stype in stypes:
        known = np.loadtxt('{}/Probability_{}_In.txt'.format(dir_name, stype))#, delimiter='\\n')#output_path/Probability_val_In.txt
        novel = np.loadtxt('{}/Probability_{}_Out_{}.txt'.format(dir_name, stype, oodname))#, delimiter='\\n')
    #known.sort()
    #novel.sort()
    if known.ndim==1:
        known = known.reshape((known.shape[0], 1))
        novel = novel.reshape((novel.shape[0], 1))
    known_sorted = np.sort(known, axis=0)
    novel_sorted = np.sort(novel, axis=0)
    num_para=known.shape[1]
    for i in range(num_para):#对每一列数据进行分析
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp[i] = -np.ones([num_k+num_n+1], dtype=int)#都为-1
        fp[i] = -np.ones([num_k+num_n+1], dtype=int)
        tp[i][0], fp[i][0] = num_k, num_n#第1位用来存放数量
        k, n = 0, 0
        for l in range(num_k+num_n):
            if k == num_k:
                tp[i][l+1:] = tp[i][l]
                fp[i][l+1:] = np.arange(fp[i][l]-1, -1, -1)
                break
            elif n == num_n:
                tp[i][l+1:] = np.arange(tp[i][l]-1, -1, -1)
                fp[i][l+1:] = fp[i][l]
                break
            else:
                if novel_sorted[n][i] < known_sorted[k][i]:
                    n += 1
                    tp[i][l+1] = tp[i][l]
                    fp[i][l+1] = fp[i][l] - 1
                else:
                    k += 1
                    tp[i][l+1] = tp[i][l] - 1
                    fp[i][l+1] = fp[i][l]
        tpr95_pos = np.abs(tp[i] / num_k - .95).argmin()
        #fpr_at_tpr95[i]=fp[i][tpr95_pos] / num_n
        tnr_at_tpr95[i] = 1. - fp[i][tpr95_pos] / num_n
        
        '''if stypes[0]=='diff_val':
            print('FP%d为:'%i,end=' ')
            print(fp[i],tp[i])'''
            





    return tp, fp, tnr_at_tpr95,  num_para

def metric(dir_name, oodname, stypes = ['Bas', 'Gau'], verbose=False):
    tp, fp, tnr_at_tpr95, num_para = get_curve(dir_name,oodname, stypes)
    results = dict()
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    if verbose:
        print('      ', end='')
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('')
    
    for i in range(num_para):
        results[i] = dict()
        # FPR
        mtype = 'FPR'
        results[i][mtype] = 1.-tnr_at_tpr95[i]
        # TNR
        mtype = 'TNR'
        results[i][mtype] = tnr_at_tpr95[i]
        # AUROC
        mtype = 'AUROC'
        tpr = np.concatenate([[1.], tp[i]/tp[i][0], [0.]])#用于将不同部分的数据连接起来，这里是将比率数组的头部和尾部分别加上1和0，使得ROC曲线起始于点 (0,0) 终止于点 (1,1)。
        fpr = np.concatenate([[1.], fp[i]/fp[i][0], [0.]])
        results[i][mtype] = -np.trapz(1.-fpr, tpr)#所确定的区域的积分值
        # DTACC
        mtype = 'DTACC'
        results[i][mtype] = .5 * (tp[i]/tp[i][0] + 1.-fp[i]/fp[i][0]).max()
        # AUIN
        mtype = 'AUIN'
        denom = tp[i]+fp[i]
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp[i]/denom, [0.]])
        results[i][mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])
        # AUOUT
        mtype = 'AUOUT'
        denom = tp[i][0]-tp[i]+fp[i][0]-fp[i]
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[i][0]-fp[i])/denom, [.5]])
        results[i][mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind]) 
    return results

def covaraiance_mean(model,network_name, num_classes, feature_list, train_loader,dir_name):
    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)#计算原始数据的协方差矩阵而不是中心化后的数据
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)#每个类别的样本数量
    num_sample_per_class.fill(0)
    list_features = []

    t0=time.time()
    for i in range(num_output):#计算总共需要计算几层的特征
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)
    num_breakpoint=0
    braekpoint_number=3000
    
    concat_parts = [[[] for _ in range(len(feature_list))] for _ in range(num_classes)]
    
    for data, target in train_loader:
        num_breakpoint+=1
        #if num_breakpoint==5:
        #    break
        
        total += data.size(0)
        data = data.cuda()
        #data = Variable(data, volatile=True)
        #output, out_features = model.feature_list(data)
        #每个模型的feature_list不同
        temp_x=data
        out_features = []
        '''
        if network_name=='ctranspath':
            for name, module in model.named_children():
                if name=='head':
                    continue
                temp_x = module(temp_x)
                temp_x_cpu=temp_x.cpu()
                out_features.append(temp_x_cpu)#.cpu().detach().numpy())'''
        '''
            mid_feature = model.feature_forward(temp_x)#out1是最浅层的特征。out1, out2, out3
            for out in mid_feature:
                temp_x_cpu=out.cpu()
                out_features.append(temp_x_cpu)'''
        if network_name == 'ctranspath':
            temp_list=ctrans_feature_list(model,temp_x)
            for out_temp in temp_list:
                temp_x_cpu=out_temp.cpu()
                out_features.append(temp_x_cpu)
        elif network_name in ['resnet','retccl']:
            temp_list = model.feature_list(temp_x)[1]
            for out_temp in temp_list:
                temp_x_cpu=out_temp.cpu()
                out_features.append(temp_x_cpu)
        elif network_name in ['ibotvit','vit']:
            temp_list = model.get_intermediate_layers(temp_x,5)
            for out_temp in temp_list:
                temp_x_cpu=out_temp.cpu()
                out_features.append(temp_x_cpu)
        output=model(data)




        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)#计算特征张量沿着第3个维度的平均值
            
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.cuda()).cpu()
        correct += equal_flag.sum()
        '''
        if num_breakpoint%braekpoint_number==0:
            t1=time.time()-t0
            t2=time.time()
        
        # construct the sample matrix方法1
        for i in range(data.size(0)):
            label = target[i]
            
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:#5个具体的特征
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                    = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)#列表拼接
                    out_count += 1                
            num_sample_per_class[label] += 1'''
        #方法2：频繁地对列表进行拼接操作确实可能会很慢，因为每次拼接都需要重新分配内存
        #      一个更高效的方法是先将要拼接的部分存储在一个列表中，然后使用 torch.cat 一次性将所有部分拼接起来。
        
        for i in range(data.size(0)):
            label = target[i]
            #label_number.append(label)
            out_count = 0
            # 对每个特征进行处理
            for out in out_features:#out是第一个特征，有64个通道数
                # 将当前 out[i] 转换为形状为 (1, -1) 的张量，并添加到 concat_parts 中
                concat_parts[label][out_count].append(out[i].view(1, -1))
                out_count += 1

    print('计算均值与协方差，花费了%.2f秒，'%(time.time()-t0))#后面花费了%.2f
    #print(num_breakpoint*16,t1,time.time()-t1)
    t0=time.time()
    out_count = 0
    t1=time.time()-t0
    t2=time.time()
    for out in out_features:
        for i in range(num_classes):
            list_features[out_count][i]=torch.cat(concat_parts[i][out_count],dim=0)
        out_count += 1
        
    
    #print('cat操作花费了%.2f'%(time.time()-t2))
    #print(num_breakpoint*16,t1,time.time()-t1)
    t0=time.time()
    temp_file_name_covaraiance = '%s/covaraiance.txt'%(dir_name)
    temp_file_name_mean = '%s/mean.txt'%(dir_name)
    g = open(temp_file_name_covaraiance, 'w')
    f = open(temp_file_name_mean, 'w')
    sample_class_mean = []#存储每个类别的特征均值
    out_count = 0
    for num_feature in feature_list:#总共5层特征
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda() #类别*通道数
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)#list_features包含有5层特征，2个类，中的的特征
            f.write(' '.join(map(str, temp_list[j].tolist())) + ' ')
        f.write('\n')
        sample_class_mean.append(temp_list)#是一个5*2的列表

        out_count += 1
        
    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            #list_features[k][i] = list_features[k][i].to('cuda')
            sample_class_mean_cpu=sample_class_mean[k][i].cpu()
            if i == 0:
                X = list_features[k][i] - sample_class_mean_cpu#都放到gpu上运算
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean_cpu), 0)
                
        # find inverse            
        group_lasso.fit(X.numpy())#X.cpu().numpy()
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()



        temp_precision_numpy = temp_precision.cpu().numpy()
        temp_precision_cuda = torch.from_numpy(temp_precision_numpy).float().cuda()
        temp_precision_list = temp_precision_cuda.tolist()
        #temp_precision_list=[[1,2,3],[4,5,6],[7,8,9]]
        temp_precision_str = ' '.join(map(str, temp_precision_list))
        
        g.write(temp_precision_str+'\n')
        precision.append(temp_precision)
        
    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))
    f.close()
    g.close()
    return sample_class_mean, precision



def Mahalanobis_score(model,network_name, test_loader, num_classes, outf, out_flag, net_type, sample_mean, precision, layer_index, magnitude, oodname='C16'):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    model.eval()
    Mahalanobis = []
    
    if out_flag == True:
        temp_file_name = '%s/confidence_Ma%s_In.txt'%(outf, str(layer_index))
    else:
        temp_file_name = '%s/confidence_Ma%s_Out_%s.txt'%(outf, str(layer_index), oodname)
        
    g = open(temp_file_name, 'w')
    t0=time.time()
    num_breakpoint=0
    for data, target in test_loader:
        num_breakpoint+=1
        #if num_breakpoint==2:
        #    break
        
        data, target = data.cuda(), target.cuda()
        data.requires_grad = True
        temp_x=data
        out_features = []
        
        #mid_feature = model.feature_forward(temp_x)#out1是最浅层的特征。out1, out2, out3
        #temp_x=mid_feature[layer_index]
        out_features = []
        '''i=0
        if network_name=='ctranspath':
            for name, module in model.named_children():
                if i==layer_index+1:
                    break
                if name=='head':
                    continue
                temp_x = module(temp_x)
                i+=1'''
        if network_name == 'ctranspath':
            temp_list=ctrans_feature_list(model,temp_x)
            temp_x=temp_list[layer_index]
        elif network_name in ['resnet', 'retccl']:
            temp_list = model.feature_list(temp_x)[1]
            temp_x=temp_list[layer_index]
        elif network_name in ['ibotvit','vit']:
            temp_list = model.get_intermediate_layers(temp_x,5)
            temp_x=temp_list[layer_index]
        
        out_features=temp_x
        #out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)
        
        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]#第 层特征、第 个类的均值
            zero_f = out_features.data - batch_sample_mean#减去均值
            zero_f=zero_f.cuda()
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()#mm是相乘，precision是协方差，表示了样本之间的马氏距离
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)#按照列进行拼接
        
        # Input_processing
        sample_pred = gaussian_score.max(1)[1]#对应着一个batch的结果
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)#index_select 方法允许在指定维度上根据索引值选择元素   目的是从样本均值中选择特定样本的均值
        zero_f = out_features -  (batch_sample_mean)
        pure_gau = -0.5*torch.mm(torch.mm(zero_f,  (precision[layer_index])), zero_f.t()).diag()
        model.zero_grad()
        loss = torch.mean(-pure_gau)
        loss.backward()
        
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        if net_type == 'densenet':
            #gradient.index_select将第0列的元素除以(63.0/255.0)，
            #index_copy_将上一步得到的结果复制回原始梯度张量的第0列，表示直接修改原始张量而不创建新的张量
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
        elif net_type == 'resnet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
        else:
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
        tempInputs = torch.add(data.data, gradient,alpha= -magnitude)
        temp_x_noise=tempInputs
        #mid_feature = model.feature_forward(temp_x_noise)#out1是最浅层的特征。out1, out2, out3
        #temp_x_noise=mid_feature[layer_index]
        '''i=0
        if network_name=='ctranspath':
            for name, module in model.named_children():
                if i==layer_index+1:
                    break
                if name=='head':
                    continue
                temp_x_noise = module(temp_x_noise)
                i+=1'''
        if network_name == 'ctranspath':
            temp_list=ctrans_feature_list(model,temp_x_noise)
            temp_x_noise=temp_list[layer_index]        
        elif network_name in ['resnet','retccl']:
            temp_list = model.feature_list(temp_x_noise)[1]
            temp_x_noise=temp_list[layer_index]
        elif network_name in ['ibotvit','vit']:
            temp_list = model.get_intermediate_layers(temp_x_noise,5)
            temp_x_noise=temp_list[layer_index]

        noise_out_features=temp_x_noise
        
        #out_features = model.intermediate_forward(data, layer_index)
        #noise_out_features = model.intermediate_forward((tempInputs), layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(noise_gaussian_score.cpu().numpy())
        
        for i in range(data.size(0)):
            g.write("{}\n".format(noise_gaussian_score[i]))
    
    print('增强%.6f，第%d个特征，前面花费了%.2f秒，'%(magnitude, layer_index, time.time()-t0))
    #print(num_breakpoint,time.time()-t0)
    t0=time.time()
    g.close()

    return Mahalanobis



def generate_labels(X_pos, X_neg):#这里是作者的失误，实际上X_pos输入的是ood数据
    """
    merge positve and nagative artifact and generate labels
    return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y



def block_split(X,Y,out):
    num_samples = X.shape[0]
    partition = math.floor(num_samples * 0.25)
    
    X_adv, Y_adv = X[:partition], Y[:partition]
    X_norm, Y_norm = X[partition: :], Y[partition: :]
    num_train = 1000

    X_train = np.concatenate((X_norm[:num_train], X_adv[:num_train]))
    Y_train = np.concatenate((Y_norm[:num_train], Y_adv[:num_train]))

    X_test = np.concatenate((X_norm[num_train:], X_adv[num_train:]))
    Y_test = np.concatenate((Y_norm[num_train:], Y_adv[num_train:]))

    return X_train, Y_train, X_test, Y_test

def get_characteristics(score, dataset, out, outf):
    X, Y = None, None
    
    file_name = os.path.join(outf, "%s_%s_%s.npy" % (score, dataset, out))
    data = np.load(file_name)
    np.random.seed(40)
    np.random.shuffle(data)
    if X is None:
        X = data[:, :-1]
    else:
        X = np.concatenate((X, data[:, :-1]), axis=1)
    if Y is None:
        Y = data[:, -1] # labels only need to load once
         
    return X, Y



def detection_performance( Y,y, outf,out):
    num_samples = y.shape[0]
    l1 = open('%s/Probability_TMP_In.txt'%outf, 'w')
    l2 = open('%s/Probability_TMP_Out_%s.txt'%(outf,out), 'w')
    for i in range(num_samples):
        if Y[i] == 0:
            for j in range(y.shape[1]):
                l1.write("{} ".format(-y[i][j]))#-y[i][j])
            l1.write("\n")
        else:
            for j in range(y.shape[1]):
                l2.write("{} ".format(-y[i][j]))#y[i][j])
            l2.write("\n")
            
    l1.close()
    l2.close()
    results = metric(outf,out, ['TMP'])
    return results


def single_feature_detection_performance(Y,X,  outf,out):
    num_samples = X.shape[0]
    l1 = open('%s/Probability_STMP_In.txt'%outf, 'w')
    l2 = open('%s/Probability_STMP_Out_%s.txt'%(outf,out), 'w')
    num_count=0
    for i in range(num_samples):
        if Y[i] == 0:
            for j in range(X.shape[1]):
                l1.write("{} ".format(X[i][j]))#X[i][j]本身就是负数
            l1.write("\n")
        else:
            for j in range(X.shape[1]):
                l2.write("{} ".format(X[i][j]))#y[i][j])
            l2.write("\n")
            
    l1.close()
    l2.close()
    results = metric(outf,out, ['STMP'])
    return results













