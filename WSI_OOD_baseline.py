

#两种方法：使用统一的magnitude，针对不同的样本使用不同的magnitude
import argparse
import torch
from torch import nn
import numpy as np
import time
import os
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms
from ood_utils.utils import path_Dataset,softmax_file,oodtest_loader,dataset_breast,dataset_lung,paths_labelsDataset,oodTCGA_Gastric,dataset_rare_gastric,dataset_breast_100,dataset_imagenet,imagenetdataset
from ood_utils.utils_Stadis import openood_metric
from ood_utils.utils_Stadis import diff_val_score, diff_test_score,get_oodlabel, diff_test_metric
from ood_utils.utils_WSI_baseline import MDS_val_score,MDS_test_score,covaraiance_mean,ODIN_val_test_score
from ood_utils.utils_WSI_baseline import React_val_test_score,EBO_val_test_score, normal_fea,KNN_val_test_score,React_threshold,uncertainty_score#,ODIN_test_score,React_val_score,React_test_score
from ood_utils.ood_utils import metric
#from ood_utils.ood_utils import probability_file,metric,covaraiance_mean,Mahalanobis_score,generate_labels,get_characteristics,detection_performance,block_split
#from ood_utils.ood_utils import single_feature_detection_performance
from ood_utils.ood_utils import ctrans_feature_list
from models import resnet_ood as resnet
from models import vit,IBOTvit
import sys
from utils.eval_utils import initiate_model,get_simple_loader,get_coords_id_loader
from models import get_encoder
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import pandas as pd

def WSI_Mahalanobis(datasets,ckpt_path,args,out_dataset_list):
    M_list = [0.0, 0.0014,0.0001]
    output_path = os.path.join(args.output_path,args.exp_code,'MDS')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    torch.cuda.manual_seed(1)
    T_list=[1]
    tt=time.time()
    num_ooddataset=len(out_dataset_list)
    shuffle= False#一定得False，否则会影响probability_file
    net = initiate_model(args, ckpt_path)
    train_dataset = datasets[0]
    train_dataset.load_from_h5(True)
    train_dataset.return_slideid()
    train_loader = get_coords_id_loader(train_dataset)

    dataset = datasets[1]
    dataset.load_from_h5(True)
    dataset.return_slideid()
    val_loader = get_coords_id_loader(dataset)
    net.eval()   
    diff_gradient_best_tnr, diff_gradient_best_val=np.zeros(num_ooddataset),np.zeros(num_ooddataset)
    diff_best_threshold_gradient= np.zeros(num_ooddataset)
    diff_best_results= [0]*num_ooddataset#np.zeros(num_ooddataset)#[0 , 0, 0]
    diff_gradient_best_temperature = np.ones(num_ooddataset)*-1
    diff_gradient_best_magnitude = np.ones(num_ooddataset)*-1#[-1, -1, -1]
    diff_best_tnr = np.zeros(num_ooddataset)
    #计算均值与方差
    covaraiance_mean(net,args.n_classes, train_loader,output_path,args)
    sample_mean=[]
    file_mean = '%s/mean.txt'%(output_path)
    with open(file_mean, 'r') as f:
        i=0
        for line in f:
            numbers_str = line.strip().split(' ')
            numbers_list = [float(num) for num in numbers_str if num.strip()]
            sample_mean.append(numbers_list)
    covariation = []
    file_covar = '%s/covaraiance.txt'%(output_path)
    with open(file_covar, 'r') as f:
        for line in f:
            numbers_str = line.strip().split('] [')
        
            # 对每个数字字符串进行处理
            numbers_list = []
            for nums in numbers_str:
                nums = nums.strip('[').strip(']').split(',')
                nums = [float(num.strip()) for num in nums]
                numbers_list.append(nums)
            covariation=torch.tensor(numbers_list).cuda()
            # 将当前行的二维列表添加到 precision 中

    MDS_val_score(sample_mean,covariation,net,val_loader,output_path,True,M_list,T_list,args.n_classes,args)
    for item_ood, (ood_dataset,ood_name) in enumerate(out_dataset_list):
        ood_dataset = ood_dataset[1]
        ood_loader = get_coords_id_loader(ood_dataset)
        print('Out-distribution: ' + ood_name)
        MDS_val_score(sample_mean,covariation,net,ood_loader,output_path,False,M_list,T_list,args.n_classes,args,oodname=str(item_ood))
        val_results = openood_metric(output_path,str(item_ood), ['MDS_val'],True)
        best_val=0
        for i in range(len(val_results)):
            if diff_best_tnr[item_ood] < val_results[i]['AUROC']:#验证集最好的参数作为测试的参数#'TNR'
                diff_best_tnr[item_ood] = val_results[i]['AUROC']
                best_val=i

        for i in range(len(val_results)):
            print('%.3f'%(val_results[i]['AUROC']), end=' ')
        print('\n')
        if diff_gradient_best_tnr[item_ood]<diff_best_tnr[item_ood]:
            diff_gradient_best_tnr[item_ood]=diff_best_tnr[item_ood]
            diff_gradient_best_val[item_ood]=best_val
            diff_gradient_best_temperature[item_ood]= T_list[int(diff_gradient_best_val[item_ood]//(len(M_list)))]#temperature
            diff_gradient_best_magnitude[item_ood] = M_list[int((diff_gradient_best_val[item_ood]%(len(M_list))))]#magnitude
    dataset = datasets[2]
    dataset.load_from_h5(True)
    dataset.return_slideid()
    test_loader = get_coords_id_loader(dataset)
    for item_ood, (ood_dataset,ood_name) in enumerate(out_dataset_list):
        #此时需要修改loader，切换为OOD测试集
        print('Out-distribution: ' + ood_name)
        
        MDS_test_score(sample_mean,covariation,net,test_loader,output_path,True,diff_gradient_best_magnitude[item_ood],
                       diff_gradient_best_temperature[item_ood],args.n_classes,args,oodname=str(item_ood))
    
        ood_dataset = ood_dataset[2]
        ood_loader = get_coords_id_loader(ood_dataset)
        MDS_test_score(sample_mean,covariation,net,ood_loader,output_path,False,diff_gradient_best_magnitude[item_ood],
                       diff_gradient_best_temperature[item_ood],args.n_classes,args,oodname=str(item_ood))
        test_results = openood_metric(output_path,str(item_ood), ['MDS_test'],False)
        diff_best_results[item_ood]=test_results

    
    print('the total time is %.2f'%(time.time()-tt))
    # print the results
    mtypes = ['FPR','FPR80','TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    result_fpr=[]
    result_fpr80=[]
    result_auc=[]
    result_aupr_in=[]
    result_aupr_out=[]
    best_thredshld = []
    print('MDS method: in_distribution: ' + args.dataset_name +'=============='+args.exp_code)
    count_out = 0
    for ood_item, results in enumerate(diff_best_results):
        print('out_distribution: '+ (out_dataset_list[ood_item][1]))
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results[0]['FPR']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['FPR80']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['TNR']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['AUROC']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['DTACC']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['AUIN']), end='')
        print('{val:6.2f}\n'.format(val=100.*results[0]['AUOUT']), end='')
        print('temperature:' + str(diff_gradient_best_temperature[count_out]),end='')
        print('magnitude:'+ str(diff_gradient_best_magnitude[count_out]),end='')
        print('threshold_gradient:'+ str(diff_best_threshold_gradient[count_out]))
        print('')
        result_fpr.append(results[0]['FPR'])
        result_auc.append(results[0]['AUROC'])
        result_aupr_in.append(results[0]['AUIN'])
        result_aupr_out.append(results[0]['AUOUT'])
        result_fpr80.append(results[0]['FPR80'])
        threshold = '%s+%s'%(str(diff_gradient_best_magnitude[count_out]),str(diff_gradient_best_temperature[count_out]))
        best_thredshld.append(threshold)
    #if len(out_dataset_list) == 1:
    #    return best_thredshld, diff_best_results[0][0]['FPR'], diff_best_results[0][0]['AUROC'], diff_best_results[0][0]['AUIN'], diff_best_results[0][0]['AUOUT']
    return best_thredshld, result_fpr, result_auc, result_aupr_in, result_aupr_out,result_fpr80


def WSI_ODIN(datasets,ckpt_path,args,out_dataset_list):
    M_list = [0.0, 0.0014,0.0028]
    T_list = [1,10,1000]
    output_path = os.path.join(args.output_path,args.exp_code,'ODIN')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    torch.cuda.manual_seed(1)
    T_list=[1]
    tt=time.time()
    num_ooddataset=len(out_dataset_list)
    net = initiate_model(args, ckpt_path)
    dataset = datasets[1]
    dataset.load_from_h5(True)
    dataset.return_slideid()
    val_loader = get_coords_id_loader(dataset)  
    diff_gradient_best_tnr, diff_gradient_best_val=np.zeros(num_ooddataset),np.zeros(num_ooddataset)
    diff_best_threshold_gradient= np.zeros(num_ooddataset)
    diff_best_results= [0]*num_ooddataset#np.zeros(num_ooddataset)#[0 , 0, 0]
    msp_best_results= [0]*num_ooddataset#np.zeros(num_ooddataset)#[0 , 0, 0]
    diff_gradient_best_temperature = np.ones(num_ooddataset)*-1
    diff_gradient_best_magnitude = np.ones(num_ooddataset)*-1#[-1, -1, -1]
    diff_gradient_best_rotation = np.ones(num_ooddataset)*-1
    diff_best_tnr = np.zeros(num_ooddataset)
    ODIN_val_test_score(net,val_loader,output_path,True,M_list,T_list,args.n_classes,args,mode = 'val')
    for item_ood, (ood_dataset,ood_name) in enumerate(out_dataset_list):
        ood_dataset = ood_dataset[1]
        ood_loader = get_coords_id_loader(ood_dataset)
        num_oodimages = len(ood_dataset)
        print('Out-distribution: ' + ood_name)
        ODIN_val_test_score(net,ood_loader,output_path,False,M_list,T_list,args.n_classes,args,oodname=str(item_ood),mode = 'val')
        val_results = openood_metric(output_path,str(item_ood), ['ODIN_val'],True)
        best_val=0
        for i in range(len(val_results)):
            if diff_best_tnr[item_ood] < val_results[i]['AUROC']:#验证集最好的参数作为测试的参数#'TNR'
                diff_best_tnr[item_ood] = val_results[i]['AUROC']
                best_val=i
            #print('\n')
        for i in range(len(val_results)):
            print('%.3f'%(val_results[i]['AUROC']), end=' ')
        print('\n')
        if diff_gradient_best_tnr[item_ood]<diff_best_tnr[item_ood]:
            diff_gradient_best_tnr[item_ood]=diff_best_tnr[item_ood]
            diff_gradient_best_val[item_ood]=best_val
            diff_gradient_best_temperature[item_ood]= T_list[int(diff_gradient_best_val[item_ood]//(len(M_list)))]#temperature
            diff_gradient_best_magnitude[item_ood] = M_list[int((diff_gradient_best_val[item_ood]%(len(M_list))))]#magnitude
    dataset = datasets[2]
    dataset.load_from_h5(True)
    dataset.return_slideid()
    test_loader = get_coords_id_loader(dataset)
    for item_ood, (ood_dataset,ood_name) in enumerate(out_dataset_list):
        #此时需要修改loader，切换为OOD测试集
        print('Out-distribution: ' + ood_name)
        ODIN_val_test_score(net,test_loader,output_path,True,[diff_gradient_best_magnitude[item_ood]],
                            [diff_gradient_best_temperature[item_ood]],args.n_classes,args,oodname=str(item_ood),mode = 'odin_test')
        ood_dataset = ood_dataset[2]
        ood_loader = get_coords_id_loader(ood_dataset)
        ODIN_val_test_score(net,ood_loader,output_path,False,[diff_gradient_best_magnitude[item_ood]],
                            [diff_gradient_best_temperature[item_ood]],args.n_classes,args,oodname=str(item_ood),mode = 'odin_test')
        test_results = openood_metric(output_path,str(item_ood), ['ODIN_test'],False)
        diff_best_results[item_ood]=test_results
    #MSP
    for item_ood, (ood_dataset,ood_name) in enumerate(out_dataset_list):
        #此时需要修改loader，切换为OOD测试集
        print('Out-distribution: ' + ood_name,' Method: MSP')
        ODIN_val_test_score(net,test_loader,output_path,True,[0],[1],args.n_classes,args,oodname=str(item_ood),mode = 'msp_test')
        ood_dataset = ood_dataset[2]
        ood_loader = get_coords_id_loader(ood_dataset)
        ODIN_val_test_score(net,ood_loader,output_path,False,[0],[1],args.n_classes,args,oodname=str(item_ood),mode = 'msp_test')
        test_results = openood_metric(output_path,str(item_ood), ['MSP_test'],False)
        msp_best_results[item_ood]=test_results
    
    print('the total time is %.2f'%(time.time()-tt))
    # print the results
    mtypes = ['FPR','FPR80','TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    print('MSP method: in_distribution: ' + args.dataset_name +'=============='+args.exp_code)
    msp_result_fpr=[]
    msp_result_fpr80=[]
    msp_result_auc=[]
    msp_result_aupr_in=[]
    msp_result_aupr_out=[]
    for ood_item, results in enumerate(msp_best_results):
        print('out_distribution: '+ (out_dataset_list[ood_item][1]))
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results[0]['FPR']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['FPR80']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['TNR']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['AUROC']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['DTACC']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['AUIN']), end='')
        print('{val:6.2f}\n'.format(val=100.*results[0]['AUOUT']), end='')
        print('temperature:' + str(1),end='')
        print('magnitude:'+ str(0),end='')
        print('threshold_gradient:'+ str(diff_best_threshold_gradient[ood_item]))
        print('')
        msp_result_fpr.append(results[0]['FPR'])
        msp_result_fpr80.append(results[0]['FPR80'])
        msp_result_auc.append(results[0]['AUROC'])
        msp_result_aupr_in.append(results[0]['AUIN'])
        msp_result_aupr_out.append(results[0]['AUOUT'])
        
    result_fpr=[]
    result_fpr80=[]
    result_auc=[]
    result_aupr_in=[]
    result_aupr_out=[]
    best_thredshld = []
    print('ODIN method: in_distribution: ' + args.dataset_name +'=============='+args.exp_code)
    for ood_item, results in enumerate(diff_best_results):
        print('out_distribution: '+ (out_dataset_list[ood_item][1]))
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results[0]['FPR']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['FPR80']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['TNR']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['AUROC']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['DTACC']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['AUIN']), end='')
        print('{val:6.2f}\n'.format(val=100.*results[0]['AUOUT']), end='')
        print('temperature:' + str(diff_gradient_best_temperature[ood_item]),end='')
        print('magnitude:'+ str(diff_gradient_best_magnitude[ood_item]),end='')
        print('threshold_gradient:'+ str(diff_best_threshold_gradient[ood_item]))
        print('')
        result_fpr.append(results[0]['FPR'])
        result_fpr80.append(results[0]['FPR80'])
        result_auc.append(results[0]['AUROC'])
        result_aupr_in.append(results[0]['AUIN'])
        result_aupr_out.append(results[0]['AUOUT'])
        threshold = '%s+%s'%(str(diff_gradient_best_magnitude[ood_item]),str(diff_gradient_best_temperature[ood_item]))
        best_thredshld.append(threshold)
    #if len(out_dataset_list) == 1:
    #    return diff_best_results, diff_best_results[0][0]['FPR'], diff_best_results[0][0]['AUROC'], diff_best_results[0][0]['AUIN'], diff_best_results[0][0]['AUOUT'],\
    #            diff_best_results[0][0]['FPR'], diff_best_results[0][0]['AUROC'], diff_best_results[0][0]['AUIN'], diff_best_results[0][0]['AUOUT']
    return best_thredshld, result_fpr, result_auc, result_aupr_in, result_aupr_out,result_fpr80,\
            msp_result_fpr, msp_result_auc, msp_result_aupr_in, msp_result_aupr_out,msp_result_fpr80


def WSI_dropout(datasets,ckpt_path,args,out_dataset_list):
    OOD_score_way = args.ood_score_way#'mean'#'variance' # 'mean'
    print(OOD_score_way)
    T_list = [1]
    method = 'MCDropout'
    output_path = os.path.join(args.output_path,args.exp_code,method)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    torch.cuda.manual_seed(1)
    #T_list=[1]
    tt=time.time()
    num_ooddataset=len(out_dataset_list)
    net = initiate_model(args, ckpt_path)
    dataset = datasets[1]
    dataset.load_from_h5(True)
    dataset.return_slideid()
    val_loader = get_coords_id_loader(dataset)
    net.eval()   
    diff_gradient_best_tnr, diff_gradient_best_val=np.zeros(num_ooddataset),np.zeros(num_ooddataset)
    diff_best_results= [0]*num_ooddataset#np.zeros(num_ooddataset)#[0 , 0, 0]
    diff_gradient_best_temperature = np.ones(num_ooddataset)*-1
    diff_best_tnr = np.zeros(num_ooddataset)
    train_dataset = datasets[0]
    train_dataset.load_from_h5(True)
    train_dataset.return_slideid()

    
    uncertainty_score(net,val_loader,output_path,True,T_list,args.n_classes,args,mode = 'val',method = method,OOD_score_way = OOD_score_way)
    for item_ood, (ood_dataset,ood_name) in enumerate(out_dataset_list):
        ood_dataset = ood_dataset[1]
        ood_loader = get_coords_id_loader(ood_dataset)
        print('Out-distribution: ' + ood_name)
        uncertainty_score(net,ood_loader,output_path,False,T_list,args.n_classes,args,oodname=str(item_ood),mode = 'val',method = method,OOD_score_way = OOD_score_way)
        val_results = openood_metric(output_path,str(item_ood), ['MCDropout_val'],True)
        best_val=0
        for i in range(len(val_results)):
            if diff_best_tnr[item_ood] < val_results[i]['AUROC']:#验证集最好的参数作为测试的参数#'TNR'
                diff_best_tnr[item_ood] = val_results[i]['AUROC']
                best_val=i
        for i in range(len(val_results)):
            print('%.3f'%(val_results[i]['AUROC']), end=' ')
        print('\n')
        if diff_gradient_best_tnr[item_ood]<diff_best_tnr[item_ood]:
            diff_gradient_best_tnr[item_ood]=diff_best_tnr[item_ood]
            diff_gradient_best_val[item_ood]=best_val
            diff_gradient_best_temperature[item_ood]= T_list[int(best_val)]#temperature
            
    dataset = datasets[2]
    dataset.load_from_h5(True)
    dataset.return_slideid()
    test_loader = get_coords_id_loader(dataset)
    for item_ood, (ood_dataset,ood_name) in enumerate(out_dataset_list):
        #此时需要修改loader，切换为OOD测试集
        print('Out-distribution: ' + ood_name)
        uncertainty_score(net,test_loader,output_path,True,
                            [diff_gradient_best_temperature[item_ood]],args.n_classes,args,oodname=str(item_ood),mode = 'test', method = method,OOD_score_way = OOD_score_way)
        ood_dataset = ood_dataset[2]
        ood_loader = get_coords_id_loader(ood_dataset)
        uncertainty_score(net,ood_loader,output_path,False,
                            [diff_gradient_best_temperature[item_ood]],args.n_classes,args,oodname=str(item_ood),mode = 'test', method = method,OOD_score_way = OOD_score_way)
        test_results = openood_metric(output_path,str(item_ood), ['MCDropout_test'],False)
        diff_best_results[item_ood]=test_results  
    print('the total time is %.2f'%(time.time()-tt))
    # print the results
    mtypes = ['FPR','FPR80','TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    result_fpr=[]
    result_fpr80=[]
    result_auc=[]
    result_aupr_in=[]
    result_aupr_out=[]
    best_threshold = []
    print('MCDropout method: in_distribution: ' + args.dataset_name +'=============='+args.exp_code)
    for ood_item, results in enumerate(diff_best_results):
        print('out_distribution: '+ (out_dataset_list[ood_item][1]))
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results[0]['FPR']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['FPR80']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['TNR']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['AUROC']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['DTACC']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['AUIN']), end='')
        print('{val:6.2f}\n'.format(val=100.*results[0]['AUOUT']), end='')
        print('temperature:' + str(diff_gradient_best_temperature[ood_item]),end='')
        print('')
        result_fpr.append(results[0]['FPR'])
        result_fpr80.append(results[0]['FPR80'])
        result_auc.append(results[0]['AUROC'])
        result_aupr_in.append(results[0]['AUIN'])
        result_aupr_out.append(results[0]['AUOUT'])
        threshold = '%s'%(str(diff_gradient_best_temperature[ood_item]))
        best_threshold.append(threshold)
    return best_threshold, result_fpr, result_auc, result_aupr_in, result_aupr_out,result_fpr80

def WSI_ensemble(datasets,ckpt_path,args,out_dataset_list):
    OOD_score_way = args.ood_score_way#'mean'#'variance' # 'mean'
    print(OOD_score_way)
    T_list = [1,10,100,1000]
    method = 'M-heads'
    output_path = os.path.join(args.output_path,args.exp_code,method)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    torch.cuda.manual_seed(1)
    #T_list=[1]
    tt=time.time()
    num_ooddataset=len(out_dataset_list)
    net = initiate_model(args, ckpt_path)
    dataset = datasets[1]
    dataset.load_from_h5(True)
    dataset.return_slideid()
    val_loader = get_coords_id_loader(dataset)
    net.eval()   
    diff_gradient_best_tnr, diff_gradient_best_val=np.zeros(num_ooddataset),np.zeros(num_ooddataset)
    diff_best_results= [0]*num_ooddataset#np.zeros(num_ooddataset)#[0 , 0, 0]
    diff_gradient_best_temperature = np.ones(num_ooddataset)*-1
    diff_best_tnr = np.zeros(num_ooddataset)
    train_dataset = datasets[0]
    train_dataset.load_from_h5(True)
    train_dataset.return_slideid()

    
    uncertainty_score(net,val_loader,output_path,True,T_list,args.n_classes,args,mode = 'val',method = method,OOD_score_way = OOD_score_way)
    for item_ood, (ood_dataset,ood_name) in enumerate(out_dataset_list):
        ood_dataset = ood_dataset[1]
        ood_loader = get_coords_id_loader(ood_dataset)
        print('Out-distribution: ' + ood_name)
        uncertainty_score(net,ood_loader,output_path,False,T_list,args.n_classes,args,oodname=str(item_ood),mode = 'val',method = method,OOD_score_way = OOD_score_way)
        val_results = openood_metric(output_path,str(item_ood), ['M-heads_val'],True)
        best_val=0
        for i in range(len(val_results)):
            if diff_best_tnr[item_ood] < val_results[i]['AUROC']:#验证集最好的参数作为测试的参数#'TNR'
                diff_best_tnr[item_ood] = val_results[i]['AUROC']
                best_val=i
        for i in range(len(val_results)):
            print('%.3f'%(val_results[i]['AUROC']), end=' ')
        print('\n')
        if diff_gradient_best_tnr[item_ood]<diff_best_tnr[item_ood]:
            diff_gradient_best_tnr[item_ood]=diff_best_tnr[item_ood]
            diff_gradient_best_val[item_ood]=best_val
            diff_gradient_best_temperature[item_ood]= T_list[int(best_val)]#temperature
            
    dataset = datasets[2]
    dataset.load_from_h5(True)
    dataset.return_slideid()
    test_loader = get_coords_id_loader(dataset)
    for item_ood, (ood_dataset,ood_name) in enumerate(out_dataset_list):
        #此时需要修改loader，切换为OOD测试集
        print('Out-distribution: ' + ood_name)
        uncertainty_score(net,test_loader,output_path,True,
                            [diff_gradient_best_temperature[item_ood]],args.n_classes,args,oodname=str(item_ood),mode = 'test', method = method,OOD_score_way = OOD_score_way)
        ood_dataset = ood_dataset[2]
        ood_loader = get_coords_id_loader(ood_dataset)
        uncertainty_score(net,ood_loader,output_path,False,
                            [diff_gradient_best_temperature[item_ood]],args.n_classes,args,oodname=str(item_ood),mode = 'test', method = method,OOD_score_way = OOD_score_way)
        test_results = openood_metric(output_path,str(item_ood), ['M-heads_test'],False)
        diff_best_results[item_ood]=test_results  
    print('the total time is %.2f'%(time.time()-tt))
    # print the results
    mtypes = ['FPR','FPR80','TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    result_fpr=[]
    result_fpr80=[]
    result_auc=[]
    result_aupr_in=[]
    result_aupr_out=[]
    best_threshold = []
    print('M-heads method: in_distribution: ' + args.dataset_name +'=============='+args.exp_code)
    for ood_item, results in enumerate(diff_best_results):
        print('out_distribution: '+ (out_dataset_list[ood_item][1]))
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results[0]['FPR']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['FPR80']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['TNR']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['AUROC']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['DTACC']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['AUIN']), end='')
        print('{val:6.2f}\n'.format(val=100.*results[0]['AUOUT']), end='')
        print('temperature:' + str(diff_gradient_best_temperature[ood_item]),end='')
        print('')
        result_fpr.append(results[0]['FPR'])
        result_fpr80.append(results[0]['FPR80'])
        result_auc.append(results[0]['AUROC'])
        result_aupr_in.append(results[0]['AUIN'])
        result_aupr_out.append(results[0]['AUOUT'])
        threshold = '%s'%(str(diff_gradient_best_temperature[ood_item]))
        best_threshold.append(threshold)
    return best_threshold, result_fpr, result_auc, result_aupr_in, result_aupr_out,result_fpr80



def WSI_React(datasets,ckpt_path,args,out_dataset_list):
    threshold_list = [0.95]
    T_list = [1]
    output_path = os.path.join(args.output_path,args.exp_code,'React')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    torch.cuda.manual_seed(1)
    T_list=[1]
    tt=time.time()
    num_ooddataset=len(out_dataset_list)
    net = initiate_model(args, ckpt_path)
    dataset = datasets[1]
    dataset.load_from_h5(True)
    dataset.return_slideid()
    val_loader = get_coords_id_loader(dataset)
    net.eval()   
    diff_gradient_best_tnr, diff_gradient_best_val=np.zeros(num_ooddataset),np.zeros(num_ooddataset)
    diff_best_threshold_gradient= np.zeros(num_ooddataset)
    diff_best_results= [0]*num_ooddataset#np.zeros(num_ooddataset)#[0 , 0, 0]
    diff_gradient_best_temperature = np.ones(num_ooddataset)*-1
    diff_gradient_best_magnitude = np.ones(num_ooddataset)*-1#[-1, -1, -1]
    diff_best_tnr = np.zeros(num_ooddataset)
    train_dataset = datasets[0]
    train_dataset.load_from_h5(True)
    train_dataset.return_slideid()
    train_loader = get_coords_id_loader(train_dataset)


    threshold_path = os.path.join(args.output_path,'React',args.experiment_mask)
    if not os.path.exists(threshold_path):
        os.makedirs(threshold_path)
    thresh_list = React_threshold(threshold_list,train_loader,net,args.model_type,threshold_path)
    React_val_test_score(net,val_loader,output_path,True,thresh_list,T_list,args.n_classes,args,mode = 'val')
    for item_ood, (ood_dataset,ood_name) in enumerate(out_dataset_list):
        ood_dataset = ood_dataset[1]
        ood_loader = get_coords_id_loader(ood_dataset)
        print('Out-distribution: ' + ood_name)
        React_val_test_score(net,ood_loader,output_path,False,thresh_list,T_list,args.n_classes,args,oodname=str(item_ood),mode = 'val')
        val_results = openood_metric(output_path,str(item_ood), ['React_val'],True)
        best_val=0
        for i in range(len(val_results)):
            if diff_best_tnr[item_ood] < val_results[i]['AUROC']:#验证集最好的参数作为测试的参数#'TNR'
                diff_best_tnr[item_ood] = val_results[i]['AUROC']
                best_val=i
        for i in range(len(val_results)):
            print('%.3f'%(val_results[i]['AUROC']), end=' ')
        print('\n')
        if diff_gradient_best_tnr[item_ood]<diff_best_tnr[item_ood]:
            diff_gradient_best_tnr[item_ood]=diff_best_tnr[item_ood]
            diff_gradient_best_val[item_ood]=best_val
            diff_gradient_best_temperature[item_ood]= T_list[int(diff_gradient_best_val[item_ood]//(len(thresh_list)))]#temperature
            diff_gradient_best_magnitude[item_ood] = thresh_list[int((diff_gradient_best_val[item_ood]%(len(thresh_list))))]#magnitude
    dataset = datasets[2]
    dataset.load_from_h5(True)
    dataset.return_slideid()
    test_loader = get_coords_id_loader(dataset)
    for item_ood, (ood_dataset,ood_name) in enumerate(out_dataset_list):
        #此时需要修改loader，切换为OOD测试集
        print('Out-distribution: ' + ood_name)
        React_val_test_score(net,test_loader,output_path,True,[diff_gradient_best_magnitude[item_ood]],
                            [diff_gradient_best_temperature[item_ood]],args.n_classes,args,oodname=str(item_ood),mode = 'test')
        ood_dataset = ood_dataset[2]
        ood_loader = get_coords_id_loader(ood_dataset)
        React_val_test_score(net,ood_loader,output_path,False,[diff_gradient_best_magnitude[item_ood]],
                            [diff_gradient_best_temperature[item_ood]],args.n_classes,args,oodname=str(item_ood),mode = 'test')
        test_results = openood_metric(output_path,str(item_ood), ['React_test'],False)
        diff_best_results[item_ood]=test_results  
    print('the total time is %.2f'%(time.time()-tt))
    # print the results
    mtypes = ['FPR','FPR80','TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    result_fpr=[]
    result_fpr80=[]
    result_auc=[]
    result_aupr_in=[]
    result_aupr_out=[]
    best_threshold = []
    print('React method: in_distribution: ' + args.dataset_name +'=============='+args.exp_code)
    for ood_item, results in enumerate(diff_best_results):
        print('out_distribution: '+ (out_dataset_list[ood_item][1]))
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results[0]['FPR']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['FPR80']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['TNR']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['AUROC']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['DTACC']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['AUIN']), end='')
        print('{val:6.2f}\n'.format(val=100.*results[0]['AUOUT']), end='')
        print('temperature:' + str(diff_gradient_best_temperature[ood_item]),end='')
        print('magnitude:'+ str(diff_gradient_best_magnitude[ood_item]),end='')
        print('threshold_gradient:'+ str(diff_best_threshold_gradient[ood_item]))
        print('')
        result_fpr.append(results[0]['FPR'])
        result_fpr80.append(results[0]['FPR80'])
        result_auc.append(results[0]['AUROC'])
        result_aupr_in.append(results[0]['AUIN'])
        result_aupr_out.append(results[0]['AUOUT'])
        threshold = '%s+%s'%(str(diff_gradient_best_magnitude[ood_item]),str(diff_gradient_best_temperature[ood_item]))
        best_threshold.append(threshold)
    return best_threshold, result_fpr, result_auc, result_aupr_in, result_aupr_out,result_fpr80

def WSI_EBO(datasets,ckpt_path,args,out_dataset_list):
    thresh_list = [0]
    T_list=[1]
    T_list = [1,10,1000]
    output_path = os.path.join(args.output_path,args.exp_code,'EBO')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    torch.cuda.manual_seed(1)
    
    tt=time.time()
    num_ooddataset=len(out_dataset_list)
    net = initiate_model(args, ckpt_path)
    dataset = datasets[1]
    dataset.load_from_h5(True)
    dataset.return_slideid()
    val_loader = get_coords_id_loader(dataset)
    net.eval()   
    diff_gradient_best_tnr, diff_gradient_best_val=np.zeros(num_ooddataset),np.zeros(num_ooddataset)
    diff_best_threshold_gradient= np.zeros(num_ooddataset)
    diff_best_results= [0]*num_ooddataset#np.zeros(num_ooddataset)#[0 , 0, 0]
    diff_gradient_best_temperature = np.ones(num_ooddataset)*-1
    diff_gradient_best_magnitude = np.ones(num_ooddataset)*-1#[-1, -1, -1]
    diff_best_tnr = np.zeros(num_ooddataset)
    EBO_val_test_score(net,val_loader,output_path,True,thresh_list,T_list,args.n_classes,args,mode = 'val')
    for item_ood, (ood_dataset,ood_name) in enumerate(out_dataset_list):
        ood_dataset = ood_dataset[1]
        ood_loader = get_coords_id_loader(ood_dataset)
        print('Out-distribution: ' + ood_name)
        EBO_val_test_score(net,ood_loader,output_path,False,thresh_list,T_list,args.n_classes,args,oodname=str(item_ood),mode = 'val')
        val_results = openood_metric(output_path,str(item_ood), ['EBO_val'],True)
        best_val=0
        for i in range(len(val_results)):
            if diff_best_tnr[item_ood] < val_results[i]['AUROC']:#验证集最好的参数作为测试的参数#'TNR'
                diff_best_tnr[item_ood] = val_results[i]['AUROC']
                best_val=i
        for i in range(len(val_results)):
            print('%.3f'%(val_results[i]['AUROC']), end=' ')
        print('\n')
        if diff_gradient_best_tnr[item_ood]<diff_best_tnr[item_ood]:
            diff_gradient_best_tnr[item_ood]=diff_best_tnr[item_ood]
            diff_gradient_best_val[item_ood]=best_val
            diff_gradient_best_temperature[item_ood]= T_list[int(diff_gradient_best_val[item_ood]//(len(thresh_list)))]#temperature
            diff_gradient_best_magnitude[item_ood] = thresh_list[int((diff_gradient_best_val[item_ood]%(len(thresh_list))))]#magnitude
    dataset = datasets[2]
    dataset.load_from_h5(True)
    dataset.return_slideid()
    test_loader = get_coords_id_loader(dataset)
    for item_ood, (ood_dataset,ood_name) in enumerate(out_dataset_list):
        #此时需要修改loader，切换为OOD测试集
        print('Out-distribution: ' + ood_name)
        EBO_val_test_score(net,test_loader,output_path,True,[diff_gradient_best_magnitude[item_ood]],
                            [diff_gradient_best_temperature[item_ood]],args.n_classes,args,oodname=str(item_ood),mode = 'test')
        ood_dataset = ood_dataset[2]
        ood_loader = get_coords_id_loader(ood_dataset)
        EBO_val_test_score(net,ood_loader,output_path,False,[diff_gradient_best_magnitude[item_ood]],
                            [diff_gradient_best_temperature[item_ood]],args.n_classes,args,oodname=str(item_ood),mode = 'test')
        test_results = openood_metric(output_path,str(item_ood), ['EBO_test'],False)
        diff_best_results[item_ood]=test_results  
    print('the total time is %.2f'%(time.time()-tt))
    # print the results
    mtypes = ['FPR','FPR80','TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    result_fpr=[]
    result_fpr80=[]
    result_auc=[]
    result_aupr_in=[]
    result_aupr_out=[]
    best_threshold = []
    print('EBO method: in_distribution: ' + args.dataset_name +'=============='+args.exp_code)
    for ood_item, results in enumerate(diff_best_results):
        print('out_distribution: '+ (out_dataset_list[ood_item][1]))
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results[0]['FPR']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['FPR80']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['TNR']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['AUROC']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['DTACC']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['AUIN']), end='')
        print('{val:6.2f}\n'.format(val=100.*results[0]['AUOUT']), end='')
        print('temperature:' + str(diff_gradient_best_temperature[ood_item]),end='')
        print('magnitude:'+ str(diff_gradient_best_magnitude[ood_item]),end='')
        print('threshold_gradient:'+ str(diff_best_threshold_gradient[ood_item]))
        print('')
        result_fpr.append(results[0]['FPR'])
        result_fpr80.append(results[0]['FPR80'])
        result_auc.append(results[0]['AUROC'])
        result_aupr_in.append(results[0]['AUIN'])
        result_aupr_out.append(results[0]['AUOUT'])
        threshold = '%s+%s'%(str(diff_gradient_best_magnitude[ood_item]),str(diff_gradient_best_temperature[ood_item]))
        best_threshold.append(threshold)

    #if len(out_dataset_list) == 1:
    #    return diff_best_results, diff_best_results[0][0]['FPR'], diff_best_results[0][0]['AUROC'], diff_best_results[0][0]['AUIN'], diff_best_results[0][0]['AUOUT'],\
    #            diff_best_results[0][0]['FPR'], diff_best_results[0][0]['AUROC'], diff_best_results[0][0]['AUIN'], diff_best_results[0][0]['AUOUT']
    return best_threshold, result_fpr, result_auc, result_aupr_in, result_aupr_out,result_fpr80

def WSI_KNN(datasets,ckpt_path,args,out_dataset_list):
    output_path = os.path.join(args.output_path,args.exp_code,'KNN')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    torch.cuda.manual_seed(1)
    T_list=[1]
    tt=time.time()
    num_ooddataset=len(out_dataset_list)
    shuffle= False#一定得False，否则会影响probability_file
    net = initiate_model(args, ckpt_path)
    train_dataset = datasets[0]
    train_dataset.load_from_h5(True)
    train_dataset.return_slideid()
    train_loader = get_coords_id_loader(train_dataset)

    dataset = datasets[1]
    dataset.load_from_h5(True)
    dataset.return_slideid()
    val_loader = get_coords_id_loader(dataset)
    net.eval()   
    diff_gradient_best_tnr, diff_gradient_best_val=np.zeros(num_ooddataset),np.zeros(num_ooddataset)
    diff_best_threshold_gradient= np.zeros(num_ooddataset)
    diff_best_results= [0]*num_ooddataset#np.zeros(num_ooddataset)#[0 , 0, 0]
    diff_gradient_best_temperature = np.ones(num_ooddataset)*-1
    diff_gradient_best_magnitude = np.ones(num_ooddataset)*-1#[-1, -1, -1]
    diff_best_tnr = np.zeros(num_ooddataset)
    #获取训练数据的标准特征
    dis_index = normal_fea(net,args.n_classes, train_loader,output_path,args)
    if args.model_name == 'resnet50':
        M_list = [4,5,50,100,10]#[20]
    elif args.model_name == 'conch_v1' and args.model_type == 'transmil':
        M_list = [int(len(train_loader)-2)]
    else:
        M_list = [50,100,200,4,5,10]
    KNN_val_test_score(dis_index,net,val_loader,output_path,True,M_list,T_list,args.n_classes,args, mode='val')
    for item_ood, (ood_dataset,ood_name) in enumerate(out_dataset_list):
        ood_dataset = ood_dataset[1]
        ood_loader = get_coords_id_loader(ood_dataset)
        print('Out-distribution: ' + ood_name)
        KNN_val_test_score(dis_index,net,ood_loader,output_path,False,M_list,T_list,args.n_classes,args,oodname=str(item_ood),mode='val')
        val_results = openood_metric(output_path,str(item_ood), ['KNN_val'],True)
        best_val=0
        for i in range(len(val_results)):
            if diff_best_tnr[item_ood] < val_results[i]['AUROC']:#验证集最好的参数作为测试的参数#'TNR'
                diff_best_tnr[item_ood] = val_results[i]['AUROC']
                best_val=i

        for i in range(len(val_results)):
            print('%.3f'%(val_results[i]['AUROC']), end=' ')
        print('\n')
        if diff_gradient_best_tnr[item_ood]<diff_best_tnr[item_ood]:
            diff_gradient_best_tnr[item_ood]=diff_best_tnr[item_ood]
            diff_gradient_best_val[item_ood]=best_val
            diff_gradient_best_temperature[item_ood]= T_list[int(diff_gradient_best_val[item_ood]//(len(M_list)))]#temperature
            diff_gradient_best_magnitude[item_ood] = M_list[int((diff_gradient_best_val[item_ood]%(len(M_list))))]#magnitude
    dataset = datasets[2]
    dataset.load_from_h5(True)
    dataset.return_slideid()
    test_loader = get_coords_id_loader(dataset)
    for item_ood, (ood_dataset,ood_name) in enumerate(out_dataset_list):
        #此时需要修改loader，切换为OOD测试集
        print('Out-distribution: ' + ood_name)
        
        KNN_val_test_score(dis_index,net,test_loader,output_path,True,
                       [diff_gradient_best_magnitude[item_ood]],[diff_gradient_best_temperature[item_ood]],args.n_classes,args,oodname=str(item_ood), mode='test')
    
        ood_dataset = ood_dataset[2]
        ood_loader = get_coords_id_loader(ood_dataset)
        KNN_val_test_score(dis_index,net,ood_loader,output_path,False,[diff_gradient_best_magnitude[item_ood]],
                       [diff_gradient_best_temperature[item_ood]],args.n_classes,args,oodname=str(item_ood), mode='test')
        test_results = openood_metric(output_path,str(item_ood), ['KNN_test'],False)
        diff_best_results[item_ood]=test_results

    
    print('the total time is %.2f'%(time.time()-tt))
    # print the results
    mtypes = ['FPR','FPR80','TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    result_fpr=[]
    result_fpr80 = []
    result_auc=[]
    result_aupr_in=[]
    result_aupr_out=[]
    best_thredshld = []
    print('KNN method: in_distribution: ' + args.dataset_name +'=============='+args.exp_code)
    count_out = 0
    for ood_item, results in enumerate(diff_best_results):
        print('out_distribution: '+ (out_dataset_list[ood_item][1]))
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results[0]['FPR']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['FPR80']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['TNR']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['AUROC']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['DTACC']), end='')
        print('{val:6.2f}'.format(val=100.*results[0]['AUIN']), end='')
        print('{val:6.2f}\n'.format(val=100.*results[0]['AUOUT']), end='')
        print('temperature:' + str(diff_gradient_best_temperature[count_out]),end='')
        print('magnitude:'+ str(diff_gradient_best_magnitude[count_out]),end='')
        print('threshold_gradient:'+ str(diff_best_threshold_gradient[count_out]))
        print('')
        result_fpr.append(results[0]['FPR'])
        result_fpr80.append(results[0]['FPR80'])
        result_auc.append(results[0]['AUROC'])
        result_aupr_in.append(results[0]['AUIN'])
        result_aupr_out.append(results[0]['AUOUT'])
        threshold = '%s+%s'%(str(diff_gradient_best_magnitude[count_out]),str(diff_gradient_best_temperature[count_out]))
        best_thredshld.append(threshold)
        count_out+=1
    #if len(out_dataset_list) == 1:
    #    return best_thredshld, diff_best_results[0][0]['FPR'], diff_best_results[0][0]['AUROC'], diff_best_results[0][0]['AUIN'], diff_best_results[0][0]['AUOUT']
    return best_thredshld, result_fpr, result_auc, result_aupr_in, result_aupr_out,result_fpr80



parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples about pathology image')
#需要修改的
parser.add_argument('--model_name', type=str, default='resnet50', choices=['resnet50_trunc', 'uni_v1', 'conch_v1','resnet50'])
parser.add_argument('--model_type', type=str, 
                    choices=['clam_sb', 'transmil', 'abmil','clam_dropout', 'transmil_dropout', 'abmil_dropout','clam_ensemble', 'transmil_ensemble', 'abmil_ensemble'], 
                    default='abmil_dropout', help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--experiment_mask', type = str,default='C16',choices=['RCC','C16','RCC_conch','C16_conch'],
					help='name about experiment')
parser.add_argument('--task', type=str, default='task_1_tumor_vs_normal', choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'])
parser.add_argument('--gpu', type=int, default=0 )
parser.add_argument('--ood_score_way', type=str, default='variance', choices=['variance',  'mean'])
#ood_score_way#'mean'#'variance



#默认的\
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--fea_csv_path', type = str,default='./dataset_csv',
					help='name about experiment')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')#default='task_2_RCC_CLAM_50'
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--embed_dim', type=str, default=1024)
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--model_results_dir', type=str, default='./results/train_wsi_model/',
                    help='model_path in this py, i.e. '+
                    'results_dir+ models_exp_code')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--result_path', type=str, default='./ood_results/Result/',
                    help='result directory')
parser.add_argument('--target_patch_size', type=int, default=224)
parser.add_argument('--drop_out', type=float, default=0.)
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--splits_dir', type=str, default='./splits/',
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--data_root_dir', type=str, default='./data/',
                    help='data directory')#在resnet下的特征
parser.add_argument('--output_path', type=str, default='./ood_results/Probability/',
                    help='data directory')
parser.add_argument('--data_h5_dir', type=str, default='./data/clam_format_patch/')#存放原始特征的地址
parser.add_argument('--csv_path', type=str, default='./data/clam_format_patch/')
parser.add_argument('--ood_csv_path', type=str)
parser.add_argument('--seed', type=int, default=1)
parser.set_defaults(argument=True)
args=parser.parse_args()
args.id_csv_path = os.path.join(args.csv_path,args.experiment_mask,'process_list_autogen.csv')
args.splits_dir = os.path.join(args.splits_dir, args.task+'_{}_{}'.format(args.experiment_mask,(args.label_frac*100))) 
args.id_data_h5_dir = os.path.join(args.data_h5_dir, args.experiment_mask) 
args.dataset_name=args.experiment_mask
args.exp_code='%s_%s_%s'%(args.experiment_mask, args.model_type,args.model_name)
args.models_dir = os.path.join(args.model_results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
args.data_root_dir =os.path.join(args.data_root_dir,'clam_format_feature_%s'%(args.model_name))
args.csv_path_id = os.path.join(args.fea_csv_path,'%s_dummy_clean_%s.csv'%(args.task,args.experiment_mask))
if args.model_name == 'conch_v1':
    args.embed_dim =512
out_dataset_list=[]
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    if args.model_name == 'resnet50':
        local_path = './data/clam_format_feature_resnet50/'
    else:
        local_path = './data/clam_format_feature_conch_v1/'
    dataset = Generic_MIL_Dataset(csv_path = args.csv_path_id,
                            data_dir= os.path.join(local_path, args.experiment_mask),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1},
                            patient_strat=False,
                            data_format = '.tif',
                            ignore=[])
    ood_name_list=['TCGA_breast_ood','TCGA_lung_ood']
    if args.model_name =='conch_v1':
        ood_name_list = ['TCGA_breast_ood_conch','TCGA_lung_ood_conch']
    for ood_name in ood_name_list:  
        args.csv_path_ood = os.path.join(args.fea_csv_path,'%s_dummy_clean_%s.csv'%(args.task,ood_name))
        ood_dataset = Generic_MIL_Dataset(csv_path=args.csv_path_ood,
                                  data_dir=os.path.join(args.data_root_dir, ood_name),
                                  shuffle=False, 
                                  print_info=True,
                                  label_dict={-1: -1},
                                  patient_strat=False,
                                  ignore=[])
        
        if '_conch' in ood_name:
            ood_name_base = ood_name.replace('_conch', '')
        else:
            ood_name_base = ood_name
        ood_csv_path = '{}/ood_detection_splits_{}.csv'.format(args.splits_dir,ood_name_base)
        ood_datasets = ood_dataset.return_splits(from_id=False, csv_path=ood_csv_path)
        for i in range(1,3):
            ood_datasets[i].load_from_h5(True)
            ood_datasets[i].return_slideid()
        out_dataset_list.append((ood_datasets,ood_name))
    '''
    ood_name_list=['RCC']
    for ood_name in ood_name_list:  
        args.csv_path_ood = os.path.join(args.fea_csv_path,'%s_dummy_clean_%s.csv'%('task_2_tumor_subtyping',ood_name))
        ood_dataset = Generic_MIL_Dataset(csv_path=args.csv_path_ood,
                                  data_dir=os.path.join(args.data_root_dir, ood_name),
                                  shuffle=False, 
                                  print_info=True,
                                  label_dict={'subtype_1':-1, 'subtype_2':-1, 'subtype_3':-1},
                                  patient_strat=False,
                                  ignore=[])
        
        if '_conch' in ood_name:
            ood_name_base = ood_name.replace('_conch', '')
        else:
            ood_name_base = ood_name
        ood_csv_path = '{}/ood_detection_splits_{}.csv'.format(args.splits_dir,ood_name_base)
        ood_datasets = ood_dataset.return_splits(from_id=False, csv_path=ood_csv_path)
        for i in range(1,3):
            ood_datasets[i].load_from_h5(True)
            ood_datasets[i].return_slideid()
        out_dataset_list.append((ood_datasets,ood_name))'''
elif args.task == 'task_2_tumor_subtyping':
    ood_name_list=['RCC_ood','RCC_Normal']
    #ood_name_list=['RCC_ood']
    ood_name = 'RCC_ood'
    ood_name_base = 'RCC_ood'
    if args.model_name =='conch_v1':
        ood_name = 'RCC_ood_conch'
        
    args.n_classes=3
    if args.model_name == 'resnet50':
        local_path = './data/clam_format_feature_resnet50/'
    else:
        local_path = './data/clam_format_feature_conch_v1/'
    dataset = Generic_MIL_Dataset(csv_path=args.csv_path_id,
                            data_dir= os.path.join(local_path, args.experiment_mask),
                            shuffle = False, 
                            print_info = False,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])
    args.csv_path_ood = os.path.join(args.fea_csv_path,'%s_dummy_clean_%s.csv'%(args.task,ood_name))
    rcc_ood_dataset = Generic_MIL_Dataset(csv_path=args.csv_path_ood,
                                  data_dir=os.path.join(args.data_root_dir, ood_name),
                                  shuffle=False, 
                                  print_info=True,
                                  label_dict={'subtype_1': 0, 'subtype_2': 1, 'subtype_3': 2},
                                  patient_strat=False,
                                  ignore=[])
    
    ood_csv_path = '{}/ood_detection_splits_{}.csv'.format(args.splits_dir,ood_name_base)
    ood_dataset = rcc_ood_dataset.return_splits(from_id=False, csv_path=ood_csv_path)
    for i in range(1,3):
        ood_dataset[i].load_from_h5(True)
        ood_dataset[i].return_slideid()
    out_dataset_list.append((ood_dataset, ood_name))

    
    #1.normal_ffpe
    other_class_dataset = Generic_MIL_Dataset(csv_path=args.csv_path_id,
                              data_dir=os.path.join(args.data_root_dir, args.experiment_mask),
                              shuffle=False, 
                              print_info=True,
                              label_dict={'0': 4},
                              patient_strat=False,
                              ignore=[])
    ood_csv_path = '{}/ood_detection_splits_RCC_normal_ffpe.csv'.format(args.splits_dir)
    
    ood_dataset1 = other_class_dataset.return_splits(from_id=False, csv_path=ood_csv_path)
    #2. normal_forzen
    
    other_class_ood_dataset = Generic_MIL_Dataset(csv_path=args.csv_path_ood,
                                              data_dir=os.path.join(args.data_root_dir, ood_name),
                                              shuffle=False, 
                                              print_info=True,
                                              label_dict={'3': 3, '0': 4},
                                              patient_strat=False,
                                              ignore=[])
    ood_csv_path = '{}/ood_detection_splits_RCC_normal_frozen.csv'.format(args.splits_dir)
    
    ood_dataset2 = other_class_ood_dataset.return_splits(from_id=False, csv_path=ood_csv_path)
    train_ood_dataset=None
    for i in range(1,3):
        ood_dataset1[i].load_from_h5(True)
        ood_dataset1[i].return_slideid()
        ood_dataset2[i].load_from_h5(True)
        ood_dataset2[i].return_slideid()

    val_ood_dataset=ConcatDataset([ood_dataset1[1], ood_dataset2[1]])
    test_ood_dataset=ConcatDataset([ood_dataset1[2], ood_dataset2[2]])
    out_dataset_list.append(([train_ood_dataset,val_ood_dataset,test_ood_dataset], 'RCC_Normal'))



    
    
    
else:
    raise NotImplementedError
# args.k=2
if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)

ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}




   #  nohup python -u WSI_OOD_baseline.py --gpu=0 --experiment_mask=C16 --model_name=resnet50 --model_type=clam_ensemble --task=task_1_tumor_vs_normal > first_review_log/ood_results/ood_C16_resnet_clam_ensemble.log 2>&1 &
if __name__=='__main__':
    torch.cuda.set_device(args.gpu)
    print(args.experiment_mask,args.model_name,args.model_type)
    ood_task=['ODIN','React','EBO','KNN']
    if "ensemble" in args.model_type:
        ood_task=['M-head']
    elif 'dropout' in args.model_type:
        ood_task=['MCDropout']
    result={}
    for ood_method in ood_task:
        if ood_method =='ODIN':
            result['MSP']={}
            result['MSP']['all_auc']=[]
            result['MSP']['all_acc']=[]
            result['MSP']['all_fpr']=[]
            result['MSP']['all_fpr80']=[]
            result['MSP']['all_tpr']=[]
            result['MSP']['all_aupr_in']=[]
            result['MSP']['all_aupr_out']=[]
            result['MSP']['threshold']=[]
        result[ood_method]={}
        result[ood_method]['all_auc']=[]
        result[ood_method]['all_acc']=[]
        result[ood_method]['all_fpr']=[]
        result[ood_method]['all_fpr80']=[]
        result[ood_method]['all_tpr']=[]
        result[ood_method]['all_aupr_in']=[]
        result[ood_method]['all_aupr_out']=[]
        result[ood_method]['threshold']=[]
    result_path=os.path.join(args.result_path,args.exp_code)
    os.makedirs(result_path, exist_ok=True)
    for ckpt_idx in range(len(ckpt_paths)):
        print('The %d-fold'%(ckpt_idx))
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/ood_detection_splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])           
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)#0:val 1:test 2:train(null)
            #split_dataset = datasets[datasets_id[args.split]]
        for ood_method in ood_task:
            if ood_method =='MDS':
                threshold, fpr, auc, aupr_in, aupr_out, fpr80 = WSI_Mahalanobis(datasets,ckpt_paths[ckpt_idx],args,out_dataset_list)
            elif ood_method =='ODIN':
                msp_tpr=[]
                threshold, fpr, auc, aupr_in, aupr_out, fpr80,msp_fpr, msp_auc, msp_aupr_in, msp_aupr_out,msp_fpr80 = WSI_ODIN(datasets,ckpt_paths[ckpt_idx],args,out_dataset_list) 
                for i in range(len(out_dataset_list)):
                    msp_tpr.append(1-msp_fpr[i])
                result['MSP']['all_fpr'].append(msp_fpr)
                result['MSP']['all_fpr80'].append(msp_fpr80)
                result['MSP']['all_tpr'].append(msp_tpr)
                result['MSP']['all_auc'].append(msp_auc)
                result['MSP']['all_aupr_in'].append(msp_aupr_in)  
                result['MSP']['all_aupr_out'].append(msp_aupr_out) 
                result['MSP']['threshold'].append('1')   
            elif ood_method =='React':
                threshold, fpr, auc, aupr_in, aupr_out, fpr80 = WSI_React(datasets,ckpt_paths[ckpt_idx],args,out_dataset_list) 
            elif ood_method == 'EBO':
                threshold, fpr, auc, aupr_in, aupr_out, fpr80 = WSI_EBO(datasets,ckpt_paths[ckpt_idx],args,out_dataset_list) 
            elif ood_method == 'KNN':
                threshold, fpr, auc, aupr_in, aupr_out, fpr80 = WSI_KNN(datasets,ckpt_paths[ckpt_idx],args,out_dataset_list) 
            elif ood_method == 'M-head':
                threshold, fpr, auc, aupr_in, aupr_out, fpr80 = WSI_ensemble(datasets,ckpt_paths[ckpt_idx],args,out_dataset_list) 
            elif ood_method == 'MCDropout':
                threshold, fpr, auc, aupr_in, aupr_out, fpr80 = WSI_dropout(datasets,ckpt_paths[ckpt_idx],args,out_dataset_list) 

            else:
                raise NotImplementedError
            tpr = []
            for i in range(len(out_dataset_list)):
                tpr.append(1-fpr[i])
            result[ood_method]['all_fpr'].append(fpr)
            result[ood_method]['all_fpr80'].append(fpr80)
            result[ood_method]['all_tpr'].append(tpr)
            result[ood_method]['all_auc'].append(auc)
            result[ood_method]['all_aupr_in'].append(aupr_in)  
            result[ood_method]['all_aupr_out'].append(aupr_out)    
            result[ood_method]['threshold'].append(threshold)    
        #OOD数据的结果  
    for ood_method in ood_task:
        final_df_dict = {'folds': folds}
        for i, (ood_dataset,ood_name) in enumerate(out_dataset_list):
            final_df_dict['ood_fpr_{}'.format(ood_name)] = [fpr[i] for fpr in result[ood_method]['all_fpr']]
            final_df_dict['ood_fpr80_{}'.format(ood_name)] = [fpr[i] for fpr in result[ood_method]['all_fpr80']]
            final_df_dict['ood_tpr_{}'.format(ood_name)] = [tpr[i] for tpr in result[ood_method]['all_tpr']]
            final_df_dict['all_auc_{}'.format(ood_name)] = [auc[i] for auc in result[ood_method]['all_auc']]
            final_df_dict['ood_aupr_in_{}'.format(ood_name)] = [aupr[i] for aupr in result[ood_method]['all_aupr_in']]
            final_df_dict['ood_aupr_out_{}'.format(ood_name)] = [aupr[i] for aupr in result[ood_method]['all_aupr_out']]
            final_df_dict['threshold_{}'.format(ood_name)] = [threshold[i] for threshold in result[ood_method]['threshold']]
        final_df = pd.DataFrame(final_df_dict)
        #average_row = final_df.mean()
        #final_df.loc['Average'] = average_row
        save_name = '{}_ood_detection_result_{}_{}.csv'.format(ood_method,folds[0], folds[-1])
        final_df.to_csv(os.path.join(result_path, save_name))
        if ood_method == 'ODIN':
            final_df_dict = {'folds': folds}
            for i, (ood_dataset,ood_name) in enumerate(out_dataset_list):
                final_df_dict['ood_fpr_{}'.format(ood_name)] = [fpr[i] for fpr in result['MSP']['all_fpr']]
                final_df_dict['ood_fpr80_{}'.format(ood_name)] = [fpr[i] for fpr in result['MSP']['all_fpr80']]
                final_df_dict['ood_tpr_{}'.format(ood_name)] = [tpr[i] for tpr in result['MSP']['all_tpr']]
                final_df_dict['all_auc_{}'.format(ood_name)] = [auc[i] for auc in result['MSP']['all_auc']]
                final_df_dict['ood_aupr_in_{}'.format(ood_name)] = [aupr[i] for aupr in result['MSP']['all_aupr_in']]
                final_df_dict['ood_aupr_out_{}'.format(ood_name)] = [aupr[i] for aupr in result['MSP']['all_aupr_out']]
            final_df = pd.DataFrame(final_df_dict)
            #average_row = final_df.mean()
            #final_df.loc['Average'] = average_row
            save_name = '{}_ood_detection_result_{}_{}.csv'.format('MSP',folds[0], folds[-1])
            final_df.to_csv(os.path.join(result_path, save_name))
            
    print('end')

