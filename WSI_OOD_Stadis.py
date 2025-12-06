

#两种方法：使用统一的magnitude，针对不同的样本使用不同的magnitude
import argparse
import torch
from torch import nn
import numpy as np
import time
import os
from torch.utils.data import  ConcatDataset
from ood_utils.utils import path_Dataset,softmax_file,oodtest_loader,dataset_breast,dataset_lung,paths_labelsDataset,oodTCGA_Gastric,dataset_rare_gastric,dataset_breast_100,dataset_imagenet,imagenetdataset
from ood_utils.utils_Stadis import openood_metric
from ood_utils.utils_Stadis import diff_val_score, diff_test_score,get_oodlabel, diff_test_metric
from ood_utils.utils_WSI_Stadis import Stadis_val_score,Stadis_test_score
from ood_utils.ood_utils import metric
from ood_utils.ood_utils import ctrans_feature_list
from models import resnet_ood as resnet
from models import vit,IBOTvit
import sys
from utils.eval_utils import initiate_model,get_coords_id_loader
from models import get_encoder
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import pandas as pd

def diff_feature_ood(datasets,ckpt_path,args,out_dataset_list,just_val):
    diff_type='feature'#'softmax'#'softmax'#'logit'
    agg_type = 'agg_noise'#'agg_noise'#'noise_agg'#'agg_noise'#'noise_agg'#'noise_agg'
    print(agg_type,args.model_type,args.experiment_mask)
    magnitude_type='unified'
    rotation_list =[0]
    output_path = os.path.join(args.output_path,args.exp_code,'diff')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    T_list=[1]

    if args.task =='task_1_tumor_vs_normal':
        M_list=[0.001,0.005,0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
        M_list = [x * 100 for x in M_list]
    elif args.task =='task_2_tumor_subtyping':
        M_list = [0.5,0.8,1,1.2,1.3,1.4,1.45,1.5,1.55,1.6,1.7,1.8,1.9,2,2.5,3,3.5,4,5,6,7]
    print('M_list:')
    for m in M_list:
        print(m, end= ' ')
    print('\n rotation_list:')
    for m in rotation_list:
        print(m, end= ' ')
    tt=time.time()
    num_ooddataset=len(out_dataset_list)
    shuffle= False#一定得False，否则会影响probability_file
    net = initiate_model(args, ckpt_path)
    dataset = datasets[1]
    dataset.load_from_h5(True)
    dataset.return_slideid()

    #test_loader = get_simple_loader(dataset)
    test_loader = get_coords_id_loader(dataset)
    net.eval()
    perturbation_feature_path=os.path.join(args.data_root_dir,'%s_perturbation'%(args.dataset_name))
    diff_gradient_best_tnr, diff_gradient_best_val=np.zeros(num_ooddataset),np.zeros(num_ooddataset)
    diff_gradient_best_au = np.zeros(num_ooddataset)
    # diff_best_threshold_gradient= np.zeros(num_ooddataset)
    diff_best_results= [0]*num_ooddataset#np.zeros(num_ooddataset)#[0 , 0, 0]
    diff_gradient_best_temperature = np.ones(num_ooddataset)*-1
    diff_gradient_best_magnitude = np.ones(num_ooddataset)*-1#[-1, -1, -1]
    diff_gradient_best_rotation = np.ones(num_ooddataset)*-1
    diff_best_tnr = np.zeros(num_ooddataset)
    diff_best_au = np.zeros(num_ooddataset)
    Stadis_val_score(net,test_loader,perturbation_feature_path,output_path,True,M_list,T_list,rotation_list,diff_type,agg_type,args)
    for item_ood, (ood_dataset,ood_name) in enumerate(out_dataset_list):
        args.ood_dataset_name=ood_name
        ood_dataset = ood_dataset[1]
        ood_loader = get_coords_id_loader(ood_dataset)
        num_oodimages = len(ood_dataset)
        print('Out-distribution: ' + ood_name)
        ood_perturbation_feature_path=os.path.join(args.data_root_dir,'%s_perturbation'%(ood_name))
        Stadis_val_score(net,ood_loader,ood_perturbation_feature_path,output_path,False,M_list,T_list,rotation_list,diff_type,agg_type,args,oodname=str(item_ood))
        Stadis_path=output_path
        val_results = openood_metric(Stadis_path,str(item_ood), ['diff_val'],True)
        best_val=0
        for i in range(len(val_results)):
            #print('%.3f'%(val_results[i]['TNR']), end=' ')
            if diff_best_tnr[item_ood] < val_results[i]['AUROC']:#验证集最好的参数作为测试的参数#'TNR'
                diff_best_tnr[item_ood] = val_results[i]['AUROC']
                diff_best_au[item_ood] = val_results[i]['FPR']
                best_val=i
            elif diff_best_tnr[item_ood] == val_results[i]['AUROC'] and diff_best_au[item_ood] > val_results[i]['FPR']:
                diff_best_tnr[item_ood] = val_results[i]['AUROC']
                diff_best_au[item_ood] = val_results[i]['FPR']
                best_val=i
            #print('\n')
        print('AUROC:',end=' ')
        for i in range(len(val_results)):
            print('%.3f'%(val_results[i]['AUROC']), end=' ')
        print('\nFPR:',end=' ')
        for i in range(len(val_results)):
            print('%.3f'%(val_results[i]['FPR']), end=' ')
        print('\n')
        if (diff_gradient_best_tnr[item_ood]<diff_best_tnr[item_ood]) or \
                    (diff_gradient_best_tnr[item_ood]==diff_best_tnr[item_ood] and diff_gradient_best_au[item_ood] > diff_best_au[item_ood]):
            diff_gradient_best_tnr[item_ood]=diff_best_tnr[item_ood]
            diff_gradient_best_au[item_ood]=diff_best_au[item_ood]
            diff_gradient_best_val[item_ood]=best_val
            diff_gradient_best_temperature[item_ood]= T_list[int(diff_gradient_best_val[item_ood]//(len(M_list)*len(rotation_list)))]#temperature
            diff_gradient_best_magnitude[item_ood] = M_list[int((diff_gradient_best_val[item_ood]%(len(M_list)*len(rotation_list)))//len(rotation_list))]#magnitude
            diff_gradient_best_rotation[item_ood] = rotation_list[int((diff_gradient_best_val[item_ood]%(len(M_list)*len(rotation_list)))%len(rotation_list))]
    if just_val:
        return 1, 1, 1, 1, 1, 1 
    for item_ood, (ood_dataset,ood_name) in enumerate(out_dataset_list):
        #此时需要修改loader，切换为OOD测试集
        args.ood_dataset_name=ood_name
        print('Out-distribution: ' + ood_name)
        dataset = datasets[2]
        dataset.load_from_h5(True)
        dataset.return_slideid()
        #test_loader = get_simple_loader(dataset)
        test_loader = get_coords_id_loader(dataset)
        perturbation_feature_path=os.path.join(args.data_root_dir,'%s_perturbation'%(args.dataset_name))
        Stadis_test_score(net,test_loader,perturbation_feature_path,output_path,True,diff_gradient_best_magnitude[item_ood],
                            diff_gradient_best_temperature[item_ood],diff_gradient_best_rotation[item_ood],magnitude_type,diff_type,agg_type,args,oodname=str(item_ood))
        ood_dataset = ood_dataset[2]
        ood_loader = get_coords_id_loader(ood_dataset)
        num_oodimages = len(ood_dataset)
        ood_perturbation_feature_path=os.path.join(args.data_root_dir,'%s_perturbation'%(ood_name))
        Stadis_test_score(net,ood_loader,ood_perturbation_feature_path,output_path,False,diff_gradient_best_magnitude[item_ood],
                            diff_gradient_best_temperature[item_ood],diff_gradient_best_rotation[item_ood],magnitude_type,diff_type,agg_type,args,oodname=str(item_ood))
        test_path=output_path
        test_results = openood_metric(test_path,str(item_ood), ['diff_test'],False)
        diff_best_results[item_ood]=test_results
        print(test_results[0]['AUROC'])
    
    print('the total time is %.2f'%(time.time()-tt))
    # print the results
    mtypes = ['FPR','FPR80','TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    result_fpr=[]
    result_fpr80=[]
    result_auc=[]
    result_aupr_in=[]
    result_aupr_out=[]
    best_threshold = [] 
    print('DIFF method: in_distribution: ' + args.dataset_name +'======'+magnitude_type+ '=========='+args.exp_code)
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
        print('threshold_rotation:'+ str(diff_gradient_best_rotation[ood_item]))
        print('')
        result_fpr.append(results[0]['FPR'])
        result_fpr80.append(results[0]['FPR80'])
        result_auc.append(results[0]['AUROC'])
        result_aupr_in.append(results[0]['AUIN'])
        result_aupr_out.append(results[0]['AUOUT'])
        threshold = '%s+%s'%(str(diff_gradient_best_magnitude[ood_item]),str(diff_gradient_best_rotation[ood_item]))
        best_threshold.append(threshold)
    #if len(out_dataset_list) == 1:
    #    return best_threshold, diff_best_results[0][0]['FPR'], diff_best_results[0][0]['AUROC'], diff_best_results[0][0]['AUIN'], diff_best_results[0][0]['AUOUT']
    return best_threshold, result_fpr, result_auc, result_aupr_in, result_aupr_out,result_fpr80

def modification_path(train_path,train_path_new,old_name,new_name):
    # 打开原始文件和创建新文件进行写入
    with open(train_path, 'r') as file, open(train_path_new, 'w') as new_file:
    # 逐行读取原始文件
        for line in file:
            # 替换每行中的 'data_nas2' 为 'data_nas'
            modified_line = line.replace(old_name, new_name)
            # 将修改后的行写入新文件
            new_file.write(modified_line)

parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples about pathology image')


#需要修改的
parser.add_argument('--model_name', type=str, default='conch_v1', choices=['resnet50_trunc', 'uni_v1', 'conch_v1','resnet50'])
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'transmil', 'abmil'], default='transmil', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--experiment_mask', type = str,default='RCC_conch',choices=['RCC','C16','RCC_conch','C16_conch'],
					help='name about experiment')
parser.add_argument('--task', type=str, default='task_2_tumor_subtyping', choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'])


#默认的\
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')#3
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')#4
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--fea_csv_path', type = str,default='./dataset_csv',
					help='name about experiment')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')#default='task_2_RCC_CLAM_50'
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--embed_dim', type=str, default=1024)



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
parser.add_argument('--gpu', type=int, default=0)
parser.set_defaults(argument=True)
args=parser.parse_args()
if args.model_name == 'conch_v1':
    args.batch_size = 128
if args.model_name == 'conch_v1':
    args.embed_dim = 512
args.id_csv_path = os.path.join(args.csv_path,args.experiment_mask,'process_list_autogen.csv')
args.splits_dir = os.path.join(args.splits_dir, args.task+'_{}_{}'.format(args.experiment_mask,(args.label_frac*100))) 
args.id_data_h5_dir = os.path.join(args.data_h5_dir, args.experiment_mask) 
args.dataset_name=args.experiment_mask
args.exp_code='%s_%s_%s'%(args.experiment_mask, args.model_type,args.model_name)
args.models_dir = os.path.join(args.model_results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
args.data_root_dir =os.path.join(args.data_root_dir,'clam_format_feature_%s'%(args.model_name))
args.csv_path_id = os.path.join(args.fea_csv_path,'%s_dummy_clean_%s.csv'%(args.task,args.experiment_mask))

out_dataset_list=[]
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = args.csv_path_id,
                            data_dir= os.path.join(args.data_root_dir, args.experiment_mask),
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
    ood_csv_path_list ={'TCGA_lung_ood':'./dataset_csv/task_1_tumor_vs_normal_dummy_clean_TCGA_lung_ood.csv',
					  'TCGA_breast_ood':'./dataset_csv/task_1_tumor_vs_normal_dummy_clean_TCGA_breast_ood.csv'}
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
elif args.task == 'task_2_tumor_subtyping':
    ood_name_list=['RCC_ood','RCC_Normal']
    ood_name_list=['RCC_ood']
    ood_name = 'RCC_ood'
    ood_name_base = 'RCC_ood'
    if args.model_name =='conch_v1':
        ood_name = 'RCC_ood_conch'
        
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path=args.csv_path_id,
                            data_dir= os.path.join(args.data_root_dir, args.experiment_mask),
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
if __name__=='__main__':
    torch.cuda.set_device(args.gpu)
    just_val = True
    all_results = []
    all_auc = []
    all_acc = []
    all_fpr = []
    all_fpr80 = []
    all_aupr_in = []
    all_aupr_out = []
    all_tpr = []
    thresholds = []
    result_path=os.path.join(args.result_path,args.exp_code)
    os.makedirs(result_path, exist_ok=True)
    for ckpt_idx in range(len(ckpt_paths)):
        print('The %d-fold'%(folds[ckpt_idx]))
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/ood_detection_splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])           
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)#0:val 1:test 2:train(null)
            #split_dataset = datasets[datasets_id[args.split]]
        threshold, fpr, auc, aupr_in, aupr_out,fpr80 = diff_feature_ood(datasets,ckpt_paths[ckpt_idx],args,out_dataset_list,just_val = just_val)
        #all_results.append(results)
        tpr = []
        for i in range(len(out_dataset_list)):
            tpr.append(1-fpr[i])
        all_fpr.append(fpr)
        all_fpr80.append(fpr80)
        all_tpr.append(tpr)
        all_auc.append(auc)
        all_aupr_in.append(aupr_in)  
        all_aupr_out.append(aupr_out)    
        thresholds.append(threshold)
    if just_val == False:
        final_df_dict = {'folds': folds}
        for i, (ood_dataset,ood_name) in enumerate(out_dataset_list):
            final_df_dict['ood_fpr_{}'.format(ood_name)] = [fpr[i] for fpr in all_fpr]
            final_df_dict['ood_fpr80_{}'.format(ood_name)] = [fpr[i] for fpr in all_fpr80]
            final_df_dict['ood_tpr_{}'.format(ood_name)] = [tpr[i] for tpr in all_tpr]
            final_df_dict['all_auc_{}'.format(ood_name)] = [auc[i] for auc in all_auc]
            final_df_dict['ood_aupr_in_{}'.format(ood_name)] = [aupr[i] for aupr in all_aupr_in]
            final_df_dict['ood_aupr_out_{}'.format(ood_name)] = [aupr[i] for aupr in all_aupr_out]
            final_df_dict['threshold_{}'.format(ood_name)] = [threshold[i] for threshold in thresholds]
        final_df = pd.DataFrame(final_df_dict)
        #average_row = final_df.mean()
        #final_df.loc['Average'] = average_row
        save_name = 'StaDis_summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
        final_df.to_csv(os.path.join(result_path, save_name))
    print('end')





