'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-06-08 15:18:32
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-06-08 20:18:43
FilePath: /mycode/StaDis/create_splits_seq.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pdb
import os
import pandas as pd
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--task', type=str, default='task_1_tumor_vs_normal', choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping'])
parser.add_argument('--experiment_mask', type = str,default='C16_conch',
                    choices=['RCC','C16','RCC_conch','C16_conch'],
					help='name about experiment')

#default
parser.add_argument('--split_path', type = str,default='./splits/',
					help='name about experiment')
parser.add_argument('--csv_path', type = str,default='./dataset_csv',
					help='name about experiment')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=5,
                    help='number of splits (default: 10)')
parser.add_argument('--val_frac', type=float, default= 0.15,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.15,
                    help='fraction of labels for test (default: 0.1)')
args = parser.parse_args()
args.csv_path = os.path.join(args.csv_path,'%s_dummy_clean_%s.csv'%(args.task,args.experiment_mask))
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = args.csv_path,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1},
                            patient_strat=True,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = args.csv_path,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

else:
    raise NotImplementedError
num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

if __name__ == '__main__':
    #python create_splits_seq.py --task task_1_tumor_vs_normal --seed 1 --k 10
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = args.split_path+ str(args.task) + '_{}_{}'.format(args.experiment_mask,lf*100)
        os.makedirs(split_dir, exist_ok=True)
        if args.task == 'task_1_tumor_vs_normal':
            my_list = list(range(270, 399))
            dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf,custom_test_ids = my_list)
        else:
            dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



