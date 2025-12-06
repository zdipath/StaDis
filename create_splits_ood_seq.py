#这一步只分割ood detection中验证集的扰动特征（多个参数），在ood detection模块中增加一个提取特征的函数。
import time
import os
import argparse
import torch



from ood_utils.utils_Stadis import ood_dataset_splits
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='Feature Extraction')
#需要修改的参数
parser.add_argument('--experiment_mask', type = str,default='C16_conch',choices=['RCC','C16','RCC_conch','C16_conch'],
					help='name about experiment')
parser.add_argument('--model_name', type=str, default='conch_v1', choices=['resnet50_trunc', 'uni_v1', 'conch_v1', 'resnet50'])
parser.add_argument('--task', type=str, default='task_1_tumor_vs_normal', choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping'])


#default
parser.add_argument('--feat_dir', type=str, default='./data/')
parser.add_argument('--ood_feat_dir', type=str, default='./data/')
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--dataset_csv', type=str, default='./dataset_csv/')
parser.add_argument('--data_h5_dir', type=str, default='./data/clam_format_patch/')
parser.add_argument('--csv_path', type=str, default='./data/clam_format_patch/')
parser.add_argument('--slide_id_csv', type=str, default='./dataset_csv/')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--seed', type=int, default=1)
#parser.add_argument('--ood_slide_id_csv', type=str, default='./dataset_csv/tumor_subtyping_dummy_clean_RCC_ood.csv')
parser.add_argument('--slide_ext', type=str, default= '.svs')

args = parser.parse_args()
args.feat_dir = os.path.join(args.feat_dir,'clam_format_feature_%s'%(args.model_name))
args.ood_feat_dir_temp = os.path.join(args.ood_feat_dir,'clam_format_feature_%s'%(args.model_name))
args.id_data_h5_dir = os.path.join(args.data_h5_dir,args.experiment_mask)
args.id_csv_path = os.path.join(args.csv_path,args.experiment_mask,'process_list_autogen.csv')
args.id_slide_id_csv = os.path.join(args.slide_id_csv,'%s_dummy_clean_%s.csv'%(args.task,args.experiment_mask))
args.feat_dir = os.path.join(args.feat_dir,'%s_perturbation'%(args.experiment_mask))
if __name__ == '__main__':
	print('initializing dataset')
	if args.experiment_mask in ['C16','C16_conch']:
		args.slide_ext = '.tif'
		#ood_name_list=['TCGA_lung_ood','TCGA_breast_ood']
		#ood_csv_path_list ={'TCGA_lung_ood':'./dataset_csv/task_1_tumor_vs_normal_dummy_clean_TCGA_lung_ood.csv',
		#			  'TCGA_breast_ood':'./dataset_csv/task_1_tumor_vs_normal_dummy_clean_TCGA_breast_ood.csv'}
		ood_name_list=['RCC']
		ood_csv_path_list ={'RCC':'./dataset_csv/task_2_tumor_subtyping_dummy_clean_RCC.csv',
					  }


	elif args.experiment_mask in ['RCC','RCC_conch']:
		ood_name_list=['RCC_ood', 'RCC_Normal']
		ood_csv_path_list =['./dataset_csv/tumor_subtyping_dummy_clean_RCC_ood.csv',
					  './dataset_csv/tumor_subtyping_dummy_clean_RCC_Normal.csv']

	#1.增加一个区分OOD detection验证集与测试集的模块
	#由于10步交叉验证的存在，需要进行10次
	#m_list=[50,20,10,5,2,1,0.6,0.2,0.1,0.01,0.001,0.0001,0,-20,-10,-5,-2,1,-0.6,-0.2,-0.1,-0.01,-0.001]
	#m_list=[20,1,0.0014,0.0001,0]#最多5个参数
	m_list=[50,20,10,5,0.1,0.01]#最多5个参数
	args.split_dir = os.path.join('./splits', args.task+'_{}_{}'.format(args.experiment_mask,(args.label_frac*100))) 
	ood_dataset_splits(args.split_dir,args,ood_name_list,ood_csv_path_list)

	print('end')





