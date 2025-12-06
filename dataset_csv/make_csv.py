import argparse
import pandas as pd
from torch.utils.data import Dataset 
import os

parser = argparse.ArgumentParser(description='Get csv')
#需要修改的
parser.add_argument('--experiment_mask', type = str,default='RCC',
					choices=['RCC','RCC_ood','C16','TCGA_lung_ood','TCGA_breast_ood','RCC_conch','RCC_ood_conch','C16_conch','TCGA_lung_ood_conch','TCGA_breast_ood_conch'],
					help='name about experiment')
parser.add_argument('--task', type = str,default='task_1_tumor_vs_normal',choices=['task_1_tumor_vs_normal','task_2_tumor_subtyping'],
					help='name about experiment')
#default
parser.add_argument('--csv_path', type=str)
parser.add_argument('--output', type=str, default='./dataset_csv/')

args = parser.parse_args()
args.csv_path = os.path.join(args.csv_path,args.experiment_mask,'process_list_autogen.csv')
class Dataset_All_Bags(Dataset):
	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
		self.df = self.df[self.df['status'] != 'failed_seg']
		self.df = self.df.reset_index(drop=True)
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]
def convert_to_number(str1):
    if isinstance(str1, str):
        try:
            # 尝试将字符串转换为数字
            return int(str1)
        except ValueError:
            # 如果转换失败,说明不是数字,直接返回原字符串
            return str1
    else:
        # 如果输入不是字符串,直接返回原输入
        return str1
	
def label_dict(df,args):
	if args.experiment_mask in ['RCC','RCC_ood','RCC_conch','RCC_ood_conch']:
		df['wild_patient_ID'] = df['slide_id'].apply(lambda x: x.split('/')[-1].split('.')[0].split('-')[2])
		df['wild_label'] = df['slide_id'].apply(lambda x: x.split('/')[-1].split('.')[0].split('-')[1])
		df['is_cancer'] = df['slide_id'].apply(lambda x: x.split('/')[-1].split('.')[0].split('-')[3])
		TCGA_name='/data_nas2/zd/paper1/input/TCGA_name.xlsx'
		df_TCGA = pd.read_excel(TCGA_name)
		TCGA_dict = dict(zip(df_TCGA.iloc[:, 0], df_TCGA.iloc[:, 2]))
		for key, value in TCGA_dict.items():
			if value == 'Kidney Chromophobe':#kich
				TCGA_dict[key] = 'subtype_1'
			elif value == 'Kidney renal clear cell carcinoma':#kirc
				TCGA_dict[key] = 'subtype_2'
			elif value == 'Kidney renal papillary cell carcinoma':#kirp
				TCGA_dict[key] = 'subtype_3'
			elif value == 'Sarcoma':#sarc
				TCGA_dict[key] = 3
		csv_df = pd.DataFrame({'slide_id_name': df['slide_id']})#第一列为'slide_id'。第二列为'case_id'。第三列为slide_id。第四列为label
		case_id=[]
		patient={}
		label=[]
		for index, row in df.iterrows():
        	# 这里可以根据需要进行处理，假设我们提取 slide_id 的特定部分并添加到新列、
			wild_patient_ID = row['wild_patient_ID']
			if wild_patient_ID not in patient:
				case_id.append(('patient_'+str(len(patient))))
				patient[wild_patient_ID] =  len(patient)
			else:
				case_id.append(('patient_'+str(patient[wild_patient_ID])))
			TCGA_class_name = convert_to_number(row['wild_label'])
			try:
				if TCGA_dict[TCGA_class_name] not in ['subtype_1','subtype_2','subtype_3',3]:
					print(index)
				if int(row['is_cancer'][:2])>10:
					label.append(0)
				else:
					label.append(TCGA_dict[TCGA_class_name])

			except KeyError:
				print("The key '%s' does not exist in flag."%(TCGA_class_name))
		csv_df['case_id'] = case_id
		csv_df['label'] = label
		csv_df['case_id_num'] = csv_df['case_id'].str.extract('(\d+)').astype(int)
		csv_df = csv_df.sort_values(by='case_id_num').reset_index(drop=True)
		csv_df = csv_df.drop(columns=['case_id_num'])
		#csv_df = csv_df.drop(columns=['slide_id'])
		csv_df['slide_id'] = ['slide_%d' % k for k in range(len(csv_df))]
	elif args.experiment_mask in ['C16','C16_conch']:
		TCGA_dict = {}
		csv_dir='/data_nas2/zd/paper1/output/patch/C16/split_train_val.csv'
		slide_data=pd.read_csv(csv_dir, index_col=0)
		wsi_train =slide_data.loc[:, 'train'].dropna().values
		wsi_val = slide_data.loc[:, 'val'].dropna().values
		wsi_test= slide_data.loc[:, 'test'].dropna().values
		csv_df = pd.DataFrame({'slide_id_name': df['slide_id']})#第一列为'slide_id'。第二列为'case_id'。第三列为slide_id。第四列为label
		case_id=[]
		label=[]
		patient={}
		for index, row in df.iterrows():
        	# 这里可以根据需要进行处理，假设我们提取 slide_id 的特定部分并添加到新列、
			name = row['slide_id']
			type_wsi = name.split('/')[-1].split('_')[0]
			wild_patient_ID = name.split('/')[-1].rstrip('.svs')
			if wild_patient_ID not in patient:
				case_id.append(('patient_'+str(len(patient))))
				patient[wild_patient_ID] =  len(patient)
			else:
				case_id.append(('patient_'+str(patient[wild_patient_ID])))
			if type_wsi in ['normal']:
				label.append(0)
			elif type_wsi in ['tumor']:
				label.append(1)
			else:	
				wild_patient = wild_patient_ID.split('.')[0]
				wsilabel = slide_data.loc[slide_data['test'] == wild_patient, 'test_label'].values
				label.append(int(wsilabel[0]))
			
		csv_df['case_id'] = case_id
		csv_df['label'] = label
		csv_df['case_id_num'] = csv_df['case_id'].str.extract('(\d+)').astype(int)
		csv_df = csv_df.sort_values(by='case_id_num').reset_index(drop=True)
		csv_df = csv_df.drop(columns=['case_id_num'])
		#csv_df = csv_df.drop(columns=['slide_id'])
		csv_df['slide_id'] = ['slide_%d' % k for k in range(len(csv_df))]
	elif args.experiment_mask in ['TCGA_lung_ood','TCGA_breast_ood','TCGA_lung_ood_conch','TCGA_breast_ood_conch']:
		if args.experiment_mask in ['TCGA_lung_ood','TCGA_lung_ood_conch']:
			df = df.sample(n=120, random_state=42)
		else:
			df = df.sample(n=135, random_state=42)
		df['wild_patient_ID'] = df['slide_id'].apply(lambda x: x.split('/')[-1].split('.')[0].split('-')[2])
		TCGA_name='/data_nas2/zd/paper1/input/TCGA_name.xlsx'
		csv_df = pd.DataFrame({'slide_id_name': df['slide_id']})#第一列为'slide_id'。第二列为'case_id'。第三列为slide_id。第四列为label
		case_id=[]
		label=[]
		patient={}
		for index, row in df.iterrows():
        	# 这里可以根据需要进行处理，假设我们提取 slide_id 的特定部分并添加到新列、
			wild_patient_ID = row['wild_patient_ID']
			if wild_patient_ID not in patient:
				case_id.append(('patient_'+str(len(patient))))
				patient[wild_patient_ID] =  len(patient)
			else:
				case_id.append(('patient_'+str(patient[wild_patient_ID])))
			
			label.append(-1)
		csv_df['case_id'] = case_id
		csv_df['label'] = label
		csv_df['case_id_num'] = csv_df['case_id'].str.extract('(\d+)').astype(int)
		csv_df = csv_df.sort_values(by='case_id_num').reset_index(drop=True)
		csv_df = csv_df.drop(columns=['case_id_num'])
		csv_df['slide_id'] = ['slide_%d' % k for k in range(len(csv_df))]
	else:
		raise NotImplementedError

	return csv_df


if __name__ == '__main__':
	output = os.path.join(args.output,'%s_dummy_clean_%s.csv'%(args.task,args.experiment_mask))
	df = pd.read_csv(args.csv_path)
	df = df[df['status'] != 'failed_seg']
	df = df.reset_index(drop=True)
	csv_df = label_dict(df,args)
	csv_df.to_csv(output, index=False)
    #total = len(bags_dataset)
	






