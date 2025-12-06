import numpy as np
import pandas as pd
import os
import time 
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
#from ood_utils import single_feature_detection_performance, detection_performance
#from ood_utils import metric
import math
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from sklearn import metrics
import sys


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

def ood_dataset_splits(split_dir,args,ood_name_list,ood_csv_path_list):
    np.random.seed(args.seed)
    rate = 0.2
    #读取splits_0.csv文件，然后将测试集划分为ood验证与ood测试（10）
    for i in range(args.k):
        split_test_dir=os.path.join(split_dir, 'splits_{}.csv'.format(i))
        all_splits = pd.read_csv(split_test_dir)
        train_data = all_splits['train'].values
        split = all_splits['test']
        split = split.dropna().reset_index(drop=True)
        total_samples = len(split)
        val_size = int(total_samples * rate)
        val_indices = np.random.choice(split.index, size=val_size, replace=False)
        test_indices = split.index.difference(val_indices)
        
        val_data = split.iloc[val_indices].values
        test_data = split.iloc[test_indices].values
        max_length = max(len(val_data), len(test_data), len(train_data))
        # 创建 DataFrame 保存划分后的结果，并用 NaN 补齐
        split_df = pd.DataFrame({
            'val': pd.Series(val_data).reindex(range(max_length)),
            'test': pd.Series(test_data).reindex(range(max_length)),
			'train': pd.Series(train_data).reindex(range(max_length))
            })
        ood_split_dir=os.path.join(split_dir, 'ood_detection_splits_{}.csv'.format(i))
        split_df.to_csv(ood_split_dir, index=False)
    #ood data
    for ood_name in ood_name_list:
        if ood_name == 'RCC_Normal':
            input = os.path.join(args.dataset_csv,'%s_dummy_clean_%s.csv'%(args.task,args.experiment_mask))
            df = pd.read_csv(input)
            ood_path = os.path.join(args.dataset_csv,'%s_dummy_clean_%s.csv'%(args.task,'RCC_ood'))
            ood_df = pd.read_csv(ood_path)
            normal1 = df[df['label'] == '0']['slide_id'].values
            normal2 = ood_df[ood_df['label'] == '0']['slide_id'].values
            normal3 = ood_df[ood_df['label'] == '3']['slide_id'].values
            #将normal1单独出来
            normal1 = np.array(normal1)
            val_size = math.ceil(normal1.shape[0] * rate)
            val_indices = np.random.choice(normal1.shape[0], size=val_size, replace=False)
            val = normal1[val_indices]
            test = np.delete(normal1, val_indices)
            max_length = max(val.shape[0], test.shape[0])
            split_df = pd.DataFrame({
                    'val': pd.Series(val).reindex(range(max_length)),
                    'test': pd.Series(test).reindex(range(max_length)),
			        'train': [None] * max_length
                    })
            ood_split_dir=os.path.join(split_dir, 'ood_detection_splits_RCC_normal_ffpe.csv')
            split_df.to_csv(ood_split_dir, index=False)
            #将normal2+normal3一起
            combined_list = np.concatenate((normal2, normal3))
            
            #combined_list = np.array(combined_list)
            val_size = math.ceil(combined_list.shape[0] * rate)
            val_indices = np.random.choice(combined_list.shape[0], size=val_size, replace=False)
            val = combined_list[val_indices]
            test = np.delete(combined_list, val_indices)
            max_length = max(val.shape[0], test.shape[0])
            split_df = pd.DataFrame({
                    'val': pd.Series(val).reindex(range(max_length)),
                    'test': pd.Series(test).reindex(range(max_length)),
			        'train': [None] * max_length
                    })
            ood_split_dir=os.path.join(split_dir, 'ood_detection_splits_RCC_normal_frozen.csv')
            split_df.to_csv(ood_split_dir, index=False)
        elif ood_name == 'RCC_ood':
            slide_data = pd.read_csv('./dataset_csv/task_2_tumor_subtyping_dummy_clean_RCC_ood.csv')
            slide_data = filter_df(slide_data, filter_dict={'label':['subtype_1','subtype_2','subtype_3']})
            slide_data = df_prep(slide_data, label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2}, ignore=[], label_col= 'label')
            split = slide_data['slide_id']
            split = split.dropna().reset_index(drop=True)
            total_samples = len(split)
            val_size = math.ceil(total_samples * rate)
            val_indices = np.random.choice(split.index, size=val_size, replace=False)
            test_indices = split.index.difference(val_indices)
            val_data = split.iloc[val_indices].values
            test_data = split.iloc[test_indices].values
            max_length = max(len(val_data), len(test_data))
            # 创建 DataFrame 保存划分后的结果，并用 NaN 补齐
            split_df = pd.DataFrame({
                    'val': pd.Series(val_data).reindex(range(max_length)),
                    'test': pd.Series(test_data).reindex(range(max_length)),
		        	'train': [None] * max_length
                    })
            ood_split_dir=os.path.join(split_dir, 'ood_detection_splits_RCC_ood.csv')
            split_df.to_csv(ood_split_dir, index=False)
        else:
            slide_data = pd.read_csv(ood_csv_path_list[ood_name])
            
            if ood_name == 'RCC':
                slide_data = filter_df(slide_data, filter_dict={'label':['subtype_1','subtype_2','subtype_3']})
                slide_data = df_prep(slide_data, label_dict = {'subtype_1':-1, 'subtype_2':-1, 'subtype_3':-1}, ignore=[], label_col= 'label')
            else:
                slide_data = filter_df(slide_data, filter_dict={})
                slide_data = df_prep(slide_data, label_dict = {-1:-1}, ignore=[], label_col= 'label')
            split = slide_data['slide_id']
            split = split.dropna().reset_index(drop=True)
            total_samples = len(split)
            val_size = math.ceil(total_samples * rate)
            val_indices = np.random.choice(split.index, size=val_size, replace=False)
            test_indices = split.index.difference(val_indices)
            val_data = split.iloc[val_indices].values
            test_data = split.iloc[test_indices].values
            max_length = max(len(val_data), len(test_data))
            # 创建 DataFrame 保存划分后的结果，并用 NaN 补齐
            split_df = pd.DataFrame({
                    'val': pd.Series(val_data).reindex(range(max_length)),
                    'test': pd.Series(test_data).reindex(range(max_length)),
		        	'train': [None] * max_length
                    })
            ood_split_dir=os.path.join(split_dir, 'ood_detection_splits_%s.csv'%(ood_name))
            split_df.to_csv(ood_split_dir, index=False)


def df_prep(data, label_dict, ignore, label_col):
		if label_col != 'label':
			data['label'] = data[label_col].copy()

		mask = data['label'].isin(ignore)
		data = data[~mask]
		data.reset_index(drop=True, inplace=True)
		for i in data.index:
			key = data.loc[i, 'label']
			data.at[i, 'label'] = label_dict[key]

		return data

def filter_df(df, filter_dict={}):
	if len(filter_dict) > 0:
		filter_mask = np.full(len(df), True, bool)
		# assert 'label' not in filter_dict.keys()
		for key, val in filter_dict.items():
			mask = df[key].isin(val)
			filter_mask = np.logical_and(filter_mask, mask)
		df = df[filter_mask]
	return df



def auc_and_fpr_recall_threa(conf, label, tpr_th):
    # following convention in ML we treat OOD as positive
    #ood_indicator = np.zeros_like(label)
    #ood_indicator[label == -1] = 1
    ood_indicator=label
    #conf = np.array([[0.95], [0.85], [0.75], [0.8], [0.7]])
    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values
    #ood_indicator = np.concatenate((np.ones(int(3)), np.zeros(int(2))))
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, conf)
    #fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]
    index = np.argmax(tpr_list >= tpr_th)
    corres_conf = thresholds[index]
    bool_conf = conf > corres_conf
    return bool_conf

def openood_metric_score(dir_name, oodname, stypes = ['Bas', 'Gau'], verbose=False):
    np.set_printoptions(precision=3)
    for stype in stypes:
        known = np.loadtxt('{}/Probability_{}_In_{}.txt'.format(dir_name, stype, oodname))#, delimiter='\\n')#output_path/Probability_val_In.txt
        novel = np.loadtxt('{}/Probability_{}_Out_{}.txt'.format(dir_name, stype, oodname))#, delimiter='\\n')
    if known.ndim==1:
        known = known.reshape((known.shape[0], 1))
        novel = novel.reshape((novel.shape[0], 1))
    confs = np.concatenate([known, novel])
    label = np.concatenate((np.ones(int(known.shape[0])), np.zeros(int(novel.shape[0]))))
    recall = 0.8
    results = dict()
    num_para=known.shape[1]
    for i in range(num_para):
        conf=confs[:,i].reshape(confs.shape[0], 1)
        bool_conf = auc_and_fpr_recall_threa(conf, label, recall)
        known_conf = bool_conf[:known.shape[0]]
        novel_conf = bool_conf[known.shape[0]:]
        results[i] = dict()
        results[i]['ID']=known_conf
        results[i]['OOD']=novel_conf
    return results


def openood_metric(dir_name, oodname, stypes = ['Bas', 'Gau'], val=False):
    np.set_printoptions(precision=3)
    for stype in stypes:
        if val:
            known = np.loadtxt('{}/Probability_{}_In.txt'.format(dir_name, stype))
        else:
            known = np.loadtxt('{}/Probability_{}_In_{}.txt'.format(dir_name, stype, oodname))#, delimiter='\\n')#output_path/Probability_val_In.txt
        novel = np.loadtxt('{}/Probability_{}_Out_{}.txt'.format(dir_name, stype, oodname))#, delimiter='\\n')
    if known.ndim==1:
        known = known.reshape((known.shape[0], 1))
        novel = novel.reshape((novel.shape[0], 1))
    confs = np.concatenate([known, novel])
    label = np.concatenate((np.ones(int(known.shape[0])), np.zeros(int(novel.shape[0]))))
    recall = 0.95
    results = dict()
    num_para=known.shape[1]
    for i in range(num_para):
        conf=confs[:,i].reshape(confs.shape[0], 1)
        auroc, aupr_in, aupr_out, fpr,fpr80 = auc_and_fpr_recall(conf, label, recall)
        results[i] = dict()
        results[i]['FPR']=fpr
        results[i]['TNR']=1-fpr
        results[i]['AUROC'] = auroc
        results[i]['DTACC'] = 0
        results[i]['AUIN'] = aupr_in
        results[i]['AUOUT'] = aupr_out
        results[i]['FPR80'] = fpr80
    return results



def acc(pred, label):
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]

    num_tp = np.sum(ind_pred == ind_label)
    acc = num_tp / len(ind_label)

    return acc
def auc_and_fpr_recall(conf, label, tpr_th):
    # following convention in ML we treat OOD as positive
    #ood_indicator = np.zeros_like(label)
    #ood_indicator[label == -1] = 1
    ood_indicator=label
    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, conf)
    unique_values = np.unique(conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]
    fpr80 = fpr_list[np.argmax(tpr_list >= 0.8)]
    precision_in, recall_in, thresholds_in \
        = metrics.precision_recall_curve(ood_indicator, conf)

    precision_out, recall_out, thresholds_out \
        = metrics.precision_recall_curve(1-ood_indicator, -conf)

    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out, fpr,fpr80

def prepare_mixup(batch, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = batch.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    return index, lam


def mixing(data, index, lam):
    return lam * data + (1 - lam) * data[index]

class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, inputs, targets):
        #inputs = F.softmax(inputs, dim=1)
        #targets = F.softmax(targets, dim=1)
        return ((inputs - targets) ** 2).mean()

# 使用自定义损失函数
class MAELoss(torch.nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, inputs, targets):
        #inputs = F.softmax(inputs, dim=1)
        #targets = F.softmax(targets, dim=1)
        absolute_difference = torch.abs(inputs - targets)
        # 计算绝对差异的平均值
        loss = torch.mean(absolute_difference)
        return loss#((inputs - targets) ** 2).mean()
    

class SingleMAELoss(torch.nn.Module):
    def __init__(self):
        super(SingleMAELoss, self).__init__()

    def forward(self, inputs, targets):
        #inputs = F.softmax(inputs, dim=1)
        #targets = F.softmax(targets, dim=1)
        absolute_difference = torch.abs(inputs - targets)
        # 计算绝对差异的平均值
        loss = absolute_difference.mean(dim=1)
        #loss = torch.mean(absolute_difference)
        return loss
def diff_softmax_logit(data,model, output, magnitude_type,net_type,magnitude_list,temperature,diff_type,threshold_gradient):
    criterion = torch.nn.CrossEntropyLoss()#交叉熵不需要经过softmax，因为自带
    criterion = MSELoss()
    is_rotation=True
    for num_j in range(len(magnitude_list)):
        if magnitude_type in ['unified','multiplication']:
            gradient = torch.ones_like(data)
        elif magnitude_type == 'rotation':
            gradient=0
        else:
            print('No magnitude_type')
        

        if magnitude_type=='rotation':#增加了数据增强
            if threshold_gradient%4 == 1:
                temp_x_noise = torch.rot90(data.data, 1, [2, 3])
            elif threshold_gradient%4 ==2:
                temp_x_noise = torch.rot90(data.data, 2, [2, 3])
            elif threshold_gradient%4 == 3:
                temp_x_noise = torch.rot90(data.data, 3, [2, 3])
            else:
                temp_x_noise=data.data
            index, lam = prepare_mixup(temp_x_noise, 1)
            #noise_Input = mixing(temp_x_noise, index, lam)
        else:
            if net_type == 'densenet':
                #gradient.index_select将第0列的元素除以(63.0/255.0)，
                #index_copy_将上一步得到的结果复制回原始梯度张量的第0列，表示直接修改原始张量而不创建新的张量
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
                gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
                gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
            else:
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
                gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
                gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
            noise_Input=torch.add(data.data, gradient,alpha= -magnitude_list[num_j])
        if is_rotation==True and magnitude_type in ['unified', 'loss', 'React','multiplication']:
            if threshold_gradient %4 ==1:
                noise_Input = torch.rot90(noise_Input, 1, [2, 3])
            elif threshold_gradient %4 ==2:
                noise_Input = torch.rot90(noise_Input, 2, [2, 3])
            elif threshold_gradient %4 ==3:
                noise_Input = torch.rot90(noise_Input, 3, [2, 3])
            else:
                noise_Input=data.data
            index, lam = prepare_mixup(noise_Input, 1)
            noise_Input = mixing(noise_Input, index, lam)
        noise_Output=model(noise_Input)/temperature#noise_output与output都是batch_size*num_classes的pytorch数组。
        if diff_type=='softmax':
            noise_Output=F.softmax(noise_Output, dim=1)
            diff_Output=torch.abs(noise_Output-output).max(dim=1)[0]
        else:
            diff_Output=torch.mean(torch.abs(noise_Output-output),1)
        if num_j==0:
            out=diff_Output.detach().cpu().numpy().reshape(diff_Output.shape[0],1)
        else:
            out=np.concatenate((out,diff_Output.detach().cpu().numpy().reshape(diff_Output.shape[0],1)),axis=1)
    return out



class PathDataset_path_nolabel(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
def wsi_feature_extrator(net_type,feature_net,paths, magnitude_type, magnitude, threshold_gradient):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_dataset = PathDataset_path_nolabel(paths, transform)
    batch_size=8
    shuffle=False
    train_sampler=False#不丢弃最后的数据
    gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=2, pin_memory=True, 
                                drop_last=False)
    out=[]
    for iteration, batch in enumerate(gen):
        #第一步，扰动
        batch = batch.cuda()
        if magnitude_type == 'multiplication':
            temp_x_noise = batch * magnitude
        elif magnitude_type == 'unified':
            gradient = torch.ones_like(batch)
            if net_type == 'densenet':
                #gradient.index_select将第0列的元素除以(63.0/255.0)，
                #index_copy_将上一步得到的结果复制回原始梯度张量的第0列，表示直接修改原始张量而不创建新的张量
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / ((63.0/255.0)/(66.7/255.0)))
                gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / ((62.1/255.0)/(66.7/255.0)))
                gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / ((66.7/255.0)/(66.7/255.0)))
            else:
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023/0.2023))
                gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994/0.2023))
                gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010/0.2023)) 
        
            temp_x_noise=torch.add(batch, gradient,alpha= -magnitude)
        #第二步：旋转+mixup
        if threshold_gradient %4 ==1:
            temp_x_noise = torch.rot90(temp_x_noise, 1, [2, 3])
        elif threshold_gradient %4 ==2:
            temp_x_noise = torch.rot90(temp_x_noise, 2, [2, 3])
        elif threshold_gradient %4 ==3:
            temp_x_noise = torch.rot90(temp_x_noise, 3, [2, 3])
        else:
            temp_x_noise=temp_x_noise
        #temp_x_noise = torch.rot90(temp_x_noise, 1, [2, 3])#默认为1
        index, lam = prepare_mixup(temp_x_noise, 1)
        
        temp_x_noise = mixing(temp_x_noise, index, lam)
        
        #第三步，进行特征提取
        if net_type == 'ctranspath':
            classifier = nn.Identity()
            feature_net.head = classifier
            feature=feature_net(temp_x_noise)       
        elif net_type in ['resnet','retccl']:
            temp_list = feature_net.feature_list(temp_x_noise)[1]
            feature = temp_list[-2]
            avgpool = nn.AdaptiveAvgPool2d(1) 
            feature = avgpool(feature)
            feature = feature.view(feature.size(0), -1)
        elif net_type in ['ibotvit','vit']:
            temp_list = feature_net.get_intermediate_layers(temp_x_noise,5)
            feature = temp_list[-1]
            feature=feature[:,0,:]
        features_np = feature.detach().cpu().numpy() 
        out.append(features_np)
    
    out = np.vstack(out)
    out = torch.from_numpy(out).cuda()

    return out













def diff_feature(data,model, output, magnitude_type,net_type,magnitude_list,temperature,diff_type,layer_index,threshold_gradient,wsi_level=False):
    #首先记录原有的特征
    is_rotation=True
    #print(layer_index)
    layer_index = int(layer_index)
    temp_x=data 
    if net_type == 'ctranspath':
        temp_list=ctrans_feature_list(model,temp_x)
        temp_x = temp_list[layer_index]
    elif net_type in ['resnet','retccl']:
        temp_list = model.feature_list(temp_x)[1]
        temp_x=temp_list[layer_index]
    elif net_type in ['ibotvit','vit']:
        temp_list = model.get_intermediate_layers(temp_x,5)
        temp_x=temp_list[layer_index]
    out_features=temp_x
    

    if magnitude_type=='React':
        logits = model(data)
        if threshold_gradient//4==0:
            logits = logits.clip(max=0.5)
        if threshold_gradient//4==1:
            logits = logits.clip(max=1)
        scores = torch.logsumexp(logits.data.cpu(), dim=1)
        scores=1/scores
        scores=scores.view(-1,1,1,1).cuda()
    for num_j in range(len(magnitude_list)):
        if magnitude_type in ['unified','multiplication']:
            gradient = torch.ones_like(data)
        elif magnitude_type=='gradient':
            criterion = torch.nn.CrossEntropyLoss()
            batch_output = model(data)
            outputs = batch_output / temperature
            labels = outputs.data.max(1)[1]
            #outputs = output
            
            model.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            gradient =  (torch.ge(data.grad.data, 0))
            gradient = (gradient.float() - 0.5) * 2
        elif magnitude_type=='loss':
            criterion = torch.nn.CrossEntropyLoss()
            batch_output = model(data)
            #print("前:", torch.cuda.memory_allocated())
            outputs = batch_output / temperature
            labels = outputs.data.max(1)[1]
            #outputs = output
            #下面的代码放到了循环外面
            loss = criterion(batch_output,labels)
            loss.backward()
            model.zero_grad()
            
            
            loss=loss.view(-1,1,1,1)
            gradient = torch.ones_like(data)
            #loss.backward()
            
            
           # grads=torch.abs(data.grad.data)
            gradient*=loss
            del loss
            #print("zhong:", torch.cuda.memory_allocated())
        elif magnitude_type == 'React':
            
            gradient = torch.ones_like(data)
            gradient*=scores

        if magnitude_type=='rotation':#增加了数据增强
            if threshold_gradient%4 == 1:
                temp_x_noise = torch.rot90(data.data, 1, [2, 3])
            elif threshold_gradient%4 ==2:
                temp_x_noise = torch.rot90(data.data, 2, [2, 3])
            elif threshold_gradient%4 == 3:
                temp_x_noise = torch.rot90(data.data, 3, [2, 3])
            else:
                temp_x_noise=data.data
            index, lam = prepare_mixup(temp_x_noise, 1)
            temp_x_noise = mixing(temp_x_noise, index, lam)
        else:
            if net_type == 'densenet':
                #gradient.index_select将第0列的元素除以(63.0/255.0)，
                #index_copy_将上一步得到的结果复制回原始梯度张量的第0列，表示直接修改原始张量而不创建新的张量
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / ((63.0/255.0)/(66.7/255.0)))
                gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / ((62.1/255.0)/(66.7/255.0)))
                gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / ((66.7/255.0)/(66.7/255.0)))
            else:
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) * (0.229/0.229))
                gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) * (0.224/0.229))
                gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) * (0.225/0.229)) 
            if magnitude_type =='multiplication':
                temp_x_noise = data.data * magnitude_list[num_j]
            else:  
                temp_x_noise=torch.add(data.data, gradient,alpha= -magnitude_list[num_j])
            if is_rotation==True and magnitude_type in ['unified', 'loss', 'React', 'multiplication', 'gradient']:
                if threshold_gradient %4 ==1:
                    temp_x_noise = torch.rot90(temp_x_noise, 1, [2, 3])
                elif threshold_gradient %4 ==2:
                    temp_x_noise = torch.rot90(temp_x_noise, 2, [2, 3])
                elif threshold_gradient %4 ==3:
                    temp_x_noise = torch.rot90(temp_x_noise, 3, [2, 3])
                else:
                    temp_x_noise=temp_x_noise
                #temp_x_noise = torch.rot90(temp_x_noise, 1, [2, 3])#默认为1
                index, lam = prepare_mixup(temp_x_noise, 1)
                temp_x_noise = mixing(temp_x_noise, index, lam)
        if net_type == 'ctranspath':
            temp_list=ctrans_feature_list(model,temp_x_noise)
            temp_x_noise = temp_list[layer_index]
        elif net_type in ['resnet','retccl']:
            temp_list = model.feature_list(temp_x_noise)[1]
            temp_x_noise=temp_list[layer_index]
        elif net_type in ['ibotvit','vit']:
            temp_list = model.get_intermediate_layers(temp_x_noise,5)
            temp_x_noise=temp_list[layer_index]
        

        #print("后:", torch.cuda.memory_allocated())
        #noise_out_features = temp_x_noise.view(temp_x_noise.size(0), temp_x_noise.size(1), -1)
        #noise_out_features = torch.mean(noise_out_features, 2)
        type_difinition='(abs+mean)_diff_max'#'(abs+mean)_diff_max'#'diff_abs_max_max'
        if magnitude_type in ['unified', 'multiplication','gradient']:
            if threshold_gradient//4==0:
                type_difinition='(abs+mean)_diff_max'#方案1
            elif threshold_gradient//4==1:
                type_difinition='abs_mean_mean'#方案2
            elif threshold_gradient//4==2:
                type_difinition='abs_mean_max'#方案3
            elif threshold_gradient//4==3:
                type_difinition='diff_abs_max_max'  #方案4
            elif threshold_gradient//4==4:
                type_difinition='abs_max_mean'   #方案5
                if net_type in ['vit','ibotvit']:
                    type_difinition= 'abs_max_0'
                
        '''
        if net_type in ['ibotvit']:
            type_difinition= 'abs_max_0'
        if net_type in ['ctranspath']:
            type_difinition='abs_max_0'
        elif net_type in ['vit']:
            type_difinition='abs_max_0'
        elif net_type in ['resnet']:
            type_difinition= 'abs_mean_mean'
        elif net_type in ['retccl']:
            type_difinition= 'diff_abs_max_max'
        '''
        if type_difinition == 'diff_abs_max_max':#方案4
            diff_feature = torch.abs(temp_x_noise-out_features)
            diff_feature = diff_feature.view(diff_feature.size(0), diff_feature.size(1), -1)
            diff_feature = diff_feature.max(dim=2)[0].max(dim=1)[0]
        elif type_difinition=='(abs+mean)_diff_max':#方案1
            A_flat = temp_x_noise.view(temp_x_noise.size(0), -1)
            B_flat = out_features.view(out_features.size(0), -1)
            diff_feature = F.cosine_similarity(A_flat, B_flat, dim=1)
            diff_feature = 1-diff_feature
            '''
            temp_x_noise = temp_x_noise.view(temp_x_noise.size(0), temp_x_noise.size(1), -1)
            out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
            diff_feature = (torch.mean(torch.abs(temp_x_noise), 2)-torch.mean(torch.abs(out_features), 2)).max(dim=1)[0]'''
        elif type_difinition=='abs_mean_mean':#方案2
            diff_feature = torch.abs(temp_x_noise-out_features)
            diff_feature = diff_feature.view(diff_feature.size(0), diff_feature.size(1), -1)
            diff_feature = torch.mean(torch.mean(diff_feature,2),1)
        elif type_difinition == 'abs_mean_max':#方案3
            diff_feature = torch.abs(temp_x_noise-out_features)
            diff_feature = diff_feature.view(diff_feature.size(0), diff_feature.size(1), -1)
            diff_feature = (torch.mean(diff_feature,2)).max(dim=1)[0]
        elif type_difinition == 'abs_max_mean':#方案5
            diff_feature = torch.abs(temp_x_noise-out_features)
            diff_feature = diff_feature.view(diff_feature.size(0), diff_feature.size(1), -1)
            diff_feature = diff_feature.max(dim=2)[0]
            diff_feature = torch.mean(diff_feature,1)
        elif type_difinition == 'abs_max_0':#方案0
            diff_feature = torch.abs(temp_x_noise-out_features)#8*197*768
            diff_feature = diff_feature.view(diff_feature.size(0), diff_feature.size(1), -1)
            diff_feature = diff_feature.max(dim=2)[0]
            diff_feature = diff_feature[:,0]
        #diff_feature = torch.mean(diff_feature, 2).max(dim=1)[0]

        #diff_feature = torch.mean(diff_feature, 2).max(dim=1)[0]
        
        #diff_feature = torch.mean(diff_feature, 1)
        
        #noise_out_features = torch.mean(noise_out_features, 2)
        #diff_feature=torch.abs(noise_out_features-out_features).max(dim=1)[0]
        #diff_feature=torch.abs(noise_out_features-out_features).max(dim=1)[0]
        if num_j==0:
            out=diff_feature.detach().cpu().numpy().reshape(diff_feature.shape[0],1)
        else:
            diff=diff_feature.detach().cpu().numpy().reshape(diff_feature.shape[0],1)
            out=np.concatenate((out,diff),axis=1)
    #for i in range(27):
    #    print(out[0][i],end=' ')
    return out


def diff_wsi_feature(data,model, output, magnitude_type,net_type,magnitude_list,temperature,diff_type,layer_index,threshold_gradient,wsi_level=False,logit_path=None):
    #首先记录原有的特征
    is_rotation=True
    #print(layer_index)
    #out_features=data#维度为[4000,1024]
    logits, _, _, A_raw, _ = model(data)#data维度为[4000,1024],out_features=[1024] A_raw: [1,2000] logit [1,2]
    for num_j in range(len(magnitude_list)):
        if magnitude_type=='rotation':#增加了数据增强
            if threshold_gradient%4 == 1:
                temp_x_noise = torch.rot90(data.data, 1, [2, 3])
            elif threshold_gradient%4 ==2:
                temp_x_noise = torch.rot90(data.data, 2, [2, 3])
            elif threshold_gradient%4 == 3:
                temp_x_noise = torch.rot90(data.data, 3, [2, 3])
            else:
                temp_x_noise=data.data
            index, lam = prepare_mixup(temp_x_noise, 1)
            temp_x_noise = mixing(temp_x_noise, index, lam)
        else:
            dir_path, file_name = os.path.split(logit_path[0])
            new_dir_path = dir_path.replace('logit_path', 'txt_path')
            new_file_name = file_name.replace('logit', 'path').replace('.npy', '.txt')
            patch_path = os.path.join(new_dir_path, new_file_name)
            paths = []
            with open(patch_path,"r") as f:
                lines = f.readlines()
                for index, line in enumerate(lines):
                    path = line.split(' ')[0].split()[0]
                    paths.append(path)
                paths = np.array(paths)
            temp_x_noise = wsi_feature_extrator(net_type,wsi_level,paths, magnitude_type, magnitude_list[num_j], threshold_gradient)
        type_difinition='(abs+mean)_diff_max'#'(abs+mean)_diff_max'#'diff_abs_max_max'
        if magnitude_type in ['unified', 'multiplication','gradient']:
            if threshold_gradient//4==0:
                type_difinition='abs_weight_mean'#方案1
            elif threshold_gradient//4==1:
                type_difinition='abs_weight_max'#方案2
            elif threshold_gradient//4==2:
                type_difinition='mean_abs_weight'#方案3
            elif threshold_gradient//4==3:
                type_difinition='max_abs_weight'  #方案4
            elif threshold_gradient//4==4:
                type_difinition='weight_abs_mean'   #方案5
            elif threshold_gradient//4==5:
                type_difinition='weight_abs_max'
        
        if type_difinition == 'abs_weight_mean':#方案1
            diff_feature = torch.abs(data-temp_x_noise)
            diff_feature = torch.matmul(A_raw, diff_feature)#矩阵相乘
            diff_feature = diff_feature.squeeze()#view(-1)
            diff_feature = diff_feature.mean()
        elif type_difinition=='abs_weight_max':#方案2
            diff_feature = torch.abs(data-temp_x_noise)
            diff_feature = torch.matmul(A_raw, diff_feature)#矩阵相乘
            diff_feature = diff_feature.squeeze()#view(-1)
            diff_feature = diff_feature.max()
        elif type_difinition=='mean_abs_weight':#方案3
            data_temp = torch.mean(data, dim=1)
            temp_x_noise = torch.mean(temp_x_noise, dim=1)
            diff_feature = torch.abs(data_temp-temp_x_noise).unsqueeze(1)
            diff_feature = torch.matmul(A_raw, diff_feature).squeeze()
        elif type_difinition == 'max_abs_weight':#方案4
            data_temp, _ = torch.max(data, dim=1)
            temp_x_noise, _ = torch.max(temp_x_noise, dim=1)
            diff_feature = torch.abs(data_temp-temp_x_noise).unsqueeze(1)
            diff_feature = torch.matmul(A_raw, diff_feature).squeeze()
        elif type_difinition == 'weight_abs_mean':#方案5
            data_temp = torch.matmul(A_raw, data)
            temp_x_noise = torch.matmul(A_raw, temp_x_noise)
            diff_feature = torch.abs(data_temp-temp_x_noise)
            diff_feature = diff_feature.mean()
        elif type_difinition == 'weight_abs_max':#方案0
            data_temp = torch.matmul(A_raw, data)
            temp_x_noise = torch.matmul(A_raw, temp_x_noise)
            diff_feature = torch.abs(data_temp-temp_x_noise)
            diff_feature = diff_feature.max()
        #diff_feature = torch.mean(diff_feature, 2).max(dim=1)[0]

        #diff_feature = torch.mean(diff_feature, 2).max(dim=1)[0]
        
        #diff_feature = torch.mean(diff_feature, 1)
        
        #noise_out_features = torch.mean(noise_out_features, 2)
        #diff_feature=torch.abs(noise_out_features-out_features).max(dim=1)[0]
        #diff_feature=torch.abs(noise_out_features-out_features).max(dim=1)[0]
        if num_j==0:
            out=diff_feature.detach().cpu().numpy().reshape(1,1)
        else:
            diff=diff_feature.detach().cpu().numpy().reshape(1,1)
            out=np.concatenate((out,diff),axis=1)
    
    return out


def diff_test_score(model, test_loader, num_classes, outf, out_flag, net_type, magnitude, temperature, num_output,\
                    magnitude_type,diff_type,batch_size,random_numbers, oodname='C16',layer_index=-1,threshold_gradient=0.5,wsi_level=False,dataset_name=''):
    model.eval()
    np.random.seed(0)
    if diff_type=='feature' and layer_index==-1:
        if out_flag == True:
            temp_file_name_test = '%s/diff/feature_diff_test_In.txt'%(outf)
        else:
            temp_file_name_test = '%s/diff/feature_diff_test_Out_%s.txt'%(outf,oodname)
    else:
        if out_flag == True:
            temp_file_name_test = '%s/diff/Probability_diff_test_In.txt'%(outf)
        else:
            temp_file_name_test = '%s/diff/Probability_diff_test_Out_%s.txt'%(outf,oodname)
    f = open(temp_file_name_test, 'w')
    t0=time.time()
    #对数据进行多次提取
    magnitude_list=[magnitude]
    item_d=-1
    item_test=0
    for batch_data in test_loader:

        if wsi_level == False:
            data, target = batch_data
        else:
            data, target, wsiname_item = batch_data
        item_d+=1
        if item_d in random_numbers and dataset_name!='imagenet':
            continue
        data = data.cuda()
        data.requires_grad = True
        if wsi_level == False:
            output=model(data)
        else:
            output, _, _, A_raw, _ = model(data)
        temp_output=output/temperature
        if diff_type=='softmax':
            soft_out = F.softmax(temp_output, dim=1)#8*2
            temp_diff_Outputs=diff_softmax_logit(data, model, soft_out, magnitude_type,net_type,magnitude_list,temperature,diff_type,threshold_gradient)
            if item_test==0:
                out=temp_diff_Outputs
                item_test+=1
            else:
                out=np.concatenate((out,temp_diff_Outputs),axis=0)
                item_test+=1
                #if item_test==3:
                #    break
        elif diff_type=='logit':
            soft_out=temp_output
            temp_diff_Outputs=diff_softmax_logit(data, model, soft_out, magnitude_type,net_type,magnitude_list,temperature,diff_type,threshold_gradient)
            if item_test==0:
                out=temp_diff_Outputs
                item_test+=1
                #break
            else:
                out=np.concatenate((out,temp_diff_Outputs),axis=0)
                item_test+=1
        elif diff_type=='feature':
            soft_out=temp_output
            if layer_index==-1:
                for i in range(num_output):
                    if wsi_level == False:
                        temp_diff_Outputs=diff_feature(data, model, soft_out, magnitude_type,net_type,magnitude_list,temperature,diff_type,i,threshold_gradient,wsi_level)
                    else:
                        temp_diff_Outputs=diff_wsi_feature(data, model, soft_out, magnitude_type,net_type,magnitude_list,temperature,diff_type,i,threshold_gradient,wsi_level,wsiname_item)
                    if i ==0:
                        outs=temp_diff_Outputs              
                    else:
                        outs=np.concatenate((outs,temp_diff_Outputs),axis=1)
                if item_test==0:
                    out=outs
                    #break
                else:
                    out=np.concatenate((out,outs),axis=0)
                    #if item_test==3:
                    #    break
                item_test+=1
            else:#单一特征
                if wsi_level == False:
                    temp_diff_Outputs=diff_feature(data, model, soft_out, magnitude_type,net_type,magnitude_list,temperature,diff_type,layer_index,threshold_gradient,wsi_level)
                else:
                    temp_diff_Outputs=diff_wsi_feature(data, model, soft_out, magnitude_type,net_type,magnitude_list,temperature,diff_type,layer_index,threshold_gradient,wsi_level,wsiname_item)
                if item_test==0:
                    out=temp_diff_Outputs
                    item_test+=1
                    #break
                else:
                    out=np.concatenate((out,temp_diff_Outputs),axis=0)
                    item_test+=1
                    #if item_test==5:
                    #    break
    print('threshold_gradient: %d, magnitude：%d，总共花费时间%.2f秒'%(threshold_gradient,magnitude,time.time()-t0))
    t0=time.time()
    if layer_index==-1 and diff_type=='feature':
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                f.write("{} ".format(out[i,j]))
            f.write('\n')
    else:
        for i in range(out.shape[0]):
            f.write("{}\n".format(out[i,0]))
    f.close()

def diff_val_score(model, test_loader, num_classes, outf, out_flag, net_type, magnitude_list, \
                   T_list,num_output,magnitude_type,diff_type,num_images,batch_size, oodname='C16',threshold_gradient=0.5,wsi_level=False,dataset_name=''):
    model.eval()
    
    np.random.seed(1)
    num_batch = num_images//batch_size
    #if diff_type=='feature':
    #    size_num==
    if wsi_level:
        random_numbers = np.random.choice(range(num_batch), size=15//batch_size, replace=False)
    elif dataset_name=='imagenet':
        random_numbers = [1]
    else:
        random_numbers = np.random.choice(range(num_batch), size=2000//batch_size, replace=False)
    if diff_type=='feature':
        if out_flag == True:
            temp_file_name_val = '%s/diff/feature_diff_val_In.txt'%(outf)
        else:
            temp_file_name_val = '%s/diff/feature_diff_val_Out_%s.txt'%(outf,oodname)
    else:
        if out_flag == True:
            temp_file_name_val = '%s/diff/Probability_diff_val_In.txt'%(outf)
        else:
            temp_file_name_val = '%s/diff/Probability_diff_val_Out_%s.txt'%(outf,oodname)
    path=os.path.join(outf,'diff')
    if not os.path.exists(path):
        os.makedirs(path)
    g = open(temp_file_name_val, 'w')
    #f = open(temp_file_name_test, 'w')
    t0=time.time()
    #对数据进行多次提取
    item_t=0
    if diff_type=='logit':
        T_list=[1]

    for temperature in T_list:
        
        item_d=-1
        item_val=0
        
        for batch_data in test_loader:
            #if item_d==2:
            #    break
            if wsi_level == False:
                data, target = batch_data
            else:
                data, target, wsiname_item = batch_data
            #print(data[0])
            item_d+=1
            if item_d not in random_numbers and dataset_name!='imagenet':
                continue
            data, target = data.cuda(), target.cuda()
            data.requires_grad = True
            if wsi_level == False:
                output=model(data)
            else:
                output, _, _, A_raw, _ = model(data)
            temp_output=output/temperature
            if diff_type=='softmax':
                soft_out = F.softmax(temp_output, dim=1)#8*2
                temp_diff_Outputs=diff_softmax_logit(data, model, soft_out, magnitude_type,net_type,magnitude_list,temperature,diff_type,threshold_gradient)
                if item_val==0:
                    out=temp_diff_Outputs  
                    item_val+=1   
                else:
                    out=np.concatenate((out,temp_diff_Outputs),axis=0)
                    item_val+=1
                    #if item_val==3:
                    #    break
            elif diff_type=='logit':
                soft_out=temp_output
                temp_diff_Outputs=diff_softmax_logit(data, model, soft_out, magnitude_type,net_type,magnitude_list,temperature,diff_type,threshold_gradient)
                if item_val==0:
                    out=temp_diff_Outputs
                    item_val+=1
                else:
                    out=np.concatenate((out,temp_diff_Outputs),axis=0)
                    item_val+=1
            elif diff_type=='feature':
                soft_out=temp_output
                for i in range(num_output):
                    if wsi_level == False:
                        temp_diff_Outputs=diff_feature(data, model, soft_out, magnitude_type,net_type,magnitude_list,temperature,diff_type,i,threshold_gradient,wsi_level)
                    else:
                        temp_diff_Outputs=diff_wsi_feature(data, model, soft_out, magnitude_type,net_type,magnitude_list,temperature,diff_type,i,
                                                           threshold_gradient,wsi_level,wsiname_item)
                    if i ==0:
                        feature_output=temp_diff_Outputs              
                    else:
                        feature_output=np.concatenate((feature_output,temp_diff_Outputs),axis=1)
                if item_val==0:
                    out=feature_output
                    #break
                else:
                    out=np.concatenate((out,feature_output),axis=0)
                    #if item_val==400:
                    #    break
                item_val+=1

        if item_t==0:
            outs=out
            item_t+=1
        else:
            outs=np.concatenate((outs,out),axis=1)
            item_t+=1
        print('threshold_gradient:%.3f, temperature：%.3f，总共花费时间%.2f秒'%(threshold_gradient,temperature,time.time()-t0))
        t0=time.time()
    for i in range(outs.shape[0]):
        for col in range(outs.shape[1]):
            g.write("{} ".format(outs[i,col]))
        g.write('\n')
    g.close()
    return random_numbers


def get_wsi_oodlabel(dir_name,oodname, M_list,T_list, num_output, stypes =['diff_val']):
    for stype in stypes:
        known = np.loadtxt('{}/feature_{}_In.txt'.format(dir_name, stype))#, delimiter='\\n')#output_path/Probability_val_In.txt
        novel = np.loadtxt('{}/feature_{}_Out_{}.txt'.format(dir_name, stype, oodname))#, delimiter='\\n')
    if known.ndim ==1:
        known=known.reshape(-1,1)
        novel=novel.reshape(-1,1)
    num_para=known.shape[1]
    num_lr=int(num_para/num_output)#需要进行多少次循环,68次
    multiplicand_list=[]
    #for num_i in range(num_para):
    for num_i in range(num_lr):#共有17*4次  表示第 次temperature, 第 次magnitude，
        temperation_index=num_i//len(M_list)#先magnitude再temperation
        magnitude_index=num_i%len(M_list)
        for num_layer in range(num_output):
            num_col=len(M_list)*num_output*temperation_index+len(M_list)*num_layer+magnitude_index
            if num_layer==0:
                data_known=known[:,num_col].reshape(-1,1)
                data_novel=novel[:,num_col].reshape(-1,1)
            else:
                data_known=np.concatenate((data_known,known[:,num_col].reshape(-1,1)),axis=1)
                data_novel=np.concatenate((data_novel,novel[:,num_col].reshape(-1,1)),axis=1)
                
        y = np.concatenate((np.ones(int(data_known.shape[0])), np.zeros(int(data_novel.shape[0]))))
        y = y.reshape((y.shape[0], 1))
        
        X_val_for_test = np.concatenate((data_known[:int(data_known.shape[0]),:], data_novel[:int(data_novel.shape[0]),:]))
        Y_val_for_test = y.reshape(-1)
        if num_i==0:
            y_single=X_val_for_test
        else:    
            y_single=np.concatenate((y_single,X_val_for_test),axis=1)
    l3 = open('%s/Probability_diff_val_In.txt'%dir_name, 'w')
    l4 = open('%s/Probability_diff_val_Out_%s.txt'%(dir_name,oodname), 'w')
    for i in range(Y_val_for_test.shape[0]):
        if Y_val_for_test[i] == 1:
            for j in range(y_single.shape[1]):
                l3.write("{} ".format(y_single[i][j]))#
            l3.write("\n")
        else:
            for j in range(y_single.shape[1]):
                l4.write("{} ".format(y_single[i][j]))#y_single本来就是负数
            l4.write("\n")
            
    l3.close()
    l4.close()
    val_results = openood_metric(dir_name,oodname, ['diff_val'])


    return val_results, multiplicand_list

def get_oodlabel(dir_name,oodname, M_list,T_list, num_output, stypes =['diff_val']):
    for stype in stypes:
        known = np.loadtxt('{}/feature_{}_In.txt'.format(dir_name, stype))#, delimiter='\\n')#output_path/Probability_val_In.txt
        novel = np.loadtxt('{}/feature_{}_Out_{}.txt'.format(dir_name, stype, oodname))#, delimiter='\\n')
    if known.ndim ==1:
        known=known.reshape(-1,1)
        novel=novel.reshape(-1,1)
    num_para=known.shape[1]
    num_lr=int(num_para/num_output)#需要进行多少次循环,68次
    lr_list=[]
    multiplicand_list=[]
    #for num_i in range(num_para):
    for num_i in range(num_lr):#共有17*4次  表示第 次temperature, 第 次magnitude，
        temperation_index=num_i//len(M_list)#先magnitude再temperation
        magnitude_index=num_i%len(M_list)
        for num_layer in range(num_output):
            num_col=len(M_list)*num_output*temperation_index+len(M_list)*num_layer+magnitude_index
            if num_layer==0:
                data_known=known[:,num_col].reshape(-1,1)
                data_novel=novel[:,num_col].reshape(-1,1)
            else:
                data_known=np.concatenate((data_known,known[:,num_col].reshape(-1,1)),axis=1)
                data_novel=np.concatenate((data_novel,novel[:,num_col].reshape(-1,1)),axis=1)
                
        #需要将data对半分。一半用于验证，一半用于测试
        #print(data_novel.shape[0]) 
        y = np.concatenate((np.ones(int(data_known.shape[0]/2)), np.zeros(int(data_novel.shape[0]/2))))
        y = y.reshape((y.shape[0], 1))
        X_train = np.concatenate((data_known[:int(data_known.shape[0]/2),:], data_novel[:int(data_novel.shape[0]/2),:]))
        if X_train[0,0]==0:
            multiplicand=10   #不同列的倍数是不同的，所以不是同常以一个数
        else:
            #multiplicand=(10**(-int(math.log10(-X_train[0,0]))+1))
            multiplicand=10
        multiplicand_list.append(multiplicand)
        #X_train*=multiplicand#原先的数据都太小了，三处修改 在diff_test_metric还有一处要一起修改
        #print(X_train[0,0])
        Y_train = y.reshape(-1)
        lr = LogisticRegressionCV(max_iter=10000,n_jobs=-1).fit(X_train, Y_train)#X是五维，就说明了是五维的特征。#max_iter=3000,
        lr_list.append(lr)
        X_val_for_test = np.concatenate((data_known[int(data_known.shape[0]/2):,:], data_novel[int(data_novel.shape[0]/2):,:]))
        #X_val_for_test *=multiplicand#原先的数据都太小了，三处修改 在diff_test_metric还有一处要一起修改
        y_tl = np.concatenate((np.ones(data_known.shape[0]-int(data_known.shape[0]/2)), np.zeros(data_novel.shape[0]-int(data_novel.shape[0]/2))))
        Y_val_for_test = y_tl.reshape(-1)
        X_val_simple=np.concatenate((data_known[:,:], data_novel[:,:]))
        Y_val_simple = np.concatenate((np.ones(int(data_known.shape[0])), np.zeros(int(data_novel.shape[0]))))
        Y_val_simple = Y_val_simple.reshape(-1)
        y_pred = lr.predict_proba(X_val_for_test)[:, 1]#属于第二个类的概率，即属于ID的概率
        #print(np.sum(y_pred[:int(y_pred.shape[0]/2)]),np.sum(y_pred[int(y_pred.shape[0]/2):]))
        y_test=lr.predict_proba(X_train)[:, 1]

        y_test=y_test.reshape(-1,1)
        y_pred = y_pred.reshape((y_pred.shape[0],1))
        
        if num_i==0:
            y_lr=y_pred#y_pred
            y_test_lr=y_test#测试用
            y_single=X_val_simple
        else:
            y_lr=np.concatenate((y_lr,y_pred),axis=1)#y_test
            y_test_lr=np.concatenate((y_test_lr,y_test),axis=1)#测试用
            y_single=np.concatenate((y_single,X_val_simple),axis=1)
    num_samples = y_lr.shape[0]
    l1 = open('%s/Probability_diff_val_In.txt'%dir_name, 'w')
    l2 = open('%s/Probability_diff_val_Out_%s.txt'%(dir_name,oodname), 'w')


    for i in range(num_samples):
        if Y_val_for_test[i] == 1:
            for j in range(y_lr.shape[1]):
                l1.write("{} ".format(y_lr[i][j]))#-y[i][j])#预测为正样本的数据会趋向于1
                #1-y_lr[i][j])表示属于ood的概率
            l1.write("\n")
        else:
            for j in range(y_lr.shape[1]):
                l2.write("{} ".format(y_lr[i][j]))#y[i][j])  
            l2.write("\n")   
    l1.close()
    l2.close()

    val_results = openood_metric(dir_name,oodname, ['diff_val'])
    #val_results = detection_performance(Y_val_for_test, y, output_path,out)
    l3 = open('%s/Probability_diff_val_simple_In.txt'%dir_name, 'w')
    l4 = open('%s/Probability_diff_val_simple_Out_%s.txt'%(dir_name,oodname), 'w')
    for i in range(len(Y_val_simple)):
        if Y_val_simple[i] == 1:
            for j in range(y_single.shape[1]):
                l3.write("{} ".format(y_single[i][j]))#
            l3.write("\n")
        else:
            for j in range(y_single.shape[1]):
                l4.write("{} ".format(y_single[i][j]))#y_single本来就是负数
            l4.write("\n")
            
    l3.close()
    l4.close()
    single_val_results = openood_metric(dir_name,oodname, ['diff_val_simple'])


    return val_results, single_val_results,lr_list,multiplicand_list






def diff_test_metric(dir_name, oodname, lr,multiplicand, stypes=['diff_test']):
    for stype in stypes:#'%s/diff/feature_diff_test_In
        known = np.loadtxt('{}/feature_{}_In.txt'.format(dir_name, stype))#, delimiter='\\n')#output_path/Probability_val_In.txt
        novel = np.loadtxt('{}/feature_{}_Out_{}.txt'.format(dir_name, stype, oodname))#, delimiter='\\n')
    y = np.concatenate((np.ones(known.shape[0]), np.zeros(novel.shape[0])))
    X_test=np.concatenate((known, novel), axis=0)
    #X_test*=multiplicand#这个有三次修改，在get_oodlabel函数还有两处要一起修改
    y_pred = lr.predict_proba(X_test)[:, 1]
    y_pred = y_pred.reshape((y_pred.shape[0],1))
    l1 = open('%s/Probability_feature_In.txt'%dir_name, 'w')
    l2 = open('%s/Probability_feature_Out_%s.txt'%(dir_name, oodname), 'w')
    for i in range(y_pred.shape[0]):
        if y[i]==1:
            l1.write("{}\n".format(y_pred[i][0]))#-y[i][j]) 属于0的概率
        else:
            l2.write("{}\n".format(y_pred[i][0]))#y_pred is 正数
            
    l1.close()
    l2.close()
    results = openood_metric(dir_name,oodname, ['feature'])
    return results






