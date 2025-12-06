import numpy as np
import torch
from utils.utils import *
import os
from dataset_modules.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB,CLAM_ensemble,CLAM_dropout
from models.model_transmil import TransMIL, TransMIL_dropout, TransMIL_ensemble
from models.model_abmil import ABMIL, ABMIL_dropout, ABMIL_ensemble
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from tqdm import tqdm
from topk.svm import SmoothTop1SVM
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report
import time
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim}
    method = None
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        if args.inst_loss == 'svm': 
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    elif args.model_type =='abmil':
        model = ABMIL(n_classes=args.n_classes,embed_dim=args.embed_dim)
    elif args.model_type == 'abmil_dropout':
        method = 'MCdropout'
        model = ABMIL_dropout(n_classes=args.n_classes,embed_dim=args.embed_dim,dropout = True)
    elif args.model_type == 'abmil_ensemble':
        method = 'M-heads'
        model = ABMIL_ensemble(n_classes=args.n_classes,embed_dim=args.embed_dim)
    elif args.model_type == 'transmil':
        model = TransMIL(n_classes=args.n_classes,embed_dim=args.embed_dim)
    elif args.model_type == 'transmil_dropout':
        method = 'MCdropout'
        model = TransMIL_dropout(n_classes=args.n_classes,embed_dim=args.embed_dim,dropout = 0.25)
    elif args.model_type == 'transmil_ensemble':
        method = 'M-heads'
        model = TransMIL_ensemble(n_classes=args.n_classes,embed_dim=args.embed_dim)
    elif args.model_type == 'clam_dropout':
        method = 'MCdropout'
        model = CLAM_dropout(n_classes=args.n_classes,embed_dim=args.embed_dim,dropout = 0.1)
    elif args.model_type == 'clam_ensemble':
        method = 'M-heads'
        model = CLAM_ensemble(n_classes=args.n_classes,embed_dim=args.embed_dim)
    else: # args.model_type == 'mil'
        raise NotImplementedError
    
    _ = model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 15, stop_epoch=35, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in (range(args.max_epochs)):
        if args.model_type in ['clam_sb', 'clam_mb','clam_dropout','clam_ensemble'] and not args.no_inst_cluster:     #使用聚类
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn,method = method)
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir,method = method)
        elif args.model_type in ['transmil','transmil_dropout','transmil_ensemble']:
            train_loop_transmil(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn,method = method)
            stop = validate_transmil(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir,method = method)
        else:#禁用实例级聚类
            t0 =time.time()
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn,method = method)
            stop = validate(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir,method = method)
            #print('一轮时间消耗',time.time()-t0)

        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _= summary(method,model, val_loader, args.n_classes,args.model_type)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger = summary(method, model, test_loader, args.n_classes,args.model_type)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 

def validate_transmil(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None,
                    results_dir=None,method = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in (enumerate(loader)):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            if data.shape[0]>10000:
                indices = torch.randperm(data.size(0))[:10000]
                data = data[indices]
            wsi_label = torch.zeros(n_classes)
            wsi_label[label.long()] = 1
            wsi_label = wsi_label.to(device)
            if method == 'M-heads':
                logits,logits2,logits3,logits4,logits5, Y_prob, Y_hat, _, instance_dict = model(data, label=label)
                Losses = []
                Y_hats = []
                for output in [logits,logits2,logits3,logits4,logits5]:
                    Losses.append(nn.CrossEntropyLoss()(output, label))  
                    Y_hats.append(torch.topk(output, 1, dim = 1)[1])
                Y_hats = torch.stack(Y_hats, dim=1)
                Y_hat, _ = torch.mode(Y_hats, dim=1)
                Y_hat = Y_hat.squeeze(0)
                min_loss = min(Losses)  # 获取最小的 loss
                min_loss_weight = 0.9
                other_loss_weight = (1 - min_loss_weight) / (len(Losses) - 1)  # 剩下的每个 loss 权重为 0.1 / 4
                loss = min_loss_weight * min_loss + sum(other_loss_weight * loss for loss in Losses if loss != min_loss)
                stacked_outputs = torch.stack([logits,logits2,logits3,logits4,logits5])
                # 对新维度求平均值，dim=0 表示沿着第一个维度（即列表中的张量）进行平均
                logits = stacked_outputs.mean(dim=0)
            else:
                logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label)
                
                loss = loss_fn(logits, label)



            acc_logger.log(Y_hat, label)
            
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    else:
        # prob[:,0] = 1-(prob[:,1:].sum(axis=1))
        # auc = roc_auc_score(labels, prob, multi_class='ovr')
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False      
def train_loop_transmil(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None,method = None):
    t0=time.time()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    print('\n')
    for batch_idx, (data, label) in (enumerate(loader)):
        data, label = data.to(device), label.to(device)
        wsi_label = torch.zeros(n_classes)
        wsi_label[label.long()] = 1
        wsi_label = wsi_label.to(device)
        #print(data.shape[0])
        if data.shape[0]>10000:
            indices = torch.randperm(data.size(0))[:10000]
            data = data[indices]

        if method == 'M-heads':
            logits,logits2,logits3,logits4,logits5, _, Y_hat, _, _ = model(data)
            Losses = []
            Y_hats = []
            for output in [logits,logits2,logits3,logits4,logits5]:
                Losses.append(nn.CrossEntropyLoss()(output, label))  
                Y_hats.append(torch.topk(output, 1, dim = 1)[1])
            Y_hats = torch.stack(Y_hats, dim=1)
            Y_hat, _ = torch.mode(Y_hats, dim=1)
            #print(Losses)
            min_loss = min(Losses)  # 获取最小的 loss
            min_loss_weight = 0.9
            other_loss_weight = (1 - min_loss_weight) / (len(Losses) - 1)  # 剩下的每个 loss 权重为 0.1 / 4
            loss = min_loss_weight * min_loss + sum(other_loss_weight * loss for loss in Losses if loss != min_loss)
            stacked_outputs = torch.stack([logits,logits2,logits3,logits4,logits5])
            # 对新维度求平均值，dim=0 表示沿着第一个维度（即列表中的张量）进行平均
            logits = stacked_outputs.mean(dim=0)
        else:
            logits, _, Y_hat, _, _ = model(data)
            
            loss = loss_fn(logits, label)
        acc_logger.log(Y_hat, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    


    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}, speed time: {:.2f}'.format(epoch, train_loss, train_error,time.time()-t0))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)




def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight,writer = None, loss_fn = None,method = None):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    #pbar = tqdm(desc=f'Epoch {epoch + 1}',postfix=dict,mininterval=0.3)
    for batch_idx, (data, label) in (enumerate(loader)):
        #if args.experiment_mask == 'RCC':
        #    if label ==0:
        #        continue
        data, label = data.to(device), label.to(device)#data N*1024


        if method == 'M-heads':
            logits,logits2,logits3,logits4,logits5, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            Losses = []
            Y_hats = []
            for output in [logits,logits2,logits3,logits4,logits5]:
                Losses.append(nn.CrossEntropyLoss()(output, label))  
                Y_hats.append(torch.topk(output, 1, dim = 1)[1])
            Y_hats = torch.stack(Y_hats, dim=1)
            Y_hat, _ = torch.mode(Y_hats, dim=1)
            Y_hat = Y_hat.squeeze(0)
            min_loss = min(Losses)  # 获取最小的 loss
            min_loss_weight = 0.9
            other_loss_weight = (1 - min_loss_weight) / (len(Losses) - 1)  # 剩下的每个 loss 权重为 0.1 / 4
            loss = min_loss_weight * min_loss + sum(other_loss_weight * loss for loss in Losses if loss != min_loss)
            stacked_outputs = torch.stack([logits,logits2,logits3,logits4,logits5])
            # 对新维度求平均值，dim=0 表示沿着第一个维度（即列表中的张量）进行平均
            logits = stacked_outputs.mean(dim=0)
        else:
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
            
            loss = loss_fn(logits, label)
        acc_logger.log(Y_hat, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)
    #pbar.close()



def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None,method = None):   
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        if method == 'M-heads':
            logits,logits2,logits3,logits4,logits5, _, Y_hat, _, _ = model(data)
            Losses = []
            Y_hats = []
            for output in [logits,logits2,logits3,logits4,logits5]:
                Losses.append(nn.CrossEntropyLoss()(output, label))  
                Y_hats.append(torch.topk(output, 1, dim = 1)[1])
            Y_hats = torch.stack(Y_hats, dim=1)
            Y_hat, _ = torch.mode(Y_hats, dim=1)
            min_loss = min(Losses)  # 获取最小的 loss
            min_loss_weight = 0.9
            other_loss_weight = (1 - min_loss_weight) / (len(Losses) - 1)  # 剩下的每个 loss 权重为 0.1 / 4
            loss = min_loss_weight * min_loss + sum(other_loss_weight * loss for loss in Losses if loss != min_loss)
            stacked_outputs = torch.stack([logits,logits2,logits3,logits4,logits5])
            # 对新维度求平均值，dim=0 表示沿着第一个维度（即列表中的张量）进行平均
            logits = stacked_outputs.mean(dim=0)
        else:
            logits, _, Y_hat, _, _ = model(data)
            
            loss = loss_fn(logits, label)

        acc_logger.log(Y_hat, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        #print("反向传播前:", torch.cuda.memory_allocated()/1024000)
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()
        #print("反向传播后:", torch.cuda.memory_allocated()/1024000)
    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None, method = None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            if method == 'M-heads':
                logits,logits2,logits3,logits4,logits5, _, Y_hat, _, _ = model(data)
                Losses = []
                Y_hats = []
                Y_prob = []
                for output in [logits,logits2,logits3,logits4,logits5]:
                    Losses.append(nn.CrossEntropyLoss()(output, label))  
                    Y_hats.append(torch.topk(output, 1, dim = 1)[1])
                    Y_prob.append(F.softmax(output, dim = 1))
                Y_hats = torch.stack(Y_hats, dim=1)
                Y_hat, _ = torch.mode(Y_hats, dim=1)
                Y_prob = torch.stack(Y_prob).mean(dim=0)
                min_loss = min(Losses)  # 获取最小的 loss
                min_loss_weight = 0.9
                other_loss_weight = (1 - min_loss_weight) / (len(Losses) - 1)  # 剩下的每个 loss 权重为 0.1 / 4
                loss = min_loss_weight * min_loss + sum(other_loss_weight * loss for loss in Losses if loss != min_loss)
                stacked_outputs = torch.stack([logits,logits2,logits3,logits4,logits5])
                # 对新维度求平均值，dim=0 表示沿着第一个维度（即列表中的张量）进行平均
                logits = stacked_outputs.mean(dim=0)
            else:
                logits, Y_prob, Y_hat, _, _ = model(data)
                loss = loss_fn(logits, label)
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            acc_logger.log(Y_hat, label)
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None,method = None):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.inference_mode():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)    
            if method == 'M-heads':
                logits,logits2,logits3,logits4,logits5, _, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
                Losses = []
                Y_hats = []
                Y_prob = []
                for output in [logits,logits2,logits3,logits4,logits5]:
                    Losses.append(nn.CrossEntropyLoss()(output, label))  
                    Y_hats.append(torch.topk(output, 1, dim = 1)[1])
                    Y_prob.append(F.softmax(output, dim = 1))
                Y_prob = torch.stack(Y_prob).mean(dim=0)
                Y_hats = torch.stack(Y_hats, dim=1)
                Y_hat, _ = torch.mode(Y_hats, dim=1)
                Y_hat = Y_hat.squeeze(0)
                min_loss = min(Losses)  # 获取最小的 loss
                min_loss_weight = 0.9
                other_loss_weight = (1 - min_loss_weight) / (len(Losses) - 1)  # 剩下的每个 loss 权重为 0.1 / 4
                loss = min_loss_weight * min_loss + sum(other_loss_weight * loss for loss in Losses if loss != min_loss)
                stacked_outputs = torch.stack([logits,logits2,logits3,logits4,logits5])
                # 对新维度求平均值，dim=0 表示沿着第一个维度（即列表中的张量）进行平均
                logits = stacked_outputs.mean(dim=0)
            else:
                logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
                
                loss = loss_fn(logits, label)




            acc_logger.log(Y_hat, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(method, model, loader, n_classes,model_type):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            if data.shape[0]>10000 and model_type in ['transmil','transmil_dropout','transmil_ensemble']:
                indices = torch.randperm(data.size(0))[:10000]
                data = data[indices]
            #if model_type == 'transmil':
            #    print(data.shape[0])
            if method == 'M-heads':
                _, _, _, _, _,Y_prob, Y_hat, _, _ = model(data)
            else:
                logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, acc_logger
