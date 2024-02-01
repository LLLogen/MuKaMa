import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import sklearn.metrics as metrics
import argparse
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
import networkx as nx

from MuKaMa_light.SaDCN_convolution_block import SaDCNConv
import MuKaMa_SKI
import MuKaMa_OKI
from transformers import BertConfig, BertModel, BertTokenizer,RobertaTokenizer,RobertaModel
from dataset_MuKaMa import NuKaMaDataset,SpanMaskDataset
from torch.utils import data
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from apex import amp
from MuKaMa_OKI import query_wikidata_and_parse_results,enrich_with_tagme,construct_knowledge_graph

lm_mp = {'roberta': '/root/demo/ditto/ditto-master/roberta-base',
         'distilbert': 'distilbert-base-uncased'}


def parsers():
    parser = argparse.ArgumentParser(description="Bert model of argparse")
    parser.add_argument("--train_file", type=str, default=os.path.join("./data", "train.txt"))
    parser.add_argument("--dev_file", type=str, default=os.path.join("./data", "dev.txt"))
    parser.add_argument("--test_file", type=str, default=os.path.join("./data", "test.txt"))
    parser.add_argument("--classification", type=str, default=os.path.join("./data", "class.txt"))
    parser.add_argument("--bert_pred", type=str, default="/root/demo/text-similarity/simcse/roberta-base", help="roberta 预训练模型")
    parser.add_argument("--select_model_last", type=bool, default=True, help="选择模型 BertTextModel_last_layer")
    parser.add_argument("--class_num", type=int, default=2, help="分类数")
    parser.add_argument("--max_len", type=int, default=38, help="句子的最大长度")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learn_rate", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2, help="失活率")
    parser.add_argument("--filter_sizes", type=list, default=[2, 3, 4], help="TextCnn 的卷积核大小")
    parser.add_argument("--num_filters", type=int, default=2, help="TextCnn 的卷积输出")
    parser.add_argument("--encode_layer", type=int, default=12, help="roberta 层数")
    parser.add_argument("--hidden_size", type=int, default=768, help="bert 层输出维度")
    parser.add_argument("--save_model_best", type=str, default=os.path.join("model", "best_model.pth"))
    parser.add_argument("--save_model_last", type=str, default=os.path.join("model", "last_model.pth"))
    args = parser.parse_args()
    return args





class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=8, concat=True, dropout=0.6)
        self.conv3 = GATConv(hidden_channels * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, torch.zeros(data.num_nodes, dtype=int).to(x.device))
        return x

# model = GATModel(in_channels=768, hidden_channels=64, out_channels=768)
# graph_embedding = model(data)





class TextCnnModel(nn.Module):
    def __init__(self):
        super(TextCnnModel, self).__init__()
        self.num_filter_total = parsers().num_filters * len(parsers().filter_sizes)
        self.Weight = nn.Linear(self.num_filter_total, parsers().class_num, bias=False)
        self.bias = nn.Parameter(torch.ones([parsers().class_num]))
        self.filter_list = nn.ModuleList([
            nn.Conv2d(1, parsers().num_filters, kernel_size=(size, parsers().hidden_size)) for size in parsers().filter_sizes
        ])
    

    def forward(self, x):
        # x: [batch_size, 12, hidden]
        x = x.unsqueeze(1)  # [batch_size, channel=1, 12, hidden] = [16,1,12,768]

        pooled_outputs = []

        for i, conv in enumerate(self.filter_list):
            p = conv(x)
            out = F.relu(p) 

            maxPool = nn.MaxPool2d(kernel_size=(parsers().encode_layer - parsers().filter_sizes[i] + 1, 1))


           
            pooled = maxPool(out)
            pooled = maxPool(out).permute(0, 3, 2, 1) 
            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, len(parsers().filter_sizes))  
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filter_total])  

        output = self.Weight(h_pool_flat) + self.bias 

        return output


class MuKaMaModel(nn.Module):
    def __init__(self,device='cuda'):
        super(MuKaMaModel, self).__init__()
        self.bert = AutoModel.from_pretrained(parsers().bert_pred)
        self.device = device
        for param in self.bert.parameters():
            param.requires_grad = True
 
        self.linear = nn.Linear(parsers().hidden_size, parsers().class_num)
        self.textCnn = TextCnnModel()

        self.snakeCnn = SaDCNConv(in_ch=6,
                                out_ch=6,
                                kernel_size=15,#卷积核 15*15
                                extend_scope=1,
                                morph=0,
                                if_offset=True,
                                device=device)
                        
    def forward(self, x):
        x = x.to(self.device)
        outputs = self.bert(x,output_hidden_states=True)
        hidden_states = outputs.hidden_states 
        cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1)  
        for i in range(2, 13):
            cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)

        '''======================================================================================'''

        cls_embeddings = cls_embeddings.unsqueeze(1)
        cls_embeddings = cls_embeddings.expand(-1,6,-1,-1)

        '''======================================================================================'''
        pred = self.snakeCnn(cls_embeddings)
        return pred
def evaluate(model, iterator, threshold=None):
    """Evaluate a model on a validation/test dataset

    Args:
        model (DMModel): the EM model
        iterator (Iterator): the valid/test dataset iterator
        threshold (float, optional): the threshold on the 0-class

    Returns:
        float: the F1 score
        float (optional): if threshold is not provided, the threshold
            value that gives the optimal F1
    """
    all_p = []
    all_y = []
    all_probs = []
    '''============================================='''
    incorrect_samples = []
    incorrect_labels = []
    incorrect_predictions = []
    '''============================================='''
    with torch.no_grad():
        for batch in iterator:
            x, y = batch
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()

            if threshold is not None:
                pred = [1 if p > threshold else 0 for p in probs]
                incorrect_indices = np.where(np.array(pred) != np.array(y))[0]
                incorrect_samples.extend(x[incorrect_indices].cpu().numpy().tolist())
                incorrect_labels.extend(np.array(y)[incorrect_indices].tolist())
                incorrect_predictions.extend(np.array(pred)[incorrect_indices].tolist())

    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        f1 = metrics.f1_score(all_y, pred)
        return f1
    else:
        best_th = 0.5
        f1 = 0.0 # metrics.f1_score(all_y, all_p)

        for th in np.arange(0.0, 1.0, 0.05):
            pred = [1 if p > th else 0 for p in all_probs]
            new_f1 = metrics.f1_score(all_y, pred)
            if new_f1 > f1:
                f1 = new_f1
                best_th = th
        return f1, best_th

def train_step(train_iter, model, model_gat,optimizer, scheduler, hp):
    """Perform a single training step

    Args:
        train_iter (Iterator): the train data loader
        model (DMModel): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """
    criterion = nn.CrossEntropyLoss()
    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()

        if len(batch) == 2:
            x, y = batch
            prediction = model(x)
        else:
            x1, x2, y = batch
            prediction = model(x1, x2)

        loss1 = criterion(prediction, y.to(model.device))
        prediction2 = model_gat(x)
        loss2= prediction2.loss
        loss = loss1+loss2
        if hp.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 10 == 0: 
            print(f"step: {i}, loss: {loss.item()}")
        del loss




def train(trainset, validset, testset, run_tag, hp):
    """训练和评估模型

    Args:
        trainset (MuKaMaDataset): 训练集
        validset (MuKaMaDataset): 验证集
        testset (MuKaMaDataset): 测试集
        run_tag (str): 运行标签
        hp (Namespace): 超参数（例如，batch_size、学习率、fp16等）

    Returns:
        None
    """
    padder = trainset.pad 
    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=padder)  
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)  # 
    test_iter = data.DataLoader(dataset=testset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)  # 
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MuKaMaModel()  # 
    model = model.cuda()  # 
    optimizer = AdamW(model.parameters(), lr=hp.lr)  # 
    
    if hp.fp16:

        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')  # 
    num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)  # 

    writer = SummaryWriter(log_dir=hp.logdir)
    best_dev_f1 = best_test_f1 = 0.0
    for epoch in range(1, hp.n_epochs+1):
        # 训练阶段
        model.train()
        train_step(train_iter, model, optimizer, scheduler, hp)

        # 评估阶段
        model.eval()
        dev_f1, th = evaluate(model, valid_iter)  # 
        test_f1 = evaluate(model, test_iter, threshold=th)  # 

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_f1 = test_f1
            if hp.save_model:
                # 
                directory = os.path.join(hp.logdir, hp.task)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # 
                ckpt_path = os.path.join(hp.logdir, hp.task, 'model.pt')
                ckpt = {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch}
                torch.save(ckpt, ckpt_path)

        print(f"epoch {epoch}: dev_f1={dev_f1}, f1={test_f1}, best_f1={best_test_f1}")

        # 
        scalars = {'f1': dev_f1,
                   't_f1': test_f1}
        writer.add_scalars(run_tag, scalars, epoch)

    writer.close()  #

