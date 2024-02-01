import torch

from torch.utils import data
from transformers import AutoTokenizer
import sys
from augment import Augmenter
from MuKaMa_SKI import tokenize_and_insert_random_char
# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased',
         'bert': 'bert-base'
         }

def get_tokenizer(lm):
    if lm in lm_mp:
        return AutoTokenizer.from_pretrained(lm_mp[lm])
    else:
        return AutoTokenizer.from_pretrained(lm)

class MuKaMaDataset(data.Dataset):
    """EM dataset"""

    def __init__(
                 self,
                 path,#数据文件的路径
                 max_len=64,#序列的最大长度
                 size=None,#数据集的大小限制，默认为不限制
                 lm='roberta',#预训练语言模型
                 da=None#数据增强，默认为None
                 ):
        self.tokenizer = get_tokenizer(lm)#编码器
        self.pairs = []#文本句对列表
        self.labels = []#标签列表
        self.max_len = max_len#最大的文本序列长度，超过这个长度的文本会被截断
        self.size = size#限制数据集的大小，如果是none则不限制

        if isinstance(path, list):#如果传入的是列表，则使用这个列表作为数据行
            lines = path
        else:#否则假定path是文件路径，打开文件进行读取
            lines = open(path)

        for line in lines:
            s1, s2, label = line.strip().split('\t')#按照tab进行切分
            self.pairs.append((s1, s2))#一对句子，以元组的形式存入pairs列表
            self.labels.append(int(label))#对应的label以int形存入label列表

        self.pairs = self.pairs[:size]#根据数据集的大小限制size，截取前size个句子对和标签
        self.labels = self.labels[:size]
        self.da = da

        if da is not None:#如果数据增强不是None，则创建一个增强器
            self.augmenter = Augmenter()
        else:
            self.augmenter = None


    def __len__(self):
        """Return the size of the dataset.返回数据集的长度（有多少个句子对）"""
        return len(self.pairs)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the two entities
            List of int: token ID's of the two entities augmented (if da is set)
            int: the label of the pair (0: unmatch, 1: match)
        """
        
        text = self.texts[idx]
        inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        inputs_ids = inputs.input_ids.squeeze()
        span_start = torch.randint(1, self.max_length - 1, (1,)).item()  
        span_end = min(span_start + torch.randint(1, 5, (1,)).item(), self.max_length - 1)  
        inputs_ids[span_start:span_end] =  self.tokenizer.mask_token_id
        attention_mask = inputs.attention_mask.squeeze()

        left = self.pairs[idx][0]#
        right = self.pairs[idx][1]#


        x = self.tokenizer.encode(text=left,#
                                  text_pair=right,#
                                  max_length=self.max_len,
                                  truncation=True#
                                  )
        x = tokenize_and_insert_random_char(x)

        if self.da is not None:
            '''
            增强操作在augment.py中
            
            '''
            combined = self.augmenter.augment_sent(left + ' [SEP] ' + right, self.da)
            left, right = combined.split(' [SEP] ')
            x_aug = self.tokenizer.encode(text=left,
                                      text_pair=right,
                                      max_length=self.max_len,
                                      truncation=True)
            x_aug = tokenize_and_insert_random_char(x_aug)
            

            return  x_aug, self.labels[idx],inputs_ids,attention_mask
        
        
        else:
            return x, self.labels[idx]


    @staticmethod
    def pad(batch):
        """Merge a list of dataset items into a train/test batch
        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: x2 of shape (batch_size, seq_len).
                        Elements of x1 and x2 are padded to the same length
            LongTensor: a batch of labels, (batch_size,)
        """
        if len(batch[0]) == 3:
            x1, x2, y = zip(*batch)

            maxlen = max([len(x) for x in x1+x2])
            x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
            x2 = [xi + [0]*(maxlen - len(xi)) for xi in x2]
            return torch.LongTensor(x1), \
                   torch.LongTensor(x2), \
                   torch.LongTensor(y)
        else:
            x12, y = zip(*batch)
            maxlen = max([len(x) for x in x12])
            x12 = [xi + [0]*(maxlen - len(xi)) for xi in x12]
            return torch.LongTensor(x12), \
                   torch.LongTensor(y)

