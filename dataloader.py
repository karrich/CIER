import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def dataset_split(dataset,split_index,args):
    
    dataset['history'] =  list(range(len(dataset)))
    with open(args.data_dir+args.dataset_name+'/'+split_index+'/train.index', 'r') as f:
         train_index = [int(x) for x in f.readline().split(' ')]
    with open(args.data_dir+args.dataset_name+'/'+split_index+'/validation.index', 'r') as f:
         valid_index = [int(x) for x in f.readline().split(' ')]
    with open(args.data_dir+args.dataset_name+'/'+split_index+'/test.index', 'r') as f:
         test_index = [int(x) for x in f.readline().split(' ')]
    train_dataset = dataset.iloc[train_index].reset_index(drop=True)
    valid_dataset = dataset.iloc[valid_index].reset_index(drop=True)
    test_dataset = dataset.iloc[test_index].reset_index(drop=True)
    
    return train_dataset,valid_dataset,test_dataset


class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe
        self.feature_set = set(dataframe['keyword_words'])
        # self.feature_set.remove('')
        self.features = dataframe['keyword_words'].tolist()
    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        data = self.df.iloc[idx]

        return data
    
class MyCollater:
    def __init__(self, max_step = 1,  word = 20, delta = 0.5):
        self.max_step = max_step
        self.cur_step = 1
        self.word = word
        self.delta = delta

    def __call__(self, data):
        input_ids, userid, itemid, curr_flag, rating  = [], [], [], [], []
        rating_inputs = []
        max_length = max([min(self.word,len(x['text'])) for x in data])
        for x in data:

            if np.random.rand() < self.cur_step/self.max_step:
                ids = x['text'][:max_length]
                curr_flag.append(1)
            else:
                ids = x['keyword'][:max_length]
                curr_flag.append(0)

            target = ids + [0]*(max_length - len(ids))
            ids = ids + [0] * (max_length - len(ids))

            if  np.random.rand() < self.delta and x['rating']>0 and x['rating']<4:
                temp = [0.,0.,0.,0.,0.]
                rand = np.random.rand()*2/3
                temp[int(x['rating'])] = 1 - rand
                temp[int(x['rating'])-1] = rand/2
                temp[int(x['rating'])+1] = rand/2
                rating_inputs.append(temp)
            else:
                if x['rating']==0:
                    rating_inputs.append([1.,0.,0.,0.,0.])
                elif x['rating'] == 1:
                    rating_inputs.append([0.,1.,0.,0.,0.])
                elif x['rating'] == 2:
                    rating_inputs.append([0.,0.,1.,0.,0.])
                elif x['rating'] == 3:
                    rating_inputs.append([0.,0.,0.,1.,0.])
                elif x['rating'] == 4:
                    rating_inputs.append([0.,0.,0.,0.,1.])

            input_ids.append(ids)

            userid.append(x['user'])
            itemid.append(x['item'])
            rating.append(x['rating'])
        self.cur_step += 1

        input_ids = torch.tensor(input_ids)
        userid = torch.tensor(userid)
        itemid = torch.tensor(itemid)
        curr_flag = torch.tensor(curr_flag)
        rating = torch.tensor(rating).long()

        rating_inputs = torch.tensor(rating_inputs)

        return input_ids, userid, itemid, rating, curr_flag, rating_inputs
