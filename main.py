import torch
import random
import os
import torch.nn as nn
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from torch.optim import *
# from typing import Optional, Callable, Any, Tuple
from transformers import LlamaTokenizer, LlamaForCausalLM
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

from dataloader import *
from utils import *
from model import *
from peft import LoraConfig, get_peft_model, TaskType
from argparse import ArgumentParser

def seed_everything(seed=5254):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

def train_step(model, 
               train_dataloader,
               optimizer,
               device,
               epoch, 
               show_train_loss_steps,
               accumulation_steps,
               scaler,
               log_name
               ):
    model.train()
    loss_log = []
    for batch_idx, [input_ids, userid, itemid, rating, curr_flag,
                    rating_inputs
                   ] in enumerate(tqdm(train_dataloader)):
        input_ids = input_ids.to(device)
        itemid = itemid.to(device)
        userid = userid.to(device)
        rating = rating.to(device)
        curr_flag = curr_flag.to(device)
        rating_inputs = rating_inputs.to(device)
        with autocast():
            loss = model.train_step(input_ids, userid, itemid, rating, curr_flag,
                                    rating_inputs
                                   )
        loss_log.append(loss.item())
        loss = loss / accumulation_steps
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
          scaler.unscale_(optimizer)
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
          scaler.step(optimizer)
          scaler.update()
          model.zero_grad()
        if (batch_idx + 1) % show_train_loss_steps == 0:
            f = open(log_name,'a+')
            f.write("Train Epoch: {} [{}/{} ({}%)]\t Loss: {}\n".format(epoch,
                                            (batch_idx + 1) * input_ids.shape[0],
                                            len(train_dataloader.dataset),
                                            round(100. * batch_idx / len(train_dataloader), 2),
                                            round(sum(loss_log)/len(loss_log), 6)))
            print("Train Epoch: {} [{}/{} ({}%)]\t Loss: {}".format(epoch,
                                            (batch_idx + 1) * input_ids.shape[0],
                                            len(train_dataloader.dataset),
                                            round(100. * batch_idx / len(train_dataloader), 2),
                                            round(sum(loss_log)/len(loss_log), 6)))
            f.close()
            print("学习率：%f" % (optimizer.param_groups[0]['lr']))
            loss_log = []
    return    
    
def valid_step(model, valid_dataloader, device, log_name
               ):
    model.eval()
    loss_log = []
    for batch_idx, [input_ids, userid, itemid, rating, curr_flag,
                    rating_inputs
                   ] in enumerate(valid_dataloader):
        input_ids = input_ids.to(device)
        itemid = itemid.to(device)
        userid = userid.to(device)
        rating = rating.to(device)
        curr_flag = curr_flag.to(device)
        rating_inputs = rating_inputs.to(device)
        with torch.no_grad():
            with autocast():
                loss = model.train_step(input_ids,  userid, itemid, rating, curr_flag,
                                        rating_inputs )
        loss_log.append(loss.item())

    f = open(log_name,'a+')
    f.write("valid Loss: {}\n".format(round(sum(loss_log)/len(loss_log), 6)))
    print("valid Loss: {}".format( round(sum(loss_log)/len(loss_log), 6)))
    f.close()


    return round(sum(loss_log)/len(loss_log), 6)

def test_step(model, test_dataloader, device, log_name,
            dataset, output_dir, word, tokenizer):
    model.eval()
    predict = []
    label = []
    lens = len(test_dataloader)
    test_pred = []
    test_true = []
    for batch_idx, [input_ids, userid, itemid, rating, curr_flag,
                    rating_inputs
                   ] in enumerate(test_dataloader):
        print('\r',batch_idx,'/',lens,end='')
        input_ids = input_ids.to(device)
        itemid = itemid.to(device)
        userid = userid.to(device)
        rating = rating.to(device)
        curr_flag = curr_flag.to(device)
        
        rating_inputs = rating_inputs.to(device)
        
        text =  torch.tensor([[]]).to(device)
        last_words = torch.tensor([[]]).to(device)
        kv_cache = None
        for idx in range(word):
            with torch.no_grad():
                with autocast():
                    if idx == 0:
                        pre_rating = model.rating_predict(userid, itemid)
                        batch_true = rating.cpu()
                        batch_pred = pre_rating.detach().cpu().numpy()
                        for item in batch_pred:
                            test_pred.append((item*[1.,2.,3.,4.,5.]).sum().item())
                        for item in np.array(batch_true):
                            test_true.append(item+1)
                    logits, kv_cache = model(last_words, userid, itemid, pre_rating, kv_cache)
                    
            word_prob = logits.exp()
            last_words = torch.argmax(word_prob, dim=1).unsqueeze(1)
            if text.shape[1]==0:
                text = last_words
            else:
                text = torch.cat([text, last_words], 1)  
        predict.extend(text.tolist())
        label.extend(input_ids.tolist())
        
    tokens_predict = [ids2words(ids_clear(ids), tokenizer) for ids in predict]
    predict_text = []
    for row in tqdm(predict):
        temp = []
        for item in row:
            if item == 2:
                break
            temp.append(item)
        predict_text.append(temp)
    result = pd.DataFrame({"text":predict_text,"rating":test_pred})
    result.to_pickle(output_dir)    
        
    
    f = open(log_name,'a+')
    # rating
    predicted_rating = [(r, p) for (r, p) in zip(test_true, test_pred)]
    RMSE = root_mean_square_error(predicted_rating, 5, 1)
    f.write('RMSE {:7.4f}\n'.format(RMSE))
    MAE = mean_absolute_error(predicted_rating, 5, 1)
    f.write('MAE {:7.4f}\n'.format(MAE))
    # text
    tokens_test = [ids2words(ids_clear(ids), tokenizer) for ids in label]
    tokens_predict = [ids2words(ids_clear(ids), tokenizer) for ids in predict]
    BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
    f.write('BLEU-1 {:7.4f}\n'.format(BLEU1))
    BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
    f.write('BLEU-4 {:7.4f}\n'.format(BLEU4))
    USR, USN = unique_sentence_percent(tokens_predict)
    f.write('USR {:7.4f} | USN {:7}\n'.format(USR, USN))
    
    text_test = [' '.join(tokens) for tokens in tokens_test]
    text_predict = [' '.join(tokens) for tokens in tokens_predict]
    
    feature_set = dataset.feature_set
    feature_batch = feature_detect(tokens_predict, feature_set)
    DIV = feature_diversity(feature_batch)  # time-consuming
    f.write('DIV {:7.4f}\n'.format(DIV))
    FCR = feature_coverage_ratio(feature_batch, feature_set)
    f.write('FCR {:7.4f}\n'.format(FCR))
    FMR = feature_matching_ratio(feature_batch, dataset.features)
    f.write('FMR {:7.4f}\n'.format(FMR))
    
    text_test = [' '.join(tokens) for tokens in tokens_test]
    text_predict = [' '.join(tokens) for tokens in tokens_predict]
    
    ROUGE = rouge_score(text_test, text_predict) 
    for (k, v) in ROUGE.items():
        f.write('{} {:7.4f}\n'.format(k, v))
    f.close()
    return 
    
    
    
def main(args):
    if not os.path.exists(args.ckpt_dir + args.dataset_name):
        os.makedirs(args.ckpt_dir + args.dataset_name)
    if not os.path.exists(args.output_dir + args.dataset_name):
        os.makedirs(args.output_dir + args.dataset_name)
    if not os.path.exists(args.log_dir + args.dataset_name):
        os.makedirs(args.log_dir + args.dataset_name)
    seed_everything(args.seed)
    device = 0
    # device = 'cpu'
    if args.dataset_name =='Yelp':
        user_num = 27147
        item_num = 20266
    elif args.dataset_name == 'TripAdvisor':
        user_num = 9765
        item_num = 6280
    elif args.dataset_name == 'Amazon/MoviesAndTV':
        user_num = 7506
        item_num = 7360
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    
    if os.path.exists(args.data_dir+args.dataset_name+'/dataset_keywords.pickle') is False:
        dataset = pd.read_pickle(args.data_dir+args.dataset_name+'/reviews.pickle')
        dataset = pd.DataFrame(dataset)
        itemid = np.array(dataset['item'].tolist())
        userid = np.array(dataset['user'].tolist())
        encoder = LabelEncoder()  
        userid = encoder.fit_transform(userid).tolist()
        itemid = encoder.fit_transform(itemid).tolist()
        dataset['user'] = userid
        dataset['item'] = itemid

        keywords, keywords_words, text = [], [], []
        for row in tqdm(dataset['template']):
            keywords.append(tokenizer(row[0])['input_ids'][1:])
            keywords_words.append(row[0])
            text.append(tokenizer(row[2])['input_ids'][1:] + [2])
        dataset['text'] = text
        dataset['keyword'] = keywords
        dataset['keyword_words']=keywords_words
        dataset = dataset[['user','item','text','keyword','keyword_words','rating']]
        dataset.to_pickle(args.data_dir+args.dataset_name+'/dataset_keywords.pickle')
    else:
        dataset = pd.read_pickle(args.data_dir+args.dataset_name+'/dataset_keywords.pickle')
    dataset['rating'] = [int(x-1) for x in dataset['rating'].tolist()]
    
    
    for split_index in ['1','2','3','4','5']:
        train_dataset, valid_dataset, test_dataset = dataset_split(dataset,split_index,args)
        train_set = MyDataset(train_dataset)
        valid_set = MyDataset(valid_dataset)
        test_set = MyDataset(test_dataset)
        collate_train = MyCollater(args.epochs*len(train_dataset)//args.batch_size,args.word,args.delta)
        collate_valid = MyCollater(1,args.word)
        train_dataloader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=collate_train, shuffle=True, pin_memory=True, num_workers=1)
        valid_dataloader = DataLoader(valid_set, batch_size=args.batch_size, collate_fn=collate_valid, shuffle=False)
        test_dataloader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_valid, shuffle=False)

        # Define LoRA Config
        lora_config = LoraConfig(
         r=4,
         lora_alpha=32,
         target_modules=["q_proj", "k_proj"],
         lora_dropout=0.05,
         bias="none",
         task_type=TaskType.CAUSAL_LM
        )
        model_llm = LlamaForCausalLM.from_pretrained(
                    args.model_name, torch_dtype=torch.float16, device_map='cuda:'+str(device)
        )

        # add LoRA adaptor
        model_llm = get_peft_model(model_llm, lora_config)
        model_llm.print_trainable_parameters()

        model = MyModel(user_num, item_num,  args.id_hidden, model_llm.config.hidden_size, tokenizer).to(device)
        model.generate_weight = args.generate_weight
        model.rating_weight = args.rating_weight
        model.model = model_llm
        ###########################分层学习率####################################
        param = filter(lambda p: p.requires_grad==True, model.prompt_encoder.parameters())
        param2 = filter(lambda p: p.requires_grad==True, model.model.parameters())
        optimizer = AdamW([
             {'params':param,'lr':args.learning_rate},
            {'params':param2,'lr':args.learning_rate/10},
        ])
        ##############################################################
        scaler = GradScaler()
        log_name = args.log_dir + args.dataset_name+'/' + args.log_name
        output_dir = args.output_dir+ args.dataset_name+'/'+split_index+'generate.dataset'
        f = open(log_name,'a+')
        f.write(args.model_name+"\n")
        f.write("                                 split_index:" +split_index+"\n")
        f.close()
        best_loss = 999
        early_stop = 1

        if args.only_eval == False:
            for epoch in range(0, args.epochs):
                a = train_step(model, train_dataloader,
                               optimizer, device,
                               epoch, args.show_train_loss_steps,
                               args.accumulation_steps,
                               scaler,
                               log_name
                        )
                collate_train.cur_step = len(train_dataloader)*(epoch+1)
                valid_loss = valid_step(model, valid_dataloader, device, log_name)
    
                f = open(log_name,'a+')
                print(valid_loss)
                f.write(str(valid_loss)+'\n')
                if best_loss<valid_loss:
                    early_stop -= 1
                else:
                    print("save model\n")
                    f.write("save model\n")
                    best_loss = valid_loss
                    model.model.save_pretrained(args.ckpt_dir + args.dataset_name + '/'+split_index+'model')
                    torch.save(model.prompt_encoder,args.ckpt_dir + args.dataset_name + '/'+split_index+'ped.bin')
                f.close()
                if early_stop == 0:
                    break
        model.model.load_adapter(args.ckpt_dir + args.dataset_name + '/'+split_index+'model', 'best_lora')
        model.model.set_adapter("best_lora")
        model.prompt_encoder = torch.load(args.ckpt_dir + args.dataset_name + '/'+split_index+'ped.bin',map_location="cuda:"+str(device))
        test_step(model, test_dataloader, device, log_name, test_set, output_dir, args.word, tokenizer)
        # del model,model_head,model_llm
        torch.cuda.empty_cache()

   
        
if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--devices', default=-1, type=int)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', default=5254, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--accumulation_steps', default=1, type=int)
    parser.add_argument('--rating_weight', default=0.1, type=float)
    parser.add_argument('--generate_weight', default=1.0, type=float)
    parser.add_argument('--delta', default=0.3, type=float)
    parser.add_argument('--word', default=20, type=int)
    parser.add_argument('--show_train_loss_steps', default=500, type=int)
    parser.add_argument('--id_hidden', default=1024, type=int)
    parser.add_argument('--only_eval', action='store_true')

    parser.add_argument('--dataset_name', default='Amazon/MoviesAndTV', type=str)
    parser.add_argument('--data_dir', default='../data/', type=str)
    parser.add_argument('--model_name', default='openlm-research/open_llama_7b_v2', type=str)
    parser.add_argument('--ckpt_dir', default='./ckpt/', type=str)
    parser.add_argument('--log_dir', default='./log/', type=str)
    parser.add_argument('--log_name', default='llama.log', type=str)
    parser.add_argument('--output_dir', default='./output/', type=str)


    args = parser.parse_args()

    main(args)