import logging
from nlp import load_dataset
import numpy as np
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm


def get_dataset(name, tokenizer, split):     
    if name == 'mnli':
        dataset = load_dataset('glue','mnli', split=split)
    else:
        dataset = load_dataset(name, split=split)      

    input_ids = np.zeros(shape=(len(dataset),512))  
    token_type_ids = np.zeros(shape=(len(dataset),512)) 
    attention_mask = np.zeros(shape=(len(dataset),512))  
    answer = np.zeros(shape=(len(dataset)))
    # input_ids = []       
    # token_type_ids = []
    # attention_mask = []    
    # answer = []

    if name=='boolq':
        for i in range(len(dataset)):         
            tensor_features = tokenizer.__call__(dataset[i]['question'], dataset[i]['passage'], stride=128, return_tensors='np', max_length = 512,  padding='max_length', truncation=True,return_overflowing_tokens=True)          
            input_ids[i] = tensor_features['input_ids']                           
            token_type_ids[i] = tensor_features['token_type_ids']                      
            attention_mask[i] = tensor_features['attention_mask']
            # append越來越慢 https://hant-kb.kutu66.com/others/post_544244         
            

            if dataset[i]['answer']==True:             
                # answer.append(1) 
                answer[i] = 1        
            elif dataset[i]['answer']==False:             
                # answer.append(0)
                answer[i] = 0
            
            # if i == 1000:                 
            #     break
        input_ids = torch.LongTensor(input_ids)     
        token_type_ids = torch.LongTensor(token_type_ids)      
        attention_mask = torch.LongTensor(attention_mask)  
        answer = torch.LongTensor(answer) 

    elif name=='snli' or name=='mnli':
        # label 0 : entailment, label 1 : neural, label 2 : contradiction 
        for i in tqdm(range(len(dataset))):
            tensor_features = tokenizer.__call__(dataset[i]['premise'], dataset[i]['hypothesis'], return_tensors='np' , stride=128, max_length = 512,  padding='max_length', truncation=True,return_overflowing_tokens=True)           
            
            input_ids[i] = tensor_features['input_ids']                           
            token_type_ids[i] = tensor_features['token_type_ids']                      
            attention_mask[i] = tensor_features['attention_mask']

            if dataset[i]['label']==-1:
                answer[i] = 3
            else:
                answer[i] = dataset[i]['label']

            # if i == 1000:
            #     break


        input_ids = torch.LongTensor(input_ids)              
        token_type_ids = torch.LongTensor(token_type_ids)               
        attention_mask = torch.LongTensor(attention_mask)         
        answer = torch.LongTensor(answer)
     
        

    return TensorDataset(input_ids, token_type_ids, attention_mask, answer)

def compute_accuracy(y_pred, y_target):
    # 計算正確率
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100  

def model_setting(model_name, num_labels):
    if model_name=='bert':
        from transformers import AutoTokenizer, BertForSequenceClassification, BertConfig            
        config = BertConfig.from_pretrained("bert-base-uncased",num_labels = num_labels)              
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")              
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased",config=config) 
        return config, tokenizer, model
            
    
    elif model_name=='albert':
        from transformers import AutoTokenizer, AlbertForSequenceClassification, AlbertConfig   
        config = AlbertConfig.from_pretrained("albert-base-v2",num_labels = num_labels)     
        tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")     
        model = AlbertForSequenceClassification.from_pretrained("albert-base-v2",config=config)
        return config, tokenizer, model
    
    elif model_name=='bert-chinese':
        from transformers import AutoTokenizer, BertForSequenceClassification, BertConfig   
        config = BertConfig.from_pretrained("bert-base-chinese",num_labels = num_labels)     
        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")     
        model = BertForSequenceClassification.from_pretrained("bert-base-chinese",config=config)
        return config, tokenizer, model
    




    


































