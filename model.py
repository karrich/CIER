import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptEncoder(nn.Module):
    def __init__(self, user_num, item_num, tokenizer, hidden=1024, output_hidden=4096):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dropout = nn.Dropout(0.1)
        self.user_embedding = nn.Embedding(user_num, hidden)
        self.item_embedding = nn.Embedding(item_num, hidden)
        self.mlp_u = nn.Sequential(
            torch.nn.Linear(hidden, output_hidden)
        )
        self.mlp_v = nn.Sequential(
            torch.nn.Linear(hidden, output_hidden)
        )
        self.instruction = torch.tensor([tokenizer('Predict the rating for the given user and item, and generate a corresponding explanation or keyword.')['input_ids']])
        self.hard_prompt1 = torch.tensor([tokenizer('The rating given by user')['input_ids'][1:]])
        self.hard_prompt2 = torch.tensor([tokenizer('to item')['input_ids'][1:]])
        self.hard_prompt3 = torch.tensor([tokenizer('is ')['input_ids'][1:]])
        self.hard_prompt4 = torch.tensor([tokenizer('and the corresponding')['input_ids'][1:]])
        self.hard_prompt5 = torch.tensor([tokenizer('is "')['input_ids'][1:]])
        self.sub_full_words = torch.tensor(tokenizer('keyword explanation')['input_ids'][1:])
        self.verbalizer = tokenizer('12345')['input_ids'][2:]
        self.prompt_length = 4 + self.instruction.shape[1] + self.hard_prompt1.shape[1] + self.hard_prompt2.shape[1] + self.hard_prompt3.shape[1] + self.hard_prompt4.shape[1] + self.hard_prompt5.shape[1]
        self.rating_index = 1 + self.instruction.shape[1] + self.hard_prompt1.shape[1] + self.hard_prompt2.shape[1] + self.hard_prompt3.shape[1]
    def forward(self, user_id=None, item_id=None,
                rating=None,embed_tokens=None,curr_flag=None
               ):
        device = user_id.device
        user_embedding = self.mlp_u(self.user_embedding(user_id)).unsqueeze(1)
        item_embedding = self.mlp_v(self.item_embedding(item_id)).unsqueeze(1)
        if rating == None:
            return torch.cat([embed_tokens(self.instruction.to(device)).repeat(user_embedding.shape[0],1,1),
                              embed_tokens(self.hard_prompt1.to(device)).repeat(user_embedding.shape[0],1,1),
                              user_embedding,
                              embed_tokens(self.hard_prompt2.to(device)).repeat(user_embedding.shape[0],1,1),
                              item_embedding,
                              embed_tokens(self.hard_prompt3.to(device)).repeat(user_embedding.shape[0],1,1),
                              ],dim=-2)
        values = embed_tokens(torch.tensor(self.verbalizer).to(device)).unsqueeze(0).repeat(rating.shape[0],1,1)
        values = (rating.unsqueeze(-1) * values).sum(dim=1)
        if curr_flag == None:
            flag_words = self.sub_full_words.to(device)[[1]].repeat(user_embedding.shape[0],1)
        else:
            flag_words = self.sub_full_words.to(device)[curr_flag].unsqueeze(1)
        return torch.cat([embed_tokens(self.instruction.to(device)).repeat(user_embedding.shape[0],1,1),
                          embed_tokens(self.hard_prompt1.to(device)).repeat(user_embedding.shape[0],1,1),
                          user_embedding,
                          embed_tokens(self.hard_prompt2.to(device)).repeat(user_embedding.shape[0],1,1),
                          item_embedding,
                          embed_tokens(self.hard_prompt3.to(device)).repeat(user_embedding.shape[0],1,1),
                          values.unsqueeze(1),
                          embed_tokens(self.hard_prompt4.to(device)).repeat(user_embedding.shape[0],1,1),
                          embed_tokens(flag_words),
                          embed_tokens(self.hard_prompt5.to(device)).repeat(user_embedding.shape[0],1,1),
                          ],dim=-2)

class MyModel(nn.Module):
    def __init__(self, user_num, item_num,  hidden, llm_hidden, tokenizer):
        super(MyModel, self).__init__()
        self.prompt_encoder = PromptEncoder(user_num, item_num, tokenizer, hidden=hidden, output_hidden=llm_hidden)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.dropout = nn.Dropout(0.1)
        self.reset_parameters()
        self.model = None

        self.generate_weight = 1.0
        self.rating_weight= 0.1
        
        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def get_embedding(self, input_ids=None,  user_id=None, item_id=None,

                      rating=None,curr_flag=None):
        embeddings = self.model.base_model.model.model.embed_tokens
        
        if input_ids == None:
            return self.prompt_encoder(user_id=user_id,item_id=item_id,embed_tokens=embeddings)
        prompt = self.prompt_encoder(user_id=user_id,item_id=item_id,
                                     rating=rating,embed_tokens=embeddings,curr_flag=curr_flag
                                    )
        if input_ids.shape[1]==0:
            return prompt
        inputs_embeds = embeddings(input_ids)
        inputs_embeds = torch.cat([prompt,inputs_embeds],dim=-2)
        return inputs_embeds
    def forward(self, input_ids=None, user_id=None, item_id=None,
                rating=None, kv_cache=None
               ):
        #enocde
        if kv_cache == None:
            inputs_embeds = self.get_embedding(input_ids=input_ids, user_id=user_id, item_id=item_id, rating=rating)
            output = self.model(inputs_embeds=inputs_embeds)
        else:
            output = self.model(input_ids = input_ids, past_key_values = kv_cache)
        #decode
        kv_cache = output['past_key_values']
        logits = output['logits'][:,-1,:]
        logits = torch.softmax(logits,dim=1)
        
        return logits, kv_cache
    def rating_predict(self, user_id=None, item_id=None):
        
        inputs_embeds = self.get_embedding(user_id=user_id, item_id=item_id)
        logits = self.model(inputs_embeds=inputs_embeds)['logits']
        #decode
        output = logits[:,self.prompt_encoder.rating_index,:]
        output = output[:,self.prompt_encoder.verbalizer]
        output = torch.softmax(output,dim=1)
        
        return output
    def train_step(self, input_ids, user_id=None, item_id=None, rating=None, curr_flag=None,
                   rating_input=None
                  ):
        #enocde
        inputs_embeds = self.get_embedding(input_ids=input_ids, user_id=user_id, item_id=item_id,
                                              rating=rating_input,curr_flag=curr_flag
                                             )
        logits = self.model(inputs_embeds=inputs_embeds)['logits']
        output = logits[:,self.prompt_encoder.rating_index,:]
        output = output[:,self.prompt_encoder.verbalizer]
        loss = F.cross_entropy(output,rating)*self.rating_weight
        
        #MLM
        logits = logits[:,self.prompt_encoder.prompt_length-1:-1,:]
        targets = input_ids
        y_mask = input_ids.clone()
        y_mask[targets!=0] = 1
        y_mask = y_mask.reshape(-1)
        targets = targets.reshape(-1)
        logits = logits.reshape(-1,logits.shape[-1])
        generate_loss = (self.ce_loss(logits,targets) * y_mask).sum(dim=0) / (y_mask.sum(dim=0))
        loss += self.generate_weight*generate_loss

            
        return loss