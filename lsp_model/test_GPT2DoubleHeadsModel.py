import torch
# from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel
from transformers import GPT2Tokenizer
from gpt2 import GPT2DoubleHeadsModel

'''
debug log:
Some weights of GPT2DoubleHeadsModel were not initialized from the model checkpoint at microsoft/DialoGPT-medium and are newly initialized: ['multiple_choice_head.summary.bias', 'multiple_choice_head.summary.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
'''

tokenizer = GPT2Tokenizer.from_pretrained("../gpt2-base")
model = GPT2DoubleHeadsModel.from_pretrained("../gpt2-base")
# tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model = GPT2DoubleHeadsModel.from_pretrained("microsoft/DialoGPT-medium")

# Add a [CLS] to the vocabulary (we should train it also!)
num_added_tokens = tokenizer.add_special_tokens({"cls_token": "[CLS]"})
# Update the model embeddings with the new vocabulary size
embedding_layer = model.resize_token_embeddings(len(tokenizer))

choices = ["Hello, my dog is cute. [CLS] But it is big.", "Hello, my cat is cute. But it is small. [CLS]"]
encoded_choices = [tokenizer.encode(s) for s in choices]
cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]
print (encoded_choices) # 能分开50257!
print (cls_token_location) # [7, 12]

# input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
# mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1
input_ids = torch.tensor(encoded_choices)  # Batch size: 1, number of choices: 2
mc_token_ids = torch.tensor(cls_token_location)  # Batch size: 1
mc_label = torch.tensor([0, 1])

print (input_ids.shape) # torch.Size([2, 13])
print (mc_token_ids.shape) # torch.Size([2])
# outputs = model(input_ids, mc_token_ids=mc_token_ids)
# outputs = model(input_ids, mc_token_ids=mc_token_ids, mc_label=mc_label)
outputs = model(input_ids, mc_token_ids=mc_token_ids, mc_label=mc_label, output_attentions=True)
lm_logits = outputs.logits
mc_logits = outputs.mc_logits

print (outputs.keys())
print (lm_logits.shape) # torch.Size([2, 13, 50258])
print (mc_logits.shape) # torch.Size([2])
# <class 'tuple'> torch.Size([2, 16, 13, 13]) torch.Size([2, 16, 13, 13])
print (len(outputs.attentions), outputs.attentions[0].size(), outputs.attentions[11].size())
print(torch.eq(outputs.attentions[0], outputs.attentions[11])) # 右上三角True
# print (outputs.attentions[0][0][0])