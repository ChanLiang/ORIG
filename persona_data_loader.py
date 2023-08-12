from torch.utils.data import DataLoader, Sampler, Dataset, RandomSampler, DistributedSampler
import json
# from lsp_model import GPT2Tokenizer
from transformers import GPT2Tokenizer
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
import random
import pickle
import os


def shuffle_persona(persona_list, persona_label):
    id_list = list(range(len(persona_list)))
    shuffled_id_list = list(range(len(persona_list)))
    random.shuffle(shuffled_id_list)

    n_list = [_ for _ in range(len(persona_list))]
    n_label = [-1] * len(persona_label)
    k = 0
    for i, j in zip(id_list, shuffled_id_list):
        n_list[j] = persona_list[i]
        if k < len(persona_label) and persona_label[k] == i:
            n_label[k] = j
            k += 1

    # for check
    # print ('==' * 20)
    # print (id_list)
    # print (shuffled_id_list)
    # print (persona_list, persona_label)
    # print (n_list, n_label)

    return n_list, n_label


def read_file(path, with_persona_label, shuffle=False, mode='label', all_seq_loss=False, single_turn=False, small_data=False, only_persona_response=False):
    END_OF_TEXT_TOKEN = '<|endoftext|>'
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2') # the smallest version of GPT-2, with 124M parameters.
    # tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-small")
    eos = tokenizer.encoder[END_OF_TEXT_TOKEN]

    examples = []
    
    pick_id = random.randint(1, 7)
    if not shuffle and os.path.exists(path.strip('output') + 'cached'):
        # return torch.load(path.strip('output') + 'cached.npy').tolist()
        return pickle.load(open(path.strip('output') + 'cached', "rb"))

    if shuffle and os.path.exists(path.strip('output') + f'shuffle{pick_id}'):
        # return torch.load(path.strip('output') + 'cached.npy').tolist()
        return pickle.load(open(path.strip('output') + f'shuffle{pick_id}', "rb"))
         
    with open(path, 'r', encoding='utf-8') as r:
        data = json.load(r)
    # [persona_list, history, response, persona_label] # response is not a list
    for i, (persona_list, history, response, persona_label) in tqdm(enumerate(data)):
    # for i, (persona_list, history, response, persona_label) in enumerate(data):
        if only_persona_response and persona_label[0] == -1:
            continue
        if shuffle:
            persona_list, persona_label = shuffle_persona(persona_list, persona_label)
        if single_turn:
            history = [history[-1]]
        response = [response]
        persona_label_entry = [persona_list[e] for e in persona_label if e > -1]
        persona_label_entry = [' '.join(persona_label_entry).strip()]
        persona_label = [str(e + 1) for e in persona_label] # idx + 1

        if with_persona_label: # add to response list
            if mode == 'entry' and persona_label_entry != ['']:
                # persona_label_entry[0] = persona_label_entry[0] + '\t'
                persona_label_entry[0] = persona_label_entry[0] 
                response = [persona_label_entry[0], response[0]] # persona entries + '\t' + response
                assert len(response) == 2
            # print (persona_list, history, response, persona_label, persona_label_entry)
            else:
                # label_str = ' '.join(persona_label).strip() + '\t'
                # label_str = ' '.join(persona_label).strip() + END_OF_TEXT_TOKEN # 直接加<eos> tokenizer不识别。。。
                label_str = ' '.join(persona_label).strip()
                response = [label_str, response[0]]
                assert len(response) == 2

        # tokenize
        persona_list = [tokenizer.encode(s) for s in persona_list]
        history = [tokenizer.encode(s) for s in history]
        response = [tokenizer.encode(s) for s in response]
        persona_label_entry = [tokenizer.encode(s) for s in persona_label_entry]
        persona_label = [tokenizer.encode(str(s)) for s in ' '.join(persona_label).strip()]
        # if not with_persona_label:
        #     persona_label = None

        # making input_ids, position_ids, lm_ids...
        example = make_example_inputs(i, persona_list, history, response, persona_label, persona_label_entry, eos, all_seq_loss=all_seq_loss)
        examples.append(example)

        if small_data and i >= 200:
            break

    if not shuffle and not os.path.exists(path.strip('output') + 'cached'):
        # torch.save(path.strip('output') + 'cached.npy', np.array(examples))
        pickle.dump(examples, open(path.strip('output') + 'cached', "wb"))

    if shuffle and not os.path.exists(path.strip('output') + f'shuffle{pick_id}'):
        # torch.save(path.strip('output') + 'cached.npy', np.array(examples))
        pickle.dump(examples, open(path.strip('output') + f'shuffle{pick_id}', "wb"))
    return examples


def make_example_inputs(id, personas, context, response, persona_label, persona_label_entry, eos, all_seq_loss=False):
    # print (personas , context , persona_label , persona_label_entry, response)
    
    # sents = None
    # if persona_label:
    #     sents = personas + context + persona_label_entry + response # 0 + 1 + 2 + 2
    # else:
    #     sents = personas + context + response
    sents = personas + context + response

    # 1. input_ids: 每个uttr加了eos，去掉了最后一位
    # print (sents)
    input_ids = [i for s in sents for i in s+[eos]][:-1]
    token_type_ids = []  # this becomes round ids
    lm_labels = []

    # 2. lm_labels: input_ids[1:] + [eos]
    #    token_type_ids: 0 for persona, 1 for context, 2 for persona_label and response
    for i, s in enumerate(sents):
        if i == 0:
            token_type_ids += [0] * len(s)
            lm_labels += [-1] * len(s) if not all_seq_loss else s[1:] + [eos]
        elif i < len(personas): # persona: 0
            token_type_ids += [0] * (len(s) + 1)
            lm_labels += [-1] * (len(s) + 1) if not all_seq_loss else s + [eos]
        elif i < len(personas) + len(context): # context: 1
            token_type_ids += [1] * (len(s) + 1)
            lm_labels += [-1] * (len(s) + 1) if not all_seq_loss else s + [eos]
        else: # persona_label/entry + '\t' + response: 2
            token_type_ids += [2] * (len(s) + 1)
            lm_labels += (s + [eos])

    # handle trailing -1's
    i = len(lm_labels) - 1
    while i >= 0:
        if lm_labels[i] != -1:
            break
        i -= 1
    input_ids = input_ids[:i+1]
    lm_labels = lm_labels[:i+1]
    token_type_ids = token_type_ids[:i+1]

    # pad to multiples of 8
    while len(input_ids) % 8 != 0:
        input_ids.append(0)
        token_type_ids.append(0)
        lm_labels.append(-1)
        
    # 3. position_ids
    position_ids = list(range(len(input_ids)))
    assert (len(input_ids) == len(position_ids) == len(token_type_ids)
            == len(lm_labels))
    assert len(input_ids) % 8 == 0

    # example = [id, input_ids, position_ids, token_type_ids,
                            # lm_labels]
    # print (all_seq_loss, input_ids, lm_labels)
    
    example = {
        'id': id, 
        'input_ids': input_ids, 
        'position_ids': position_ids,
        'token_type_ids': token_type_ids, 
        'lm_labels': lm_labels,
        'input_len': len(input_ids)
    }
    return example


class PersonaDataset(Dataset):
    """ pytorch dataset for GPT2 training """
    def __init__(self, path, max_len=None, with_persona_label=True, shuffle=False, all_seq_loss=False, single_turn=False, small_data=False, only_persona_response=False, **kwargs):
        self.example_ids = read_file(path, with_persona_label, shuffle, all_seq_loss=all_seq_loss, single_turn=single_turn, small_data=small_data, only_persona_response=only_persona_response)
        # print ('data_num = ', len(self.example_ids))
        self.max_len = max_len  # this max_len do truncate

    def __getitem__(self, i):
        return self.example_ids[i]

    def __len__(self):
        return len(self.example_ids)

    @staticmethod
    def collate(features):
        # print (features)
        input_ids = pad_sequence([torch.tensor(f['input_ids'], dtype=torch.long)
                                  for f in features],
                                 batch_first=True, padding_value=0)
        position_ids = pad_sequence([torch.tensor(f['position_ids'],
                                                  dtype=torch.long)
                                     for f in features],
                                    batch_first=True, padding_value=0)
        token_type_ids = pad_sequence([torch.tensor(f['token_type_ids'],
                                                    dtype=torch.long)
                                       for f in features],
                                      batch_first=True, padding_value=0)
        labels = pad_sequence([torch.tensor(f['lm_labels'], dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=-1)
        return (input_ids, position_ids, token_type_ids, labels)


# test
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# path = '/misc/kfdata01/kf_grp/lchen/D3/data_manipulation/data_distillation/predictions/test/output'
# path = '/misc/kfdata01/kf_grp/lchen/D3/data_manipulation/data_distillation/predictions/valid/output'
# path = '/misc/kfdata01/kf_grp/lchen/D3/data_manipulation/data_distillation/predictions/train/output'
# dataset = PersonaDataset(path, max_len=180, with_persona_label=False, shuffle=False)
# dataset = PersonaDataset(path, max_len=180, with_persona_label=False, shuffle=True)
# # dataset = PersonaDataset(path, max_len=256, with_persona_label=False, shuffle=False, single_turn=True)
# dataset = PersonaDataset(path, max_len=256, with_persona_label=False, shuffle=False, single_turn=True, only_persona_response=True)
# sampler = RandomSampler(dataset) if True else DistributedSampler(dataset)
# dataloader = DataLoader(dataset, sampler=sampler, batch_size=4, collate_fn=PersonaDataset.collate)


# for i, batch in enumerate(dataloader):
#     seq_len = batch[0].shape[1]
#     input_ids, position_ids, token_ids, label_ids, *_ = batch

#     if i > 5:
#         break
    
#     # visualize data
#     print ('=='*10 + ' visualize data ' + '=='*10)
#     print ('input_ids.shape, position_ids.shape, label_ids.shape = ', input_ids.shape, position_ids.shape, label_ids.shape) # torch.Size([4, 512]) torch.Size([4, 512])
#     print ('input_ids[0] = ', input_ids[0])
#     print ('position_ids[0] = ', position_ids[0])
#     print ('token_ids[0] = ', token_ids[0])
#     print ('label_ids[0] = ', label_ids[0])
#     # 'GPT2Tokenizer' object has no attribute 'batch_decode'???
#     # print (tokenizer.batch_decode(inputs[0], skip_special_tokens=True))
#     # print (tokenizer.batch_decode(labels[0], skip_special_tokens=True))

#     print ('input_ids = \n', tokenizer.decode(input_ids[0].tolist()))
#     # mask = token_ids.eq(2) | token_ids.eq(3)
#     mask = token_ids.eq(2)
#     print (mask.shape)
#     # print (mask[0])
#     print ('label_ids = \n', tokenizer.decode(label_ids[0][mask[0]].tolist()))
#     print ()
#     # break