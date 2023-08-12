import torch
import json
import os
import argparse
from tqdm import tqdm
import logging
from transformers import (
        GPT2Tokenizer,
        AutoModelForCausalLM
    )
import itertools

logger = logging.getLogger(__name__)

torch.manual_seed(42)

END_OF_TEXT_TOKEN = '<|endoftext|>'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
eos = tokenizer.encoder[END_OF_TEXT_TOKEN]

permutations = list(itertools.permutations([0,1,2,3,4]))


def get_normalized_seq_score(token_scores, output_ids, padding_id):
    '''
    Inplementation in beam_search:
    https://github.com/huggingface/transformers/blob/fe9152f67c61c9af4721fdc9abbc9578acf5f16f/src/transformers/generation/beam_search.py#L874
    
    Reference: 
    score = sum_logprobs / hyp.shape[-1]

    Args:
    token_scores: decode_len x [bz, |V|]
    output_ids: [bz, decode_len]
    padding_id: int

    Returns:
    seq_scores: [bz]
    '''
    token_scores = torch.stack(token_scores, dim=0).permute(1, 0, 2) # [bz, decode_len, |V|]
    token_probs = torch.nn.functional.log_softmax(token_scores, dim=-1) # log_softmax is faster and has better numerical properties
    # gather: https://discuss.pytorch.org/t/indexing-3d-tensor-using-2d-tensor/112011
    output_probs = torch.gather(token_probs, 2, output_ids[..., None]).squeeze() # [bz, decode_len] as desired
    padding_idx = torch.eq(output_ids, padding_id) # Computes element-wise equality
    output_probs[padding_idx] = 0
    # normalize seqence score by length
    decode_lens = torch.sum(torch.logical_not(padding_idx), dim=-1)
    seq_scores = torch.sum(output_probs, dim=-1) / decode_lens
    return seq_scores


def read_file(path, input_ground_truth_persona_label, tokenizer, device, order, mode='label', padding_to=128, permutaion_id=-1, input_assigned_persona_label=-1):

    examples = {} # raw text, ids 都放在里边了
    with open(f'{path}_{order}', 'r', encoding='utf-8') as r:
        sorted_data = json.load(r)

    # cross example
    input_assigned_persona_label = [str(e + 1) for e in input_assigned_persona_label] 
    
    for i, (persona_list, history, response, persona_label) in tqdm(enumerate(sorted_data)):
        if permutaion_id != -1: # -1 for normal order
            permutation = permutations[permutaion_id]
            n_persona = [persona_list[i] for i in permutation if i < len(persona_list)]
            persona_list = n_persona

        l_ = sum([len(s.split()) for s in persona_list + history]) + len(response.split())
        while l_ > 128:
            # print (l_, persona_list, history, response)
            history = history[1:]
            l_ = sum([len(s.split()) for s in persona_list + history]) + len(response.split())
            assert len(history) > 0

        examples['response'] = examples.get('response', []) # raw text
        examples['response'].append(response)
        examples['persona_list'] = examples.get('persona_list', []) # raw text
        examples['persona_list'].append(persona_list)
        examples['history'] = examples.get('history', []) # raw text
        examples['history'].append(history)
        examples['persona_label'] = examples.get('persona_label', []) # raw text
        examples['persona_label'].append(persona_label)

        persona_label_entry = [persona_list[e] for e in persona_label if e > -1]
        persona_label_entry = [' '.join(persona_label_entry).strip()]

        persona_label = [str(e + 1) for e in persona_label]

        if mode == 'entry' and persona_label_entry != ['']:
            persona_label_entry[0] = persona_label_entry[0] 
            persona_label_list = [persona_label_entry[0]]
        else: # this way
            truth_label_str = ' '.join(persona_label).strip()
            assign_label_str = ' '.join(input_assigned_persona_label).strip()
            persona_label_list = [truth_label_str]
            input_assigned_persona_label_list = [assign_label_str]

        persona_list = [tokenizer.encode(s) for s in persona_list]
        history = [tokenizer.encode(s) for s in history]
        persona_label_list = [tokenizer.encode(s) for s in persona_label_list]
        input_assigned_persona_label_list = [tokenizer.encode(s) for s in input_assigned_persona_label_list]

        example = make_example_inputs(i, persona_list, history, response, \
            persona_label_list, input_assigned_persona_label_list, eos, device, \
            input_ground_truth_persona_label=input_ground_truth_persona_label, \
            input_assigned_persona_label=input_assigned_persona_label, \
            padding_to=padding_to
        )

        for k in ['input_ids', 'position_ids', 'token_type_ids', 'attention_mask', 'input_len']:
            examples[k] = examples.get(k, [])
            examples[k].append(example[k])
        
    for k in ['input_ids', 'position_ids', 'token_type_ids', 'attention_mask', 'input_len']:
        # print (type(examples[k]), len(examples[k]), examples[k][0])
        # examples[k] = torch.tensor(examples[k]).to(device)
        examples[k] = torch.cat(examples[k], 0)
    return examples


def make_example_inputs(id, personas, context, response, persona_label, assigned_persona_label, eos, device, input_ground_truth_persona_label=False, input_assigned_persona_label=-1, padding_to=128):
    # print (personas , context , persona_label , response)
    sents = personas + context
    if input_ground_truth_persona_label:
        if input_assigned_persona_label == ['-1']:
            sents = personas + context + persona_label
        else:
            sents = personas + context + assigned_persona_label
    # 1. input_ids: 每个uttr加了eos，去掉了最后一位
    # print (sents)
    input_ids = [i for s in sents for i in s+[eos]]
    token_type_ids = []  # this becomes round ids

    # 2. lm_labels: input_ids[1:] + [eos]
    #    token_type_ids: 0 for persona, 1 for context, 2 for persona_label, 3 for response
    for i, s in enumerate(sents):
        if i == 0: # 注意到，第一个句子</s>的token_type_ids，其实已经是1了，属于第二个句子了
            token_type_ids += [0] * len(s)
        elif i < len(personas): # persona
            token_type_ids += [0] * (len(s) + 1)
        elif i < len(sents) - 1: # context
            token_type_ids += [1] * (len(s) + 1)
        elif not input_ground_truth_persona_label: # it's context
            token_type_ids += [1] * (len(s) + 1)
        else:                                      # it's persona_label
            token_type_ids += [2] * (len(s) + 1)

    token_type_ids += [2] # 最后一个位置统一加2 （eos）

    attention_mask = [1] * len(input_ids)
    # 3. position_ids
    position_ids = list(range(len(input_ids)))

    true_len = len(input_ids)

    # pad to 128
    assert len(input_ids) <= padding_to, (len(input_ids), padding_to)
    while len(input_ids) < padding_to:
        input_ids.insert(0, 0)
        token_type_ids.insert(0, 0)
        attention_mask.insert(0, 0)
        position_ids.insert(0, 0)
        
    
    assert (len(input_ids) == len(position_ids) == len(token_type_ids) == padding_to), (len(input_ids), len(position_ids), len(token_type_ids), padding_to)
    # assert len(input_ids) % 8 == 0

    # example = [id, input_ids, position_ids, token_type_ids,
                            # lm_labels]
    example = {
        'id': id, 
        'input_ids': torch.tensor(input_ids).view(-1, padding_to).to(device),  # [1, 128]
        'position_ids': torch.tensor(position_ids).view(-1, padding_to).to(device),
        'token_type_ids': torch.tensor(token_type_ids).view(-1, padding_to).to(device), 
        'attention_mask': torch.tensor(attention_mask).view(-1, padding_to).to(device),
        'input_len': torch.tensor(true_len).view(1, 1).to(device)
    }
    return example


def fix_state_dict_namespace(model_state_dict):
    old_keys = []
    new_keys = []
    for t in model_state_dict:
        new_key = t
        if t.startswith('module.'):
            new_key = t.replace('module.', '')
        old_keys.append(t)
        new_keys.append(new_key)

    for old_key, new_key in zip(old_keys, new_keys):
        model_state_dict[new_key] = model_state_dict.pop(old_key)

    model_state_dict['lm_head.weight'] = model_state_dict.pop('lm_head.decoder.weight')

    return model_state_dict


def load_model(model, checkpoint, device, verbose=False):
    if checkpoint is None or checkpoint == "None":
        if verbose:
            logger.info('no checkpoint provided for %s!' % model._get_name())
    else:
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        if verbose:
            logger.info('loading finetuned model from %s' % checkpoint)
        model_state_dict = torch.load(checkpoint)

        model_state_dict = fix_state_dict_namespace(model_state_dict)

        start_model = model
        if (hasattr(model, "transformer")
            and all(not s.startswith('transformer.')
                    for s in model_state_dict.keys())):
            logger.info('loading transfomer only')
            start_model = model.transformer
        start_model.load_state_dict(model_state_dict, strict=False)

    model.to(device)

    # print ('ending loading model')

    return model


def batch_generation(model_path, model_size, input_file, output_file, strategy='beam_search', debug=False, eos_in_decoding=False, input_ground_truth_label=False, human_view_file='', batch_size=8, order='pos_order', padding_to=128, max_len=200, min_length=1, device=0, permutaion_id=-1, input_assigned_persona_label=-1):

    # modify max length
    if '3' in order:
        padding_to += 50
    elif '10' in order:
        padding_to += 160
    max_len = max(max_len, padding_to + 40)

    print ('load tokenizer and model...')
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained(f"microsoft/DialoGPT-{model_size}")
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    model = load_model(AutoModelForCausalLM.from_pretrained(f"microsoft/DialoGPT-{model_size}"), model_path, device)

    print ('read file...')
    # print (eos_in_decoding, input_ground_truth_label)
    inputs = read_file(input_file, input_ground_truth_label, tokenizer, device, order, padding_to=padding_to, permutaion_id=permutaion_id, input_assigned_persona_label=input_assigned_persona_label)
    
    if not args.debug:
        rw = open(output_file + '_pred_response', 'w', encoding='utf-8')
        sw = open(output_file + '_seq_score', 'w', encoding='utf-8')
        lw = open(output_file + '_pred_label', 'w', encoding='utf-8') if eos_in_decoding else None
        hw = open(output_file + '_human_view', 'w', encoding='utf-8') if human_view_file else None
    else: # view sample by sample
        batch_size = 1

    print ('start to decode...')
    cnt = 0
    itr = tqdm(range(0, len(inputs['input_ids']), batch_size)) if args.bar and not args.debug else range(0, len(inputs['input_ids']), batch_size)
    for i in itr:
    # for i in range(0, len(inputs['input_ids']), batch_size): 
        if args.w_typeId:
            kwargs = {'token_type_ids':inputs["token_type_ids"][i: i + batch_size]}
        else:
            kwargs = {}

        # handle special case: joint decoding, <eos> in decoding sequence
        eos_token_id = 0 if eos_in_decoding else eos
        skip_special_tokens = False if eos_in_decoding else True

        # beam_search10
        if strategy == 'beam_search':
            output_dic = model.generate(
                input_ids=inputs["input_ids"][i: i + batch_size],
                attention_mask=inputs["attention_mask"][i: i + batch_size],
                num_beams=args.beam,
                # max_length=148,
                # max_length=200,
                # max_length=max_len,
                max_new_tokens=40,
                min_length=padding_to + min_length,
                eos_token_id=eos_token_id, 
                output_scores=True,
                return_dict_in_generate=True,
                **kwargs
            )
        else:
            # top10_top0.9_T0.9
            output_dic = model.generate(
                input_ids=inputs["input_ids"][i: i + batch_size],
                attention_mask=inputs["attention_mask"][i: i + batch_size],
                topk=10,
                topp=0.9,
                do_sampling=True,
                temperature=0.8,
                # max_length=146,
                max_length=max_len,
                eos_token_id=eos_token_id, # normal case
                output_scores=True,
                return_dict_in_generate=True,
                **kwargs
            )
        
        outputs = output_dic['sequences']
        token_scores = output_dic['scores']

        padding_id = eos # need to ensure...
        # seq_scores = get_normalized_seq_score(token_scores, outputs[:, padding_to:], padding_id)
        batch_out_sentence = tokenizer.batch_decode(outputs[:, padding_to:], skip_special_tokens=skip_special_tokens)
        
        # parse and save results...
        if not args.debug:
            if not eos_in_decoding:
                for hyp, ref in zip(batch_out_sentence, inputs['response'][i: i + batch_size]):
                    rw.write(hyp.strip() + '\n')
                # for score in seq_scores.cpu().tolist():
                #     sw.write(str(score) + '\n')
            else: # parse predicted labels and response
                for hyp, ref in zip(batch_out_sentence, inputs['response'][i: i + batch_size]):
                    line = hyp.strip()
                    label, resp = '', ''
                    if '<|endoftext|>' not in line:
                        label, resp = line[0], line[1:]
                    else:
                        parts = line.split('<|endoftext|>')
                        label, resp = parts[0], parts[1]
                    lw.write(label + '\n') # label
                    rw.write(resp + '\n') # response
                    # debug parsing step
                    # print ('==' * 20)
                    # print (line.strip())
                    # print (label)
                    # print (resp)

        # unit test: visualize input and output
        if debug:
            print ('==' * 50)
            print ('input = ', tokenizer.batch_decode(outputs[:, 0:padding_to], skip_special_tokens=False)[0])
            print ()
            print ('output = ', tokenizer.batch_decode(outputs[:, padding_to:], skip_special_tokens=False)[0])

        # generate txt files for human view
        if not args.debug and human_view_file:
            for (persona_list, history, response, persona_label, hyp) in zip(inputs['persona_list'][i: i + batch_size], inputs['history'][i: i + batch_size], inputs['response'][i: i + batch_size], inputs['persona_label'][i: i + batch_size], batch_out_sentence):
                hw.write('===' * 20 + '\n')
                hw.write(f'example_id: {cnt}\n')
                for j, persona in enumerate(persona_list):
                    hw.write(f'persona_{j + 1}: {persona.strip()}\n')
                hw.write('\n')
                context = ' <eou> '.join(history)
                hw.write(f'history: {context.strip()}\n')
                hw.write(f'ref_response: {response.strip()}\n')
                hw.write(f'hyp_response: {hyp.strip()}\n')
                cnt += 1


def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'
            

if __name__ == '__main__':
    # NOTE: default parameter for unit test
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='2022-11-06')
    # parser.add_argument("--exp_name", type=str, default='joint_decoding')
    parser.add_argument("--exp_name", type=str, default='joint_decoding_input_label')
    parser.add_argument("--model_size", type=str, default='small')
    parser.add_argument("--model_path", type=str, default='output/persona/DialoGPT-w_persona_label_eos_response_unshuffle-lr-1e-05-bz-32-time-2022-10-02142905/GP2-pretrain-step-7000.pkl')
    parser.add_argument("--input_file", type=str, default='./persona_data/sorted_test_files/test')
    # parser.add_argument("--input_file", type=str, default='./persona_data/train')
    # parser.add_argument("--output_file", type=str) # we generate it dynamically 
    parser.add_argument("--order", type=str, default='normal_ord')
    parser.add_argument("--eos_in_decoding", type=boolean_string, default=True)
    parser.add_argument("--input_ground_truth_label", type=boolean_string, default=False)
    parser.add_argument("--decoding_strategy", type=str, default='top10_top0.9_T0.9')
    parser.add_argument("--beam", type=int, default=5)
    # parser.add_argument("--max_seq_length", type=int, default=148)
    parser.add_argument("--max_seq_length", type=int, default=180)
    parser.add_argument("--min_decode_length", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--w_typeId", type=boolean_string, default=True)
    parser.add_argument("--debug", type=boolean_string, default=True)
    parser.add_argument("--bar", type=boolean_string, default=True)
    parser.add_argument("--human_view_file", type=boolean_string, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--permutaion_id", type=int, default=-1)
    parser.add_argument("--input_assigned_persona_label", type=str, default='-2') # -1 for dont use persona

    args = parser.parse_args()
    print (f'args = {args}\n')

    # generate output file name
    # base_dir = '../ACL23/Nov/decoding_results'
    base_dir = '../ACL23/Jan/decoding_results'
    ckp = args.model_path.strip().split('/')[-1].split('.')[0]
    strategy_name = f'beam_search_{args.beam}_minlen_{args.min_decode_length}' if args.decoding_strategy == 'beam_search' else 'top10_top0.9_T0.9'
    args.output_file = f'{base_dir}/{args.exp_name}_{ckp}_{args.order}_{strategy_name}_permutaion_id_{args.permutaion_id}_{args.date}' if args.input_assigned_persona_label == '-2' \
        else f'{base_dir}/{args.exp_name}_{ckp}_{args.order}_{strategy_name}_permutaion_id_{args.permutaion_id}_assigned_label_{args.input_assigned_persona_label.strip()}_{args.date}'
    args.input_assigned_persona_label = [int(e) for e in args.input_assigned_persona_label.strip().split('_')]

    assert args.order in ['normal_ord', 'pos_ord', 'neg_ord', 'lex_pos_ord', 'lex_neg_ord', 'pos_maj3', 'pos_maj10', 'neg_maj3', 'neg_maj10', 'single_pos', 'multi_pos'], (args.order)
    
    # decoding for single model, single order
    batch_generation(
        model_path=args.model_path, \
        model_size=args.model_size, \
        input_file=args.input_file, \
        output_file=args.output_file, \
        strategy=args.decoding_strategy, \
        debug=args.debug, \
        eos_in_decoding=args.eos_in_decoding, \
        input_ground_truth_label=args.input_ground_truth_label, \
        human_view_file=args.human_view_file, \
        batch_size=args.batch_size, \
        order=args.order, \
        max_len=args.max_seq_length, \
        min_length=args.min_decode_length, \
        permutaion_id=args.permutaion_id, \
        input_assigned_persona_label=args.input_assigned_persona_label, \
        padding_to=140,
        device=args.gpu
    )