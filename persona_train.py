#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
'''
 * @Desc: fine tuning GPT2
          Modified based on Huggingface GPT-2 implementation
'''

import json
import os
import sys
import argparse
import logging
import time
import tqdm
import datetime
import torch
import numpy as np
from os.path import join
from torch.distributed import get_rank, get_world_size
from lsp_model import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, Adam
from transformers import GPT2Tokenizer
from gpt2_training.train_utils import load_model, boolean_string, set_lr, get_eval_list_same_length
from gpt2_training.eval_utils import eval_model_loss
# from data_loader import BucketingDataLoader, DynamicBatchingLoader, DistributedBucketingDataLoader
from persona_data_loader import PersonaDataset
from torch.utils.data import DataLoader, Sampler, Dataset, RandomSampler, DistributedSampler
from gpt2_training.distributed import all_reduce_and_rescale_tensors, all_gather_list

#########################################################################
# logging, random_seed
##########################################################################

INF = 100000000
CACHE_EMPTY_STEP = 1000
SEED=42

def init():
    np.random.seed()
    torch.random.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    if torch.cuda.device_count()> 0:
        torch.cuda.manual_seed_all(SEED)

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

logger = init()

#########################################################################
# 0. Parsing arguments
##########################################################################
def process_arguments():
    parser = argparse.ArgumentParser()

    # normal arguments
    parser.add_argument('--model_name_or_path', type=str,
                        help='pretrained model name or path to local checkpoint')
    parser.add_argument("--max_seq_length", type=int, default=128)

    parser.add_argument("--skip_eval", action='store_true',
                        help='If true, skip evaluation.')
    parser.add_argument("--init_checkpoint", type=str)
    parser.add_argument("--train_input_file", type=str)
    parser.add_argument("--eval_input_file", type=str)
    parser.add_argument("--test_input_file", type=str)
    parser.add_argument("--continue_from", type=int, default=0)

    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="batch size now means per GPU per step")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="to increase effective batch size "
                            "and reduce synchronization")
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_optim_steps", type=int, default=1000000,
                        help="new API specifies num update steps")
    parser.add_argument("--num_epoch", type=int, default=10,
                        help="new API specifies num update steps")
    parser.add_argument("--valid_step", type=int, default=10000,
                        help="how many optim steps between validations")
    parser.add_argument("--test_step", type=int, default=10000,
                        help="how many optim steps between testings")
    parser.add_argument("--log_step", type=int, default=50,
                        help="how many steps between logs")
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=2000)

    parser.add_argument("--normalize_data", type=boolean_string, default=True)
    parser.add_argument("--fp16", type=boolean_string, default=False)
    parser.add_argument("--lr_schedule", type=str,
                        choices=['noam', 'noamwd', 'BERT', 'None'], default='noam')
    parser.add_argument("--loss_scale", type=float, default=0)

    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument('--pbar', type=boolean_string, default=False, help='turn on progress bar')

    # distributed training
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='for torch.distributed')

    parser.add_argument('--exp_name', type=str, default='')

    parser.add_argument('--visualize_train_data', type=boolean_string, default=False)

    parser.add_argument('--with_persona_label', type=boolean_string, default=True)

    parser.add_argument('--shuffle', type=boolean_string, default=True)

    parser.add_argument("--no_token_id", type=boolean_string, default=False)

    parser.add_argument("--all_seq_loss", type=boolean_string, default=False)

    parser.add_argument("--single_turn", type=boolean_string, default=False)

    parser.add_argument("--new_type_ids", type=boolean_string, default=False)

    parser.add_argument("--small_data", type=boolean_string, default=False)

    parser.add_argument("--only_persona_response", type=boolean_string, default=False)

    # do normal arguments parsing
    args = parser.parse_args()

    # single gpu or distributed training
    if args.local_rank == -1:
        logger.info('CUDA available? {}'.format(str(torch.cuda.is_available())))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        args.device, args.n_gpu = device, n_gpu
    else:
        # distributed training
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of
        # sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        n_gpu = torch.distributed.get_world_size()
        args.device, args.n_gpu = device, 1
        logger.info("device: {} n_gpu: {}, distributed training: {}, "
                    "16-bits training: {}".format(
                        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    # batch size accummulation
    assert args.train_batch_size % args.gradient_accumulation_steps == 0, \
        "batch size % gradient accumulation steps != 0!"
    logger.info('Train batch size per gpu = {}'.format(args.train_batch_size))
    logger.info('Total train batch size = {}'.format(args.train_batch_size * n_gpu))
    # actual bz per gpu per forward pass
    args.train_batch_size = (args.train_batch_size // args.gradient_accumulation_steps)

    # print argument information
    logger.info('Input Argument Information')
    args_dict = vars(args)
    for a in args_dict:
        logger.info('%-28s  %s' % (a, args_dict[a]))

    return args

def manage_model_and_log_dir(args):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
    output_dir = join(args.output_dir, '{}-lr-{}-bz-{}-time-{}'.format(
        args.exp_name, args.learning_rate,
        args.train_batch_size * args.gradient_accumulation_steps * args.n_gpu,
        timestamp)
    )
    log_dir = args.log_dir if args.log_dir is not None and len(args.log_dir) > 0 else output_dir
    if args.local_rank == -1 or get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)

    train_logger, eval_logger = None, None
    if args.local_rank == -1 or get_rank() == 0 and not args.visualize_train_data:
        suffix = '{}-lr-{}-bz-{}-time-{}'.format(args.exp_name, args.learning_rate,
                                                args.train_batch_size * args.gradient_accumulation_steps * args.n_gpu,
                                                timestamp)
        train_logger = open(join(log_dir, f'train_log_{suffix}.txt'), 'a+', buffering=1)
        eval_logger = open(join(log_dir, f'eval_log_{suffix}.txt'), 'a+', buffering=1)
    
    return output_dir, train_logger, eval_logger

args = process_arguments()
output_dir, train_logger, eval_logger = manage_model_and_log_dir(args)

#########################################################################
# 1. Prepare Dataset, DataLoader
##########################################################################
def get_dataloader(args):
    from persona_data_loader import PersonaDataset
    if args.new_type_ids:
        print ('use 3d ids (position & type ids)...')
        from persona_data_loader_3D import PersonaDataset

    train_dataset = PersonaDataset(args.train_input_file, max_len=args.max_seq_length, with_persona_label=args.with_persona_label, shuffle=False, all_seq_loss=args.all_seq_loss, single_turn=args.single_turn, small_data=args.small_data, only_persona_response=args.only_persona_response)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=PersonaDataset.collate)

    eval_dataset = PersonaDataset(args.eval_input_file, max_len=args.max_seq_length, with_persona_label=args.with_persona_label, shuffle=False, all_seq_loss=args.all_seq_loss, single_turn=args.single_turn, small_data=args.small_data, only_persona_response=args.only_persona_response)
    eval_sampler = RandomSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader_loss = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=PersonaDataset.collate)

    test_dataset = PersonaDataset(args.test_input_file, max_len=args.max_seq_length, with_persona_label=args.with_persona_label, shuffle=False, all_seq_loss=args.all_seq_loss, single_turn=args.single_turn, small_data=args.small_data, only_persona_response=args.only_persona_response)
    test_sampler = RandomSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader_loss = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, collate_fn=PersonaDataset.collate)
    return train_dataloader, eval_dataloader_loss, test_dataloader_loss

train_dataloader, eval_dataloader_loss, test_dataloader_loss = get_dataloader(args)

#########################################################################
# 2. Prepare Model and Optimizer
##########################################################################
def modify_tokenizer(tokenizer):
    additional_special_tokens = ['<info_bos>', '<talker1_bos>', '<talker2_bos>']
    tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})

    tokenizer.info_bos_id = tokenizer.added_tokens_encoder['<info_bos>']                                    
    tokenizer.talker1_bos_id = tokenizer.added_tokens_encoder['<talker1_bos>']                                                          
    tokenizer.talker2_bos_id = tokenizer.added_tokens_encoder['<talker2_bos>']

    return tokenizer, len(additional_special_tokens)

def modify_model(model, tokenizer):
    '''Modify the model to make it fit the data'''
    tokenizer, additional_length = modify_tokenizer(tokenizer)
    model.embeddings_size = 768
    model.n_embeddings = len(tokenizer)
   
    # 处理新增加的embedding
    model_embedding_weight = model.transformer.wte.weight
    model.transformer.wte = torch.nn.Embedding(model.n_embeddings, model.embeddings_size)
    model.lm_head.decoder = torch.nn.Linear(model.embeddings_size, model.n_embeddings, bias=False)
    model.transformer.wte.weight.data[:-additional_length, :] = model_embedding_weight.data
    model.transformer.wte.weight.data[-additional_length:, :] = 0 # 特殊token的embedding
    # 改名，和ckp对应
    model.lm_head.decoder.weight = model.transformer.wte.weight 

def get_model_and_optimizer(args):
    # enc = GPT2Tokenizer.from_pretrained('microsoft/DialoGPT-small')
    # model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-small")

    enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    config = GPT2Config.from_json_file(join(args.model_name_or_path, 'config.json'))
    model = load_model(GPT2LMHeadModel(config), args.init_checkpoint, args, verbose=True)
    if args.new_type_ids:
        modify_model(model, enc)
    print ('model = ', model)
    model.to(args.device)
    if args.n_gpu > 1:
        logging.info('data parallel because more than one gpu')
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        # when from scratch make sure initial models are the same
        params = [p.data for p in model.parameters()]
        all_reduce_and_rescale_tensors(
            params, float(torch.distributed.get_world_size()))

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info('Number of parameter = {}'.format(total_params))

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'ln']   # no decay for bias and LayerNorm (ln)
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # it's not for fp16
    optimizer = Adam(optimizer_grouped_parameters, args.learning_rate,
                        max_grad_norm=1.0)
    
    return config, enc, model, optimizer

config, enc, model, optimizer = get_model_and_optimizer(args)

#########################################################################
# 3. Training/evaluation/testing
##########################################################################
global_step = 0
step = 0
epoch = 0

if args.continue_from:
    global_step = args.continue_from
    step = global_step * 2 - 1

if args.local_rank != -1:
    args.n_gpu = 1

if args.local_rank == -1 or get_rank() == 0:
    if args.pbar:
        pbar = tqdm.tqdm(total=args.num_optim_steps, desc=f"training")
    else:
        pbar = None

def visualize_train_data(batch, tokenizer, args):
    # visualize data
    input_ids, position_ids, token_ids, label_ids, *_ = batch
    print ('=='*10 + ' visualize data ' + '=='*10)
    print ('input_ids.shape, position_ids.shape, label_ids.shape = ', input_ids.shape, position_ids.shape, label_ids.shape) # torch.Size([4, 512]) torch.Size([4, 512])
    print ('input_ids[0] = ', input_ids[0])
    print ('position_ids[0] = ', position_ids[0])
    print ('token_ids[0] = ', token_ids[0])
    print ('label_ids[0] = ', label_ids[0])
    # 'GPT2Tokenizer' object has no attribute 'batch_decode'???
    # print (tokenizer.batch_decode(inputs[0], skip_special_tokens=True))
    # print (tokenizer.batch_decode(labels[0], skip_special_tokens=True))

    print ('input_ids[0] = \n', tokenizer.decode(input_ids[0].tolist()))
    mask = ~label_ids.eq(-1)
    print (mask.shape)
    # print (mask[0])
    print ('input_ids[0] = \n', tokenizer.decode(input_ids[0][mask[0]].tolist()))
    print ('label_ids[0] = \n', tokenizer.decode(label_ids[0][mask[0]].tolist()))
    print ('position_ids[0] = \n', position_ids[0][mask[0]].tolist())
    print ('token_ids[0] = \n', token_ids[0][mask[0]].tolist())
    print ()

def my_eval_model_loss(model, tokenizer, eval_dataloader, epoch_id, global_step, args, is_test=False):
    # use the same signature with eval_model_generation
    logger.info('compute eval model loss, using eval mode, '
                'please change it back to train after calling this function')
    model.eval()
    tot_loss = []
    tot_ppl = []
    # tot_sample = []
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            # [bz, T], src_len = [bz]
            input_ids, position_ids, token_ids, label_ids = batch

            if args.no_token_id:
                token_ids = None
            loss, ppl, _ = model(input_ids, position_ids, token_ids, label_ids)

            if args.n_gpu > 1:
                loss = loss.mean()
                ppl = ppl.mean()
            loss = loss / (args.eval_batch_size / input_ids.shape[0])
            ppl = ppl / (args.eval_batch_size / input_ids.shape[0])
            tot_loss.append(float(loss.item()) * (args.eval_batch_size / input_ids.shape[0]))
            tot_ppl.append(float(ppl.item()) * (args.eval_batch_size / input_ids.shape[0]))

            # if step % 500 == 0:
            #     print (step)

    if args.local_rank == -1 or get_rank() == 0:
        loss = np.sum(tot_loss) / len(tot_loss)
        ppl = np.sum(tot_ppl) / len(tot_ppl)
        
        mode = 'Test' if is_test else 'Valid'
        print(
            f"\n Epoch {epoch_id} Step {global_step}: {mode}_loss {loss:.2f} "
            f"ppl {ppl:.2f} ", flush=True)

        print(
            f"\n Epoch {epoch_id} Step {global_step}: {mode}_loss {loss:.2f} "
            f"ppl {ppl:.2f} ",
            file=eval_logger)

        # tensorboard
        # writer.add_scalar('valid_loss', loss, global_step=global_step)
        # writer.add_scalar('valid_ppl', ppl, global_step=global_step)

    return loss, ppl


(tr_loss, tr_ppl, mean_ppl, nb_tr_examples, nb_tr_steps) = 0.0, 0.0, 0.0, 0, 0
prev_best, tolerance=10000, 0 # for early stop
MAX_TOLERANCE=3

print ('#### Training ####')
while True:
    model.train()
    # 每个epoch report的都是平均loss......
    # (tr_loss, tr_ppl, mean_ppl, nb_tr_examples, nb_tr_steps) = 0.0, 0.0, 0.0, 0, 0
    n_token_real, n_token_total = 0, 0
    train_start_time_epoch = time.time()
    for batch in train_dataloader:
        batch = tuple(t.to(args.device) for t in batch)
        
        input_ids, position_ids, token_ids, label_ids, *_ = batch
        
        if args.visualize_train_data and (args.local_rank == -1 or get_rank() == 0):
            visualize_train_data(batch, enc, args)

        if args.no_token_id:
            token_ids = None
        loss, ppl, _ = model(input_ids, position_ids, token_ids, label_ids)


        if args.n_gpu > 1:
            loss = loss.mean()
            ppl = ppl.mean()
        loss = loss / (args.train_batch_size / input_ids.shape[0])
        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        # tr_loss += float(loss.item()) * (args.train_batch_size / input_ids.shape[0])
        tr_loss = 0.97 * tr_loss + 0.03 * float(loss.item()) * (args.train_batch_size / input_ids.shape[0])
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        # mean_loss = tr_loss / nb_tr_steps
        mean_loss = tr_loss
        if ppl.item() < INF:
            # tr_ppl += ppl.item()
            tr_ppl = 0.97 * tr_ppl + 0.03 * ppl.item()
        else:
            # tr_ppl += mean_ppl
            tr_ppl = 0.97 * tr_ppl + 0.03 * mean_ppl
        # mean_ppl = tr_ppl / nb_tr_steps
        mean_ppl = tr_ppl

        n_token_total += input_ids.shape[0] * input_ids.shape[1]
        n_token_real += (input_ids != 0).sum().item()

        # gradient update
        step += 1
        if step % args.gradient_accumulation_steps == 0:
            set_lr(optimizer, global_step,
                   args.lr_schedule, args.learning_rate,
                   args.warmup_steps, args.warmup_proportion,
                   config.n_embd, args.num_optim_steps)

            if args.local_rank != -1:
                grads = [p.grad.data for p in model.parameters()
                         if p.requires_grad and p.grad is not None]
                all_reduce_and_rescale_tensors(grads, float(1))

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            # Print log info to file
            if args.local_rank != -1:
                mean_loss = sum(all_gather_list(mean_loss)) / get_world_size()
                mean_ppl = sum(all_gather_list(mean_ppl)) / get_world_size()
                n_token_real_all_proc = sum(all_gather_list(n_token_real))
                n_token_total_all_proc = sum(all_gather_list(n_token_total))
            else:
                n_token_real_all_proc = n_token_real
                n_token_total_all_proc = n_token_total

            if args.local_rank == -1 or get_rank() == 0:
                epoch_time = time.time() - train_start_time_epoch
                if pbar is not None:
                    pbar.set_postfix_str(
                        f"tok/s: {n_token_real_all_proc//epoch_time//1000}k "
                        f"ppl: {mean_ppl:.2f} epoch: {epoch}")
                    pbar.update(1)

                if global_step % args.log_step == 0:
                    print ('epoch/step/lr:{}/{}/{}  '
                        'loss:{:.2f}  ppl:{:.2f} '.format(
                        epoch + 1, global_step + 1, optimizer.param_groups[0]['lr'], mean_loss, mean_ppl
                        )
                    )

                    if not args.visualize_train_data:
                        print ('epoch/step/lr:{}/{}/{}  '
                            'loss:{:.2f}  ppl:{:.2f} '.format(
                            epoch + 1, global_step + 1, optimizer.param_groups[0]['lr'], mean_loss, mean_ppl
                            ),
                            file=train_logger
                        )

            if global_step % args.valid_step == 0:
            # if global_step % args.valid_step == 0 or global_step == 1:
                if args.local_rank == -1 or get_rank() == 0 and not args.visualize_train_data:
                    # only rank 0 process evaluate
                    torch.save(
                        {k: (v.cpu() if v is not None else None)  # save to cpu tensors
                         for k, v in model.state_dict().items()},
                        join(output_dir, f'{global_step}.pkl')
                        )

                    eval_loss, eval_ppl = my_eval_model_loss(
                        model, enc, eval_dataloader_loss, epoch, global_step, args)

                    print('eval: {},{},{},{},{}'.format(
                        epoch+1, global_step+1, step+1, eval_loss, eval_ppl),
                        file=eval_logger)

                    if global_step % args.test_step == 0:
                        test_loss, test_ppl = my_eval_model_loss(
                            model, enc, test_dataloader_loss, epoch, global_step, args, is_test=True)

                        print('test: {},{},{},{},{}'.format(
                            epoch+1, global_step+1, step+1, test_loss, test_ppl),
                            file=eval_logger)

                        if test_ppl < prev_best:
                            tolerance = 0
                            prev_best = test_ppl
                        else:
                            tolerance += 1

                    logger.info('current learning rate: '
                                + str(optimizer.param_groups[0]['lr']))
                    model.train()

            if tolerance >= MAX_TOLERANCE: # early stop
                break

            if global_step >= args.num_optim_steps:
                break

        if (step+1) % CACHE_EMPTY_STEP == 0:
            torch.cuda.empty_cache()

    if global_step >= args.num_optim_steps:
        break

    epoch += 1
    if epoch > args.num_epoch:
        break
    
    # shuffle training set per epoch
    if args.shuffle:
        print (f'shuffle training set at the beginning of epoch {epoch}')
        train_dataset = PersonaDataset(args.train_input_file, max_len=args.max_seq_length, with_persona_label=args.with_persona_label, shuffle=True, all_seq_loss=args.all_seq_loss, single_turn=args.single_turn)
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=PersonaDataset.collate)


if args.local_rank == -1 or get_rank() == 0:
    if pbar is not None:
        pbar.close()
    if not args.visualize_train_data:
        train_logger.close()
        eval_logger.close()
