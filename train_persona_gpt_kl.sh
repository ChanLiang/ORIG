
export CUDA_VISIBLE_DEVICES=1,2
# python persona_train.py \
time=`date '+%F-%H:%M:%S'`

path=./gpt2-base
ckp=./gpt2-base/pytorch_model.bin

no_token_id=True
new_type_ids=False

# for debug
small_data=False
visualize_train_data=False

exp_name=new_code_kl1.0
only_persona_response=False

single_turn=False
input_persona_label=False

alpha=1.0

# python -m torch.distributed.launch --nproc_per_node=2 ./persona_train.py  \
python -m torch.distributed.launch --nproc_per_node=2  ./persona_train_kl.py.py  \
--model_name_or_path $path \
--init_checkpoint $ckp \
--max_seq_length 180 \
--alpha $alpha \
--train_input_file  './data/train/output' \
--eval_input_file  './data/valid/output' \
--test_input_file  './data/test/output' \
--train_batch_size 32 \
--gradient_accumulation_steps 2 \
--eval_batch_size 8 \
--num_epoch 6 \
--num_optim_steps 6000 \
--learning_rate 1e-5 \
--valid_step 200 \
--test_step 500 \
--log_step 50 \
--warmup_proportion 0.1 \
--warmup_steps 600 \
--output_dir 'output/persona' \
--log_dir log/persona \
--exp_name $exp_name \
--visualize_train_data $visualize_train_data \
--no_token_id $no_token_id \
--new_type_ids $new_type_ids \
--shuffle False \
--single_turn $single_turn \
--only_persona_response $only_persona_response \
--with_persona_label $input_persona_label 1>log/persona/res_${exp_name}_${time} 2>log/persona/err_${exp_name}_${time}
