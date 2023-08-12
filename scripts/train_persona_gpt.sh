
export CUDA_VISIBLE_DEVICES=0,3
# python persona_train.py \
time=`date '+%F-%H:%M:%S'`
exp_name='YOUR_EXP_NAME'
ckp_path=./gpt2-base/pytorch_model.bin

no_token_id=True
new_type_ids=False

small_data=False
visualize_train_data=False
only_persona_response=True
single_turn=False
input_persona_label=False

python -m torch.distributed.launch --nproc_per_node=2 ./persona_train.py  \
--model_name_or_path $path \
--init_checkpoint $ckp \
--max_seq_length 180 \
--train_input_file  './data/train/output' \
--eval_input_file  './data/valid/output' \
--test_input_file  './data/test/output' \
--train_batch_size 32 \
--gradient_accumulation_steps 2 \
--eval_batch_size 8 \
--num_epoch 10 \
--learning_rate 2e-5 \
--valid_step 500 \
--test_step 500 \
--log_step 50 \
--warmup_proportion 0.1 \
--warmup_steps 2000 \
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
