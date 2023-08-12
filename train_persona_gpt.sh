
export CUDA_VISIBLE_DEVICES=0,3
# python persona_train.py \
time=`date '+%F-%H:%M:%S'`

# path=./base-model
# ckp=./base-model/small_ft.pkl

path=./gpt2-base
ckp=./gpt2-base/pytorch_model.bin


# exp_name='w_persona_label_shuffle'
# exp_name='w_persona_label_eos_response_unshuffle'
# exp_name='w_persona_label_eos_response_shuffle'
# exp_name='w_persona_label_wo_threshold_eos_response_shuffle'
# exp_name=naive_dialogpt_base
# exp_name=naive_gpt2_base
# exp_name=naive_gpt2_base_single_turn


# exp_name=gpt2_base_joint_decoding_wo_typeId # use decoding_3D.py, wo_typeId=True
no_token_id=True
new_type_ids=False

# for debug
small_data=False
visualize_train_data=False

# exp_name=gpt2_base_joint_decoding_wo_typeId_only_persona_response
exp_name=gpt2_base_baseline_wo_typeId_only_persona_response
only_persona_response=True

single_turn=False
# input_persona_label=True
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
# --with_persona_label $input_persona_label
