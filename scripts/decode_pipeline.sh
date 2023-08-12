# 1. general parameters
exp_name=new_code_bl
ckp_path=output/persona/new_code_bl_adam-lr-1e-05-bz-32-time-2023-01-14194003/1400.pkl

model_size='small'
eos_in_decoding=False
w_typeId=False
input_ground_truth_persona_label=False

# ===========================================================================
# 2.【general parameters】
date="2023-01-14"
decoding_strategy="beam_search" # or "top10_top0.9_T0.9"
beam=5
min_decode_length=1
batch_size=32 
debug=False # or True
gpu=0

# ===========================================================================
# 3.【decoding shell】
echo $exp_name

# for((id=0;id<=119;id++)) # for order-permutation experiment
for((id=0;id<=0;id++)) # for normal-order experiment
do
for input_assigned_persona_label in '-2' # expired: don't use any assigned labels
do
    for order in normal_ord # expired...
    do
        echo $id $input_assigned_persona_label
        python decoding.py \
        --date $date \
        --exp_name $exp_name \
        --model_path $ckp_path \
        --model_size $model_size \
        --order $order \
        --decoding_strategy $decoding_strategy \
        --beam $beam \
        --eos_in_decoding $eos_in_decoding \
        --input_ground_truth_label $input_ground_truth_persona_label \
        --batch_size $batch_size \
        --debug $debug \
        --gpu $gpu \
        --w_typeId $w_typeId \
        --min_decode_length $min_decode_length \
        --input_assigned_persona_label $input_assigned_persona_label \
        --permutaion_id ${id} \
        --bar True
        wait
    done
done
done