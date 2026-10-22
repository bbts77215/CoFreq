seq_len=96
model_name=CoFreq

data_name=ETTm2
root_path_name=./dataset/ETT
data_path_name=ETTm2.csv
model_id_name=ETTm2

learning_rate=0.0001
for pred_len in 96 192 336 720
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name\
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --des 'Exp' \
      --train_epochs 100\
      --patience 10\
      --lradj 'sigmoid'\
      --pct_start 0.3\
      --itr 1\
      --batch_size 32\
      --learning_rate $learning_rate
done