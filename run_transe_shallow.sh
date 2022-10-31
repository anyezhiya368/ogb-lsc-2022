SAVE_PATH=/raid/wgh1/model_dglke/model_output
DATA_PATH=/raid/liuxiao/data/WikiKG90Mv2
GPU=4,5,6,7

CUDA_VISIBLE_DEVICES=$GPU  dglke_train --model_name TransE_l2 \
  --hidden_dim 600 --gamma 10  --valid -adv --mix_cpu_gpu --num_proc 4 --num_thread 16 \
  --gpu 0 1 2 3 \
  --async_update \
  --lr_decay_rate 1 --lr_decay_interval 10000 \
  --print_on_screen --encoder_model_name shallow --save_path $SAVE_PATH \
  --data_path $DATA_PATH \
  --mlp_lr 0.0001 \
  --seed $RANDOM \
  --neg_sample_size 8192 --batch_size 8192 --lr 0.1 --regularization_coef 1.0e-9 \
  --max_step 20000000 --force_sync_interval 1000 --eval_interval 400000 \
  --LRE --LRE_rank 200 --eval_percent 1. \
  --valid_candidate_path /raid/liuxiao/data/WikiKG90Mv2/wikikg90m-v2/processed/val_t_candidate.npy
