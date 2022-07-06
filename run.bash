## train 
# full
nohup python train.py --gpu_ids 0 --net_version Unsup31_15_53 --loss_version Lossv3 --losses L1_smooth --batch_size 8  --input_size 64 --use_views 9 --pad 0 --out_c 1 --input_c 3 --alpha 1e-2 --n_epochs 100 > full_synth_2.out &


# finetune 
nohup python train.py --gpu_ids 1 --net_version Unsup27_16_16 --loss_version Lossv3 --losses L1_smooth --batch_size 16  --input_size 64 --use_views 9 --pad 0 --out_c 1 --input_c 3 --alpha 1e-2 --n_epochs 70 > onlyhci.out &

# pretrain
--epoch_count () --time_str ()

# test
python test.py --gpu_ids 3 --net_version Unsup27_16_16 --input_c 3 --time_str 2022-07-05-05-23



# exp