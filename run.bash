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
2022-07-06-11-44 finetune       hci          nozq 
# 2022-07-06-12-15 finetune       hci          zq
# 2022-07-06-12-31 trans          hci          zq
2022-07-06-12-49 trans          hci-hciold   zq
2022-07-06-16-59 finetune       hci-hciold   nozq
2022-07-06-17-04 trans          hci-hciold   zq           sm2


# 2022-07-07-05-10 finetune       hci          zq           sm
# 2022-07-07-03-56 finetune       hci          zq           sm5
2022-07-07-05-11 trans          hci            zq           sm
2022-07-07-08-22 fine          hci-all         zq           sm
2022-07-07-12-07 trans         hci-all         zq           sm

