#!/bin/sh
python train_premirna.py --workers 0 --arch deepmir_cw_bn --epochs 1 --batch-size 64 --lr 0.001 --whitened_layers 12 --concepts AU_pair_begin_maturemiRNA,base_beginstem_4nt_clean,presence_terminal_loop,palindrome_True --prefix DEEPMIR_PREMIRNA_CPT_WHITEN_TRANSFER modhsa_original/

# training script
python train_premirna.py --workers 0 --arch deepmir_cw_bn --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 6 --concepts presence_terminal_loop  --prefix DEEPMIR_PREMIRNA_CPT_WHITEN_TRANSFER modhsa_original/


# evaluation script
python train_premirna.py --workers 0 --arch deepmir_cw_bn --epochs 100 --batch-size 64 --lr 0.001 --whitened_layers 6 --concepts presence_terminal_loop --prefix DEEPMIR_PREMIRNA_CPT_WHITEN_TRANSFER --resume ./checkpoints/DEEPMIR_PREMIRNA_CPT_WHITEN_TRANSFER_model_best.pth.tar modhsa_original/ --evaluate

# pretrain script
python train_premirna.py --workers 0 --arch deepmir_bn  --epochs 100 --batch-size 128 --lr 0.001 --whitened_layers 6  --concepts presence_terminal_loop --prefix DEEPMIR_PREMIRNA_BN modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_bn  --epochs 10 --batch-size 32 --lr 0.001 --whitened_layers 6  --concepts presence_terminal_loop --prefix DEEPMIR_PREMIRNA_BN modhsa_original/

# resnet script
# for resnet, you can go up to 8 with the original code
python train_premirna.py --workers 0 --arch resnet_cw --epochs 1 --depth 18 --batch-size 64 --lr 0.001 --whitened_layers 6 --concepts presence_terminal_loop --prefix RESNET_PREMIRNA_CPT_WHITEN_TRANSFER  modhsa_padded/
# finetune resnet
python train_premirna.py --workers 0 --arch resnet_bn --epochs 100 --depth 18 --batch-size 128 --lr 0.001 --whitened_layers 6 --concepts AU_pair_begin_maturemiRNA,base_beginstem_4nt_clean,presence_terminal_loop,palindrome_True --prefix RESNET_PREMIRNA_CPT_WHITEN_TRANSFER  modhsa_original/
# pretrain resnet
python train_premirna.py --workers 0 --arch resnet_bn --epochs 40 --depth 18 --batch-size 128 --lr 0.001 --whitened_layers 6 --concepts AU_pair_begin_maturemiRNA --prefix RESNET_PREMIRNA_PRETRAIN  preparation/datasets/nonhsa_modmirbase_pretraininCWcode/
# evaluate
python train_premirna.py --workers 0 --arch resnet_cw --epochs 1 --depth 18 --batch-size 64 --lr 0.001 --whitened_layers 6 --concepts presence_terminal_loop --prefix RESNET_PREMIRNA_CPT_WHITEN_TRANSFER --resume ./checkpoints/presence_terminal_loop/RESNET_PREMIRNA_CPT_WHITEN_TRANSFER_6_checkpoint.pth.tar modhsa_padded/ --evaluate




# 6 june tests: WORKS
# pretrain
python train_premirna.py --workers 0 --arch deepmir_resnet_bn --epochs 40 --batch-size 128 --lr 0.001 --whitened_layers 2 --concepts presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_PRETRAIN  preparation/datasets/nonhsa_modmirbase_pretraininCWcode/

# fine-tune
python train_premirna.py --workers 0 --arch deepmir_resnet_bn --epochs 100 --batch-size 128 --lr 0.001 --whitened_layers 2 --concepts presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_BN modhsa_original/
# CW
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 5 --batch-size 64 --lr 0.1 --whitened_layers 1 --concepts AU_pair_begin_maturemiRNA,base_beginstem_4nt_clean,presence_terminal_loop,palindrome_True --prefix DEEPMIR_RESNET_PREMIRNA_CW modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 2 --concepts AU_pair_begin_maturemiRNA,base_beginstem_4nt_clean,presence_terminal_loop,palindrome_True --prefix DEEPMIR_RESNET_PREMIRNA_CW modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 3 --concepts AU_pair_begin_maturemiRNA,base_beginstem_4nt_clean,presence_terminal_loop,palindrome_True --prefix DEEPMIR_RESNET_PREMIRNA_CW modhsa_original/

# CW evaluation
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --batch-size 64 --lr 0.001 --whitened_layers 1 --concepts AU_pair_begin_maturemiRNA,base_beginstem_4nt_clean,presence_terminal_loop,palindrome_True --prefix DEEPMIR_RESNET_PREMIRNA_CW --resume ./checkpoints/resnet_premirna_checkpoints/AU_pair_begin_maturemiRNA_base_beginstem_4nt_clean_presence_terminal_loop_palindrome_True/DEEPMIR_RESNET_PREMIRNA_CW_1_model_best.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 2 --concepts AU_pair_begin_maturemiRNA,base_beginstem_4nt_clean,presence_terminal_loop,palindrome_True --prefix DEEPMIR_RESNET_PREMIRNA_CW --resume ./checkpoints/resnet_premirna_checkpoints/AU_pair_begin_maturemiRNA_base_beginstem_4nt_clean_presence_terminal_loop_palindrome_True/DEEPMIR_RESNET_PREMIRNA_CW_2_model_best.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 3 --concepts AU_pair_begin_maturemiRNA,base_beginstem_4nt_clean,presence_terminal_loop,palindrome_True --prefix DEEPMIR_RESNET_PREMIRNA_CW --resume ./checkpoints/resnet_premirna_checkpoints/AU_pair_begin_maturemiRNA_base_beginstem_4nt_clean_presence_terminal_loop_palindrome_True/DEEPMIR_RESNET_PREMIRNA_CW_2_model_best.pth.tar modhsa_original/ --evaluate




# CW
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 3 --batch-size 64 --lr 0.001 --whitened_layers 1 --concepts base_beginstem_4nt_clean,palindrome_True --prefix DEEPMIR_RESNET_PREMIRNA_CW modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 3 --batch-size 64 --lr 0.001 --whitened_layers 2 --concepts base_beginstem_4nt_clean,palindrome_True --prefix DEEPMIR_RESNET_PREMIRNA_CW modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 3 --concepts base_beginstem_4nt_clean,palindrome_True --prefix DEEPMIR_RESNET_PREMIRNA_CW modhsa_original/

# CW evaluation
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 1 --concepts base_beginstem_4nt_clean,palindrome_True --prefix DEEPMIR_RESNET_PREMIRNA_CW --resume ./checkpoints/resnet_premirna_checkpoints/base_beginstem_4nt_clean,palindrome_True/DEEPMIR_RESNET_PREMIRNA_CW_1_model_best.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 3 --batch-size 64 --lr 0.001 --whitened_layers 2 --concepts base_beginstem_4nt_clean,palindrome_True --prefix DEEPMIR_RESNET_PREMIRNA_CW --resume ./checkpoints/resnet_premirna_checkpoints/base_beginstem_4nt_clean,palindrome_True/DEEPMIR_RESNET_PREMIRNA_CW_2_model_best.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 3 --concepts base_beginstem_4nt_clean,palindrome_True --prefix DEEPMIR_RESNET_PREMIRNA_CW --resume ./checkpoints/resnet_premirna_checkpoints/base_beginstem_4nt_clean,palindrome_True/DEEPMIR_RESNET_PREMIRNA_CW_2_model_best.pth.tar modhsa_original/ --evaluate

# BN evaluation
python train_premirna.py --workers 0 --arch deepmir_resnet_bn --epochs 1 --batch-size 64 --lr 0.001 --whitened_layers 1 --concepts presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_BN --resume ./checkpoints/DEEPMIR_RESNET_PREMIRNA_BN_2_checkpoint.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_bn --epochs 1 --batch-size 64 --lr 0.001 --whitened_layers 2 --concepts presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_BN --resume ./checkpoints/DEEPMIR_RESNET_PREMIRNA_BN_2_checkpoint.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_bn --epochs 1 --batch-size 64 --lr 0.001 --whitened_layers 3 --concepts presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_BN --resume ./checkpoints/DEEPMIR_RESNET_PREMIRNA_BN_2_checkpoint.pth.tar modhsa_original/ --evaluate



# CW with mean activation
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --act_mode 'mean' --epochs 1 --batch-size 128 --lr 0.5 --whitened_layers 1 --concepts base_beginstem_4nt_clean,palindrome_True --prefix DEEPMIR_RESNET_PREMIRNA_CW_mean modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --act_mode 'mean' --epochs 1 --batch-size 128 --lr 0.1 --whitened_layers 2 --concepts base_beginstem_4nt_clean,palindrome_True --prefix DEEPMIR_RESNET_PREMIRNA_CW_mean modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --act_mode 'mean' --epochs 1 --batch-size 32 --lr 0.001 --whitened_layers 3 --concepts base_beginstem_4nt_clean,palindrome_True --prefix DEEPMIR_RESNET_PREMIRNA_CW_mean modhsa_original/

# CW with mean activation evaluation
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --act_mode 'mean' --epochs 1 --batch-size 128 --lr 0.5 --whitened_layers 1 --concepts base_beginstem_4nt_clean,palindrome_True --prefix DEEPMIR_RESNET_PREMIRNA_CW_mean --resume ./checkpoints/base_beginstem_4nt_clean_palindrome_True/DEEPMIR_RESNET_PREMIRNA_CW_mean_1_checkpoint.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --act_mode 'mean' --epochs 1 --batch-size 128 --lr 0.1 --whitened_layers 2 --concepts base_beginstem_4nt_clean,palindrome_True --prefix DEEPMIR_RESNET_PREMIRNA_CW_mean --resume ./checkpoints/base_beginstem_4nt_clean_palindrome_True/DEEPMIR_RESNET_PREMIRNA_CW_mean_2_checkpoint.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --act_mode 'mean' --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 3 --concepts base_beginstem_4nt_clean,palindrome_True --prefix DEEPMIR_RESNET_PREMIRNA_CW_mean --resume ./checkpoints/base_beginstem_4nt_clean_palindrome_True/DEEPMIR_RESNET_PREMIRNA_CW_mean_3_checkpoint.pth.tar modhsa_original/ --evaluate


# CW with terminal loop + clean base
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --act_mode 'mean' --batch-size 128 --lr 0.5 --whitened_layers 1 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_CW_mean modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --batch-size 128 --lr 0.1 --whitened_layers 2 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_CW_mean modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --batch-size 128 --lr 0.1 --whitened_layers 3 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_CW_mean modhsa_original/

# CW with terminal loop + clean base evaluation
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --act_mode 'mean' --batch-size 128 --lr 0.5 --whitened_layers 1 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_CW_mean --resume ./checkpoints/base_beginstem_4nt_clean_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_CW_mean_1_checkpoint.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --batch-size 128 --lr 0.1 --whitened_layers 2 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_CW_mean --resume ./checkpoints/base_beginstem_4nt_clean_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_CW_mean_2_checkpoint.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --batch-size 128 --lr 0.1 --whitened_layers 3 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_CW_mean --resume ./checkpoints/base_beginstem_4nt_clean_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_CW_mean_3_checkpoint.pth.tar modhsa_original/ --evaluate


# CW with terminal loop + clean base and 3 layers together
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --batch-size 32 --lr 0.001 --whitened_layers 1,2,3 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_CW modhsa_original/

# CW with terminal loop + clean base evaluation
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --batch-size 32 --lr 0.001 --whitened_layers 1,2,3 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_CW --resume ./checkpoints/base_beginstem_4nt_clean_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_CW_1_2_3_checkpoint.pth.tar modhsa_original/ --evaluate



# Resnet-deepmir with extra relu's to get permutation function working
# pretrain
python train_premirna.py --workers 0 --arch deepmir_resnet_bn_v2 --epochs 40 --batch-size 128 --lr 0.001 --whitened_layers 1 --concepts presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_BN  preparation/datasets/nonhsa_modmirbase_pretraininCWcode/
# finetune
python train_premirna.py --workers 0 --arch deepmir_resnet_bn_v2 --epochs 100 --batch-size 128 --lr 0.001 --whitened_layers 1 --concepts presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_BN  modhsa_original/
# cw
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 2 --batch-size 64 --lr 0.1 --whitened_layers 1 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 2 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 3 --concepts base_beginstem_4nt_clean,presence_terminal_loop  --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 5 --batch-size 64 --lr 0.1 --whitened_layers 1,2,3 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER modhsa_original/
# evaluation
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 1 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER --resume ./checkpoints/base_beginstem_4nt_clean_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER_1_checkpoint.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 2 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER --resume ./checkpoints/base_beginstem_4nt_clean_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER_2_checkpoint.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 3 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER --resume ./checkpoints/base_beginstem_4nt_clean_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER_3_checkpoint.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 1,2,3 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER --resume ./checkpoints/base_beginstem_4nt_clean_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER_1_2_3_checkpoint.pth.tar modhsa_original/ --evaluate



