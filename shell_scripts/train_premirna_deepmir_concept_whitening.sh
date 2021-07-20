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
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 3 --batch-size 64 --lr 0.001 --whitened_layers 1 --concepts base_beginstem_4nt_clean,palindrome_True --prefix DEEPMIR_RESNET_PREMIRNA_CW modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 3 --batch-size 64 --lr 0.001 --whitened_layers 2 --concepts base_beginstem_4nt_clean,palindrome_True --prefix DEEPMIR_RESNET_PREMIRNA_CW modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 3 --concepts base_beginstem_4nt_clean,palindrome_True --prefix DEEPMIR_RESNET_PREMIRNA_CW modhsa_original/

# CW evaluation
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 1 --concepts base_beginstem_4nt_clean,palindrome_True --prefix DEEPMIR_RESNET_PREMIRNA_CW --resume ./checkpoints/resnet_premirna_checkpoints/base_beginstem_4nt_clean,palindrome_True/DEEPMIR_RESNET_PREMIRNA_CW_1_model_best.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 3 --batch-size 64 --lr 0.001 --whitened_layers 2 --concepts base_beginstem_4nt_clean,palindrome_True --prefix DEEPMIR_RESNET_PREMIRNA_CW --resume ./checkpoints/resnet_premirna_checkpoints/base_beginstem_4nt_clean,palindrome_True/DEEPMIR_RESNET_PREMIRNA_CW_2_model_best.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 3 --concepts base_beginstem_4nt_clean,palindrome_True --prefix DEEPMIR_RESNET_PREMIRNA_CW --resume ./checkpoints/resnet_premirna_checkpoints/base_beginstem_4nt_clean,palindrome_True/DEEPMIR_RESNET_PREMIRNA_CW_2_model_best.pth.tar modhsa_original/ --evaluate

# BN evaluation
python train_premirna.py --workers 0 --arch deepmir_resnet_bn --epochs 1 --batch-size 64 --lr 0.001 --whitened_layers 1 --concepts presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_BN --resume ./checkpoints/DEEPMIR_RESNET_PREMIRNA_BN_2_checkpoint.pth.tar modhsa_original/ --type_training=evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_bn --epochs 1 --batch-size 64 --lr 0.001 --whitened_layers 2 --concepts presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_BN --resume ./checkpoints/DEEPMIR_RESNET_PREMIRNA_BN_2_checkpoint.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_bn --epochs 1 --batch-size 64 --lr 0.001 --whitened_layers 3 --concepts presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_BN --resume ./checkpoints/DEEPMIR_RESNET_PREMIRNA_BN_2_checkpoint.pth.tar modhsa_original/ --evaluate



# CW with terminal loop + clean base
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --act_mode 'mean' --batch-size 128 --lr 0.5 --whitened_layers 1 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_CW_mean modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --batch-size 128 --lr 0.1 --whitened_layers 2 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_CW_mean modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --batch-size 128 --lr 0.1 --whitened_layers 3 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_CW_mean modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --batch-size 32 --lr 0.001 --whitened_layers 1,2,3 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_CW modhsa_original/

# CW with terminal loop + clean base evaluation
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --act_mode 'mean' --batch-size 128 --lr 0.5 --whitened_layers 1 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_CW_mean --resume ./checkpoints/base_beginstem_4nt_clean_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_CW_mean_1_checkpoint.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --batch-size 128 --lr 0.1 --whitened_layers 2 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_CW_mean --resume ./checkpoints/base_beginstem_4nt_clean_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_CW_mean_2_checkpoint.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --batch-size 128 --lr 0.1 --whitened_layers 3 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_CW_mean --resume ./checkpoints/base_beginstem_4nt_clean_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_CW_mean_3_checkpoint.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw --epochs 1 --batch-size 32 --lr 0.001 --whitened_layers 1,2,3 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_CW --resume ./checkpoints/base_beginstem_4nt_clean_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_CW_1_2_3_checkpoint.pth.tar modhsa_original/ --evaluate


# Resnet-deepmir with extra relu's to get permutation function working
# pretrain
python train_premirna.py --workers 0 --arch deepmir_resnet_bn_v2 --epochs 40 --batch-size 128 --lr 0.001 --whitened_layers 1 --concepts presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_BN  preparation/datasets/nonhsa_modmirbase_pretraininCWcode/
# fine-tune
python train_premirna.py --workers 0 --arch deepmir_resnet_bn_v2 --epochs 100 --batch-size 128 --lr 0.001 --whitened_layers 1 --concepts presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_BN  modhsa_original/
# CW
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 2 --batch-size 64 --lr 0.1 --whitened_layers 1 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 2 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 3 --concepts base_beginstem_4nt_clean,presence_terminal_loop  --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 10 --batch-size 64 --lr 0.1 --whitened_layers 1,2,3 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER modhsa_original/
# evaluation
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 2 --batch-size 64 --lr 0.1 --whitened_layers 1 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER --resume ./checkpoints/base_beginstem_4nt_clean_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER_1_checkpoint.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 2 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER --resume ./checkpoints/base_beginstem_4nt_clean_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER_2_checkpoint.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 3 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER --resume ./checkpoints/base_beginstem_4nt_clean_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER_3_checkpoint.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 1,2,3 --concepts base_beginstem_4nt_clean,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER --resume ./checkpoints/base_beginstem_4nt_clean_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER_1_2_3_checkpoint.pth.tar modhsa_original/ --evaluate


# try out several concepts
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 2 --batch-size 64 --lr 0.1 --whitened_layers 1 --concepts 4+_base_pair_successions,AU_pair_begin_maturemiRNA,base_wobble_stem_1nt_4nt,high_asymmetric_bulge,large_asymmetric_bulge,wide_asymmetric_bulge,palindrome_True,UGU,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 3 --batch-size 64 --lr 0.5 --whitened_layers 2 --concepts 4+_base_pair_successions,AU_pair_begin_maturemiRNA,base_wobble_stem_1nt_4nt,high_asymmetric_bulge,large_asymmetric_bulge,wide_asymmetric_bulge,palindrome_True,UGU,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 3 --concepts 4+_base_pair_successions,AU_pair_begin_maturemiRNA,base_wobble_stem_1nt_4nt,high_asymmetric_bulge,large_asymmetric_bulge,wide_asymmetric_bulge,palindrome_True,UGU,presence_terminal_loop  --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER modhsa_original/
# all layers together still not working nicely....
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 20 --batch-size 64 --lr 0.1 --whitened_layers 1,2,3 --concepts 4+_base_pair_successions,AU_pair_begin_maturemiRNA,base_wobble_stem_1nt_4nt,high_asymmetric_bulge,large_asymmetric_bulge,wide_asymmetric_bulge,palindrome_True,UGU,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER modhsa_original/
# evaluation
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 2 --batch-size 1 --lr 0.1 --whitened_layers 1 --concepts 4+_base_pair_successions,AU_pair_begin_maturemiRNA,base_wobble_stem_1nt_4nt,high_asymmetric_bulge,large_asymmetric_bulge,wide_asymmetric_bulge,palindrome_True,UGU,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER --resume ./checkpoints/4+_base_pair_successions_AU_pair_begin_maturemiRNA_base_wobble_stem_1nt_4nt_high_asymmetric_bulge_large_asymmetric_bulge_wide_asymmetric_bulge_palindrome_True_UGU_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER_1_checkpoint.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 1 --lr 0.5 --whitened_layers 2 --concepts 4+_base_pair_successions,AU_pair_begin_maturemiRNA,base_wobble_stem_1nt_4nt,high_asymmetric_bulge,large_asymmetric_bulge,wide_asymmetric_bulge,palindrome_True,UGU,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER --resume ./checkpoints/4+_base_pair_successions_AU_pair_begin_maturemiRNA_base_wobble_stem_1nt_4nt_high_asymmetric_bulge_large_asymmetric_bulge_wide_asymmetric_bulge_palindrome_True_UGU_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER_2_checkpoint.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 1 --lr 0.001 --whitened_layers 3 --concepts 4+_base_pair_successions,AU_pair_begin_maturemiRNA,base_wobble_stem_1nt_4nt,high_asymmetric_bulge,large_asymmetric_bulge,wide_asymmetric_bulge,palindrome_True,UGU,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER --resume ./checkpoints/4+_base_pair_successions_AU_pair_begin_maturemiRNA_base_wobble_stem_1nt_4nt_high_asymmetric_bulge_large_asymmetric_bulge_wide_asymmetric_bulge_palindrome_True_UGU_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER_3_checkpoint.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 64 --lr 0.001 --whitened_layers 1,2,3 --concepts 4+_base_pair_successions,AU_pair_begin_maturemiRNA,base_wobble_stem_1nt_4nt,high_asymmetric_bulge,large_asymmetric_bulge,wide_asymmetric_bulge,palindrome_True,UGU,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER --resume ./checkpoints/4+_base_pair_successions_AU_pair_begin_maturemiRNA_base_wobble_stem_1nt_4nt_high_asymmetric_bulge_large_asymmetric_bulge_wide_asymmetric_bulge_palindrome_True_UGU_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER_1_2_3_checkpoint.pth.tar modhsa_original/ --evaluate

# BN evaluation for resnet_deepmir_v2
python train_premirna.py --workers 0 --arch deepmir_resnet_bn_v2 --epochs 1 --batch-size 64 --lr 0.001 --whitened_layers 1 --concepts presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_BN_v2 --resume ./checkpoints/resnet_deepmir_v2/DEEPMIR_RESNET_PREMIRNA_v2_BN_1_checkpoint.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_bn_v2 --epochs 1 --batch-size 64 --lr 0.001 --whitened_layers 2 --concepts presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_BN_v2 --resume ./checkpoints/resnet_deepmir_v2/DEEPMIR_RESNET_PREMIRNA_BN_1_checkpoint.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_bn_v2 --epochs 1 --batch-size 64 --lr 0.001 --whitened_layers 3 --concepts presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_BN_v2 --resume ./checkpoints/resnet_deepmir_v2/DEEPMIR_RESNET_PREMIRNA_BN_1_checkpoint.pth.tar modhsa_original/ --evaluate




# try out several concepts
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 1 --concepts 5+_base_pair_successions,AU_pair_begin_maturemiRNA,base_wobble_stem_1nt_4nt,palindrome_True,UGU,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 64 --lr 0.1 --whitened_layers 2 --concepts 5+_base_pair_successions,AU_pair_begin_maturemiRNA,base_wobble_stem_1nt_4nt,palindrome_True,UGU,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 3 --batch-size 64 --lr 0.01 --whitened_layers 3 --concepts 5+_base_pair_successions,AU_pair_begin_maturemiRNA,base_wobble_stem_1nt_4nt,palindrome_True,UGU,presence_terminal_loop  --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER modhsa_original/
# didnt try 3 layers together yet...
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 20 --batch-size 64 --lr 0.1 --whitened_layers 1,2,3 --concepts 5+_base_pair_successions,AU_pair_begin_maturemiRNA,base_wobble_stem_1nt_4nt,palindrome_True,UGU,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER modhsa_original/
# evaluation
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 2 --batch-size 1 --lr 0.1 --whitened_layers 1 --concepts 5+_base_pair_successions,AU_pair_begin_maturemiRNA,base_wobble_stem_1nt_4nt,palindrome_True,UGU,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER --resume ./checkpoints/5+_base_pair_successions_AU_pair_begin_maturemiRNA_base_wobble_stem_1nt_4nt_palindrome_True_UGU_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER_1_checkpoint.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 1 --lr 0.1 --whitened_layers 2 --concepts 5+_base_pair_successions,AU_pair_begin_maturemiRNA,base_wobble_stem_1nt_4nt,palindrome_True,UGU,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER --resume ./checkpoints/5+_base_pair_successions_AU_pair_begin_maturemiRNA_base_wobble_stem_1nt_4nt_palindrome_True_UGU_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER_2_checkpoint.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 1 --lr 0.01 --whitened_layers 3 --concepts 5+_base_pair_successions,AU_pair_begin_maturemiRNA,base_wobble_stem_1nt_4nt,palindrome_True,UGU,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER --resume ./checkpoints/5+_base_pair_successions_AU_pair_begin_maturemiRNA_base_wobble_stem_1nt_4nt_palindrome_True_UGU_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER_3_checkpoint.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 1 --lr 0.001 --whitened_layers 1,2,3 --concepts 5+_base_pair_successions,AU_pair_begin_maturemiRNA,base_wobble_stem_1nt_4nt,palindrome_True,UGU,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER --resume ./checkpoints/5+_base_pair_successions_AU_pair_begin_maturemiRNA_base_wobble_stem_1nt_4nt_palindrome_True_UGU_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER_1_2_3_checkpoint.pth.tar modhsa_original/ --evaluate



# try out several concepts
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 32 --lr 0.1 --whitened_layers 1 --concepts presence_terminal_loop,asymmetric,base_pairs_wobbles_in_stem,width_gap_start --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 2 --batch-size 62 --lr 0.1 --whitened_layers 2 --concepts asymmetric,base_pairs_wobbles_in_stem --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 30 --batch-size 12 --lr 0.1 --whitened_layers 3 --concepts asymmetric,base_pairs_wobbles_in_stem --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER modhsa_original/
# all layers together still not working nicely....
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 64 --lr 0.001 --whitened_layers 2,3 --concepts presence_terminal_loop,asymmetric,base_pairs_wobbles_in_stem,width_gap_start --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER modhsa_original/
# evaluation
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 1 --lr 0.1 --whitened_layers 1 --concepts presence_terminal_loop,asymmetric,base_pairs_wobbles_in_stem,width_gap_start --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER --resume ./checkpoints/presence_terminal_loop_asymmetric_base_pairs_wobbles_in_stem_width_gap_start/DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER_1_checkpoint.pth.tar modhsa_original/ --evaluate
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 1 --lr 0.01 --whitened_layers 2 --concepts asymmetric,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER --resume ./checkpoints/asymmetric_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER_2_foldn2_checkpoint.pth.tar modhsa_original/original_dataset/ --evaluate --foldn_bestmodel 2
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 1 --lr 0.01 --whitened_layers 3 --concepts asymmetric,base_pairs_wobbles_in_stem --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER --resume ./checkpoints/asymmetric_base_pairs_wobbles_in_stem/DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER_3_foldn4_checkpoint.pth.tar modhsa_original/original_dataset/ --evaluate --foldn_bestmodel 4
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 1 --lr 0.001 --whitened_layers 2,3 --concepts presence_terminal_loop,asymmetric,base_pairs_wobbles_in_stem,width_gap_start --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER --resume ./checkpoints/presence_terminal_loop_asymmetric_base_pairs_wobbles_in_stem_width_gap_start/DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER_2_3_checkpoint.pth.tar modhsa_original/ --evaluate


# try out several concepts
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 32 --lr 0.1 --whitened_layers 1 --concepts largest_asymmetric_bulge,base_pairs_wobbles_in_stem --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 12 --lr 0.1 --whitened_layers 2 --concepts largest_asymmetric_bulge,base_pairs_wobbles_in_stem --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER modhsa_original/
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 30 --batch-size 12 --lr 0.01 --whitened_layers 3 --concepts AU_pair_begin_maturemiRNA,base_pairs_wobbles_in_stem,gap_start,large_asymmetric_bulge,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER modhsa_original/
# evaluation
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 1 --lr 0.1 --whitened_layers 1 --concepts largest_asymmetric_bulge,base_pairs_wobbles_in_stem,AU_pair_begin_maturemiRNA,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER --resume ./checkpoints/largest_asymmetric_bulge_base_pairs_wobbles_in_stem/DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER_1_foldn2_checkpoint.pth.tar modhsa_original/original_dataset/ --evaluate --foldn_bestmodel 2
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 1 --lr 0.01 --whitened_layers 2 --concepts largest_asymmetric_bulge,base_pairs_wobbles_in_stem --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER --resume ./checkpoints/largest_asymmetric_bulge_base_pairs_wobbles_in_stem/DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER_2_foldn2_checkpoint.pth.tar modhsa_original/original_dataset/ --evaluate --foldn_bestmodel 2
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v2 --epochs 1 --batch-size 1 --lr 0.001 --whitened_layers 3 --concepts largest_asymmetric_bulge,base_pairs_wobbles_in_stem --prefix DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER --resume ./checkpoints/largest_asymmetric_bulge_base_pairs_wobbles_in_stem/DEEPMIR_RESNET_PREMIRNA_v2_CPT_WHITEN_TRANSFER_3_foldn1_model_best.pth.tar modhsa_original/original_dataset/ --evaluate --foldn_bestmodel 1


#%%
# finetuning model
python train_premirna.py --workers 0 --arch deepmir_resnet_bn_v3 --epochs 20 --batch-size 128 --lr 0.0001 --whitened_layers 1 --concepts presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v3_BN_final  preparation/datasets/nonhsa_modmirbase_pretraininCWcode/ --type_training pretrain

python train_premirna.py --workers 0 --arch deepmir_resnet_bn_v3 --epochs 100 --batch-size 128 --lr 0.00001 --whitened_layers 1 --concepts presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v3_BN_final  modhsa_original/original_dataset/ --type_training finetune

python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v3 --epochs 100 --batch-size 12 --lr 0.01 --whitened_layers 3 --concepts large_asymmetric_bulge,base_pairs_wobbles_in_stem,gap_start,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v3_CPT_WHITEN_TRANSFER modhsa_original/ --type_training cw
python train_premirna.py --workers 0 --arch deepmir_resnet_cw_v3 --epochs 30 --batch-size 1 --lr 0.1 --whitened_layers 3 --concepts large_asymmetric_bulge,base_pairs_wobbles_in_stem,gap_start,presence_terminal_loop --prefix DEEPMIR_RESNET_PREMIRNA_v3_CPT_WHITEN_TRANSFER --resume ./checkpoints/large_asymmetric_bulge_base_pairs_wobbles_in_stem_gap_start_presence_terminal_loop/DEEPMIR_RESNET_PREMIRNA_v3_CPT_WHITEN_TRANSFER_3_foldn1_checkpoint.pth.tar modhsa_original/original_dataset/ --type_training evaluate --foldn_bestmodel 1
