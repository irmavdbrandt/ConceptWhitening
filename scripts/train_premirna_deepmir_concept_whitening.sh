#!/bin/sh
python train_premirna.py --workers 0 --arch deepmir_cw --depth 8 --epochs 10 --batch-size 64 --lr 0.001 --whitened_layers 12 --concepts AU_pair_begin_maturemiRNA,base_beginstem_4nt_clean,base_pair_successions_presence,presence_terminal_loop,symmetric_bulge_presence --prefix DEEPMIR_PREMIRNA_CPT_WHITEN_TRANSFER /Users/IrmavandenBrandt/PycharmProjects/ConceptWhitening/modhsa_original/


python train_premirna.py --workers 0 --arch deepmir_cw --depth 8 --epochs 10 --batch-size 64 --lr 0.001 --whitened_layers 12 --concepts presence_terminal_loop --prefix DEEPMIR_PREMIRNA_CPT_WHITEN_TRANSFER /Users/IrmavandenBrandt/PycharmProjects/ConceptWhitening/modhsa_original/


python train_premirna.py --workers 4 --arch resnet_cw --depth 18 --epochs 10 --batch-size 64 --lr 0.1 --whitened_layers 6 --concepts presence_terminal_loop --prefix DEEPMIR_PREMIRNA_CPT_WHITEN_TRANSFER /Users/IrmavandenBrandt/PycharmProjects/ConceptWhitening/modhsa_original/
