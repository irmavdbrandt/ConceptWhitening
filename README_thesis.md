## Towards Concept-based Interpretability of Pre-miRNA Detection using Convolutional Neural Networks

This repository is an extension to the Concept Whitening technique created by Zhi Chen, Yijie Bei, Cynthia Rudin (2020). 
The technique is extended and adjusted as part of a master thesis: "Towards concept-based interpretability of pre-miRNA 
detection using convolutional neural networks". The main contribution of this thesis is first step towards concept-based interpretability of the
pre-miRNA detection predictions. We propose a framework that aims to provide domain experts
with an understanding of a CNN's class predictions for the image-based pre-miRNA detection
task in terms of concepts defined based on existing structural pre-miRNA knowledge.

### Project

#### Built With

- PyTorch
- Neptune

The experiments are logged on a Neptune observer. The results, code version and model settings of all experiments can be 
found in the [Interpret-rna project](https://app.neptune.ai/irmavdbrandt/Interpret-rna/experiments?split=tbl&dash=charts&viewId=standard-view). For 
access please e-mail to irma.vdbrandt@gmail.com. This repository contains the code, and results/figures included in the 
thesis.

#### Project Structure
The structure of the project is shown in the framework figure below. 

- Concept definition, annotation and quantitation: First, one needs to define concepts related to 
(pre-)miRNAs. Currently, this is done based on existing research on their detection, the pre-miRNA encoding algorithm of 
[Cordero, J., Menkovski, V., & Allmer, J. (2020)](https://www.biorxiv.org/content/10.1101/840579v2.abstract), and 
their saliency results from training the DeepMir model with the encoded pre-miRNA data. The concepts are defined
in the concept_detection.py file in the preparation folder. The concept_saving.py file checks for each data instance
whether a concept defined in concept_detection.py is present. The pretrain_data.py file reads data instances and stores
them in folders in the project, based on the class label. After collecting the concept information from all data
  instances, summary statistics such as the presence of a concept per class label are collected in the concept_saving.py
  file. Finally, the training data is split into k folds in order to use k-fold cross validation during training. Also,
  in this step the example images of the concepts are collected and stored into folders according to the desired
  folder structure for the Concept Whitening procedure defined in the README_originalCWpaper. 
  
- Concept alignment: Next, a concept whitening model can be trained using the collected datasets from the previous step.
In order to use a concept whitening model, one needs pretrained weights from a convolutional neural network with the same 
  architecture as the concept whitening model but excluding one or multiple concept whitening layers. In our case, we have 
  adapted the original DeepMir model from [Cordero, J., Menkovski, V., & Allmer, J. (2020)](https://www.biorxiv.org/content/10.1101/840579v2.abstract),
  hence we first pretrain this adapted model. Next, we apply the concept whitening technique to this model. Both training
  procedures can be started using the train_premirna.py file.


- Concept-based interpretability and evaluation: As a final step, the results of the trained concept whitening model
are evaluated. This includes both the classification performance on the detection task, and the learning of our
  pre-miRNA concepts, and their importance for the detection task. The classification performance is evaluated using the 
  train_premirna.py file, the learning of concepts, and their importance for the detection task using the plot_functions_cw.py
  and concept_influence.py file, which is stored in the report_figures folder. 

![CW_framework.pdf](CW_framework.pdf)


#### Prerequisites
The packages needed to run the project are listed in the requirements.txt file.


### Usage
Running the training procedures works as follows.

When pretraining or finetuning, use the architectures that include a "bn", meaning that no CW layer is included. Set the number of epochs,
  batch size and learning rate. In case of hardware where multiple workers can be activated, increase the number of workers.
  The prefix will be used to store the trained model. Specify the dataset used during pretraining in the data argument.

When training a concept whitening (CW) model, use architecturs including a "cw". Additionally, specify which layer is whitened,
for reference of how this indexing of layers works, please look at the README_originalCWpaper file. Also, specify
which concepts should be included. Evaluating CW models is similar, however, one has to include which checkpoint of a trained CW
model should be used during evaluation and which data fold is associated with this checkpoint. 


- Pretraining:
````
python train_premirna.py --workers 0 --arch deepmir_vfinal_bn --epochs 40 --batch-size 128 --lr 0.0001 --prefix DEEPMIR_vfinal_BN_pretrain --data datasets/nonhsa_modmirbase_pretraininCWcode/ --type_training pretrain
````

- Fine-tuning:
````
python train_premirna.py --workers 0 --arch deepmir_vfinal_bn --epochs 100 --batch-size 128 --lr 0.00001 --prefix DEEPMIR_vfinal_BN_finetune --data datasets/modhsa_original/original_dataset/ --type_training finetune

````

- Training a concept whitening (CW) model:
````
python train_premirna.py --workers 0 --arch deepmir_vfinal_cw --epochs 100 --batch-size 12 --lr 0.001 --whitened_layers 3 --concepts large_asymmetric_bulge,base_pairs_wobbles_in_stem --prefix DEEPMIR_vfinal_CPT_WHITEN_TRANSFER_final --data datasets/modhsa_original/CW_dataset --type_training cw
````

- Evaluating a CW model:
````
python train_premirna.py --workers 0 --arch deepmir_vfinal_cw --epochs 30 --batch-size 1 --lr 0.1 --whitened_layers 3 --concepts large_asymmetric_bulge,base_pairs_wobbles_in_stem --prefix DEEPMIR_vfinal_CPT_WHITEN_TRANSFER_final --checkpoint_name ./checkpoints/large_asymmetric_bulge_base_pairs_wobbles_in_stem/DEEPMIR_vfinal_CPT_WHITEN_TRANSFER_final_3_foldn1_model_best.pth.tar --data datasets/modhsa_original/original_dataset/ --type_training evaluate --foldn_bestmodel 1
````

- Collecting the activation values from the final layer of the CW model: for the concept influence figures, activation 
  values of the CW models are required. The python command is similar to the evaluation of a CW model command, except for
  the type_training argument. Creating the concept influence figures requires running the concept_influence.py file with
  the desired arguments given to the functions in that file. 

````
python train_premirna.py --workers 0 --arch deepmir_vfinal_cw --epochs 30 --batch-size 1 --lr 0.1 --whitened_layers 3 --concepts large_asymmetric_bulge,base_pairs_wobbles_in_stem --prefix DEEPMIR_vfinal_CPT_WHITEN_TRANSFER_final --checkpoint_name ./checkpoints/large_asymmetric_bulge_base_pairs_wobbles_in_stem/DEEPMIR_vfinal_CPT_WHITEN_TRANSFER_final_3_foldn1_model_best.pth.tar --data datasets/modhsa_original/original_dataset --type_training get_activations --foldn_bestmodel 1
````
