import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

train_acc = pd.read_csv('training_results_performances/training_acc_finetuning.csv', header=None)
train_acc.columns = ['epoch', '??', 'accuracy']
train_loss = pd.read_csv('training_results_performances/training_loss_finetuning.csv', header=None)
train_loss.columns = ['epoch', '??', 'loss']
test_acc = pd.read_csv('training_results_performances/validation_acc_finetuning.csv', header=None)
test_acc.columns = ['epoch', '??', 'accuracy']
test_loss = pd.read_csv('training_results_performances/validation_loss_finetuning.csv', header=None)
test_loss.columns = ['epoch', '??', 'loss']

# %%
fig = plt.figure()
ax = plt.axes()

ax.plot(train_acc['epoch'], train_acc['accuracy'], label='Training accuracy')
ax.plot(test_acc['epoch'], test_acc['accuracy'], color='orange', label='Test accuracy')
plt.ylim(85, 100)
plt.xlim(0, 100)
plt.legend()
plt.title('Fine-tuning classification accuracy', fontsize=15, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (in %)')
plt.savefig('finetuning_accuracies.svg', format='svg')
plt.show()

# %%
fig_loss = plt.figure()
ax = plt.axes()

ax.plot(train_loss['epoch'], train_loss['loss'], label='Training loss')
ax.plot(test_loss['epoch'], test_loss['loss'], color='orange', label='Test loss')
plt.ylim(0, 0.3)
plt.xlim(0, 100)
plt.legend()
plt.title('Fine-tuning cross-entropy loss', fontsize=15, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('finetuning_loss.svg', format='svg')
plt.show()


# %%
train_acc_6concepts_fold0 = pd.read_csv('training_results_performances/training_0_acc_6concepts.csv', header=None)
train_acc_6concepts_fold0.columns = ['epoch', '??', 'accuracy']
train_acc_6concepts_fold1 = pd.read_csv('training_results_performances/training_1_acc_6concepts.csv', header=None)
train_acc_6concepts_fold1.columns = ['epoch', '??', 'accuracy']
train_acc_6concepts_fold2 = pd.read_csv('training_results_performances/training_2_acc_6concepts.csv', header=None)
train_acc_6concepts_fold2.columns = ['epoch', '??', 'accuracy']
train_acc_6concepts_fold3 = pd.read_csv('training_results_performances/training_3_acc_6concepts.csv', header=None)
train_acc_6concepts_fold3.columns = ['epoch', '??', 'accuracy']
train_acc_6concepts_fold4 = pd.read_csv('training_results_performances/training_4_acc_6concepts.csv', header=None)
train_acc_6concepts_fold4.columns = ['epoch', '??', 'accuracy']
train_loss_6concepts_fold1 = pd.read_csv('training_results_performances/training_1_loss_6concepts.csv', header=None)
train_loss_6concepts_fold1.columns = ['epoch', '??', 'loss']
val_acc_6concepts_fold0 = pd.read_csv('training_results_performances/validation_0_acc_6concepts.csv', header=None)
val_acc_6concepts_fold0.columns = ['epoch', '??', 'accuracy']
val_acc_6concepts_fold1 = pd.read_csv('training_results_performances/validation_1_acc_6concepts.csv', header=None)
val_acc_6concepts_fold1.columns = ['epoch', '??', 'accuracy']
val_acc_6concepts_fold2 = pd.read_csv('training_results_performances/validation_2_acc_6concepts.csv', header=None)
val_acc_6concepts_fold2.columns = ['epoch', '??', 'accuracy']
val_acc_6concepts_fold3 = pd.read_csv('training_results_performances/validation_3_acc_6concepts.csv', header=None)
val_acc_6concepts_fold3.columns = ['epoch', '??', 'accuracy']
val_acc_6concepts_fold4 = pd.read_csv('training_results_performances/validation_4_acc_6concepts.csv', header=None)
val_acc_6concepts_fold4.columns = ['epoch', '??', 'accuracy']
val_loss_6concepts = pd.read_csv('training_results_performances/validation_1_loss_6concepts.csv', header=None)
val_loss_6concepts.columns = ['epoch', '??', 'loss']

#%%
fig = plt.figure()
ax = plt.axes()
ax.plot(train_acc_6concepts_fold0['epoch'], train_acc_6concepts_fold0['accuracy'], label='Fold 1')
ax.plot(train_acc_6concepts_fold1['epoch'], train_acc_6concepts_fold1['accuracy'], label='Fold 2')
ax.plot(train_acc_6concepts_fold2['epoch'], train_acc_6concepts_fold2['accuracy'], label='Fold 3')
ax.plot(train_acc_6concepts_fold3['epoch'], train_acc_6concepts_fold3['accuracy'], label='Fold 4')
ax.plot(train_acc_6concepts_fold4['epoch'], train_acc_6concepts_fold4['accuracy'], label='Fold 5')
plt.ylim(50, 100)
plt.xlim(0, 45)
plt.legend()
plt.title('Training accuracies CW model with 6 concepts', fontsize=15, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (in %)')
plt.savefig('train_accuracies_CWmodel_6concepts.svg', format='svg')
plt.show()

#%%
fig = plt.figure()
ax = plt.axes()
ax.plot(val_acc_6concepts_fold0['epoch'], val_acc_6concepts_fold0['accuracy'], label='Fold 1')
ax.plot(val_acc_6concepts_fold1['epoch'], val_acc_6concepts_fold1['accuracy'], label='Fold 2')
ax.plot(val_acc_6concepts_fold2['epoch'], val_acc_6concepts_fold2['accuracy'], label='Fold 3')
ax.plot(val_acc_6concepts_fold3['epoch'], val_acc_6concepts_fold3['accuracy'], label='Fold 4')
ax.plot(val_acc_6concepts_fold4['epoch'], val_acc_6concepts_fold4['accuracy'], label='Fold 5')
plt.ylim(50, 100)
plt.xlim(0, 45)
plt.legend()
plt.title('Validation accuracies CW model with 6 concepts', fontsize=15, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (in %)')
plt.savefig('val_accuracies_CWmodel_6concepts.svg', format='svg')
plt.show()


#%%
fig_loss = plt.figure()
ax = plt.axes()

ax.plot(train_loss_6concepts['epoch'], train_loss_6concepts['loss'], label='Training loss')
ax.plot(val_loss_6concepts['epoch'], val_loss_6concepts['loss'], color='orange', label='Validation loss')
plt.ylim(0, 6)
plt.legend()
plt.title('Cross-entropy loss CW model (fold 1)', fontsize=15, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss_CWmodel_6concepts.svg', format='svg')
plt.show()

#%%
train_acc_2concepts_fold0 = pd.read_csv('training_results_performances/training_0_acc_2concepts.csv', header=None)
train_acc_2concepts_fold0.columns = ['epoch', '??', 'accuracy']
train_acc_2concepts_fold1 = pd.read_csv('training_results_performances/training_1_acc_2concepts.csv', header=None)
train_acc_2concepts_fold1.columns = ['epoch', '??', 'accuracy']
train_acc_2concepts_fold2 = pd.read_csv('training_results_performances/training_2_acc_2concepts.csv', header=None)
train_acc_2concepts_fold2.columns = ['epoch', '??', 'accuracy']
train_acc_2concepts_fold3 = pd.read_csv('training_results_performances/training_3_acc_2concepts.csv', header=None)
train_acc_2concepts_fold3.columns = ['epoch', '??', 'accuracy']
train_acc_2concepts_fold4 = pd.read_csv('training_results_performances/training_4_acc_2concepts.csv', header=None)
train_acc_2concepts_fold4.columns = ['epoch', '??', 'accuracy']
val_acc_2concepts_fold0 = pd.read_csv('training_results_performances/validation_0_acc_2concepts.csv', header=None)
val_acc_2concepts_fold0.columns = ['epoch', '??', 'accuracy']
val_acc_2concepts_fold1 = pd.read_csv('training_results_performances/validation_1_acc_2concepts.csv', header=None)
val_acc_2concepts_fold1.columns = ['epoch', '??', 'accuracy']
val_acc_2concepts_fold2 = pd.read_csv('training_results_performances/validation_2_acc_2concepts.csv', header=None)
val_acc_2concepts_fold2.columns = ['epoch', '??', 'accuracy']
val_acc_2concepts_fold3 = pd.read_csv('training_results_performances/validation_3_acc_2concepts.csv', header=None)
val_acc_2concepts_fold3.columns = ['epoch', '??', 'accuracy']
val_acc_2concepts_fold4 = pd.read_csv('training_results_performances/validation_4_acc_2concepts.csv', header=None)
val_acc_2concepts_fold4.columns = ['epoch', '??', 'accuracy']

#%%
fig = plt.figure()
ax = plt.axes()
ax.plot(train_acc_2concepts_fold0['epoch'], train_acc_2concepts_fold0['accuracy'], label='Fold 1')
ax.plot(train_acc_2concepts_fold1['epoch'], train_acc_2concepts_fold1['accuracy'], label='Fold 2')
ax.plot(train_acc_2concepts_fold2['epoch'], train_acc_2concepts_fold2['accuracy'], label='Fold 3')
ax.plot(train_acc_2concepts_fold3['epoch'], train_acc_2concepts_fold3['accuracy'], label='Fold 4')
ax.plot(train_acc_2concepts_fold4['epoch'], train_acc_2concepts_fold4['accuracy'], label='Fold 5')
plt.ylim(60, 100)
plt.legend()
plt.title('Training accuracies CW model with 2 concepts', fontsize=15, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (in %)')
plt.savefig('train_accuracies_CWmodel_2concepts.svg', format='svg')
plt.show()

#%%
fig = plt.figure()
ax = plt.axes()
ax.plot(val_acc_2concepts_fold0['epoch'], val_acc_2concepts_fold0['accuracy'], label='Fold 1')
ax.plot(val_acc_2concepts_fold1['epoch'], val_acc_2concepts_fold1['accuracy'], label='Fold 2')
ax.plot(val_acc_2concepts_fold2['epoch'], val_acc_2concepts_fold2['accuracy'], label='Fold 3')
ax.plot(val_acc_2concepts_fold3['epoch'], val_acc_2concepts_fold3['accuracy'], label='Fold 4')
ax.plot(val_acc_2concepts_fold4['epoch'], val_acc_2concepts_fold4['accuracy'], label='Fold 5')
plt.ylim(60, 100)
plt.legend()
plt.title('Validation accuracies CW model with 2 concepts', fontsize=15, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (in %)')
plt.savefig('val_accuracies_CWmodel_2concepts.svg', format='svg')
plt.show()


#%%
train_acc_1output_fold0 = pd.read_csv('training_results_performances/training_0_acc_oneoutputmodel.csv', header=None)
train_acc_1output_fold0.columns = ['epoch', '??', 'accuracy']
train_acc_1output_fold1 = pd.read_csv('training_results_performances/training_1_acc_oneoutputmodel.csv', header=None)
train_acc_1output_fold1.columns = ['epoch', '??', 'accuracy']
train_acc_1output_fold2 = pd.read_csv('training_results_performances/training_2_acc_oneoutputmodel.csv', header=None)
train_acc_1output_fold2.columns = ['epoch', '??', 'accuracy']
train_acc_1output_fold3 = pd.read_csv('training_results_performances/training_3_acc_oneoutputmodel.csv', header=None)
train_acc_1output_fold3.columns = ['epoch', '??', 'accuracy']
train_acc_1output_fold4 = pd.read_csv('training_results_performances/training_4_acc_oneoutputmodel.csv', header=None)
train_acc_1output_fold4.columns = ['epoch', '??', 'accuracy']
val_acc_1output_fold0 = pd.read_csv('training_results_performances/validation_0_acc_oneoutputmodel.csv', header=None)
val_acc_1output_fold0.columns = ['epoch', '??', 'accuracy']
val_acc_1output_fold1 = pd.read_csv('training_results_performances/validation_1_acc_oneoutputmodel.csv', header=None)
val_acc_1output_fold1.columns = ['epoch', '??', 'accuracy']
val_acc_1output_fold2 = pd.read_csv('training_results_performances/validation_2_acc_oneoutputmodel.csv', header=None)
val_acc_1output_fold2.columns = ['epoch', '??', 'accuracy']
val_acc_1output_fold3 = pd.read_csv('training_results_performances/validation_3_acc_oneoutputmodel.csv', header=None)
val_acc_1output_fold3.columns = ['epoch', '??', 'accuracy']
val_acc_1output_fold4 = pd.read_csv('training_results_performances/validation_4_acc_oneoutputmodel.csv', header=None)
val_acc_1output_fold4.columns = ['epoch', '??', 'accuracy']

#%%
fig = plt.figure()
ax = plt.axes()
ax.plot(train_acc_1output_fold0['epoch'], train_acc_1output_fold0['accuracy'], label='Fold 1')
ax.plot(train_acc_1output_fold1['epoch'], train_acc_1output_fold1['accuracy'], label='Fold 2')
ax.plot(train_acc_1output_fold2['epoch'], train_acc_1output_fold2['accuracy'], label='Fold 3')
ax.plot(train_acc_1output_fold3['epoch'], train_acc_1output_fold3['accuracy'], label='Fold 4')
ax.plot(train_acc_1output_fold4['epoch'], train_acc_1output_fold4['accuracy'], label='Fold 5')
plt.ylim(40, 100)
plt.xlim(0, 30)
plt.legend()
plt.title('Training accuracies CW model with 1 output node', fontsize=15, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (in %)')
plt.savefig('train_accuracies_CWmodel_1output.svg', format='svg')
plt.show()

#%%
fig = plt.figure()
ax = plt.axes()
ax.plot(val_acc_1output_fold0['epoch'], val_acc_1output_fold0['accuracy'], label='Fold 1')
ax.plot(val_acc_1output_fold1['epoch'], val_acc_1output_fold1['accuracy'], label='Fold 2')
ax.plot(val_acc_1output_fold2['epoch'], val_acc_1output_fold2['accuracy'], label='Fold 3')
ax.plot(val_acc_1output_fold3['epoch'], val_acc_1output_fold3['accuracy'], label='Fold 4')
ax.plot(val_acc_1output_fold4['epoch'], val_acc_1output_fold4['accuracy'], label='Fold 5')
plt.ylim(40, 100)
plt.xlim(0, 30)
plt.legend()
plt.title('Validation accuracies CW model with 1 output node', fontsize=15, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (in %)')
plt.savefig('val_accuracies_CWmodel_1output.svg', format='svg')
plt.show()