import os
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim
import numpy as np
from keras.models import load_model
from skimage import io
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from sklearn.metrics import accuracy_score
import tensorflow as tf
import keras
from keras import backend as K
from tensorflow.keras import layers
from torchvision import transforms
import cv2
from PIL import Image

torch.manual_seed(2)
# %%
deepmir_model = load_model('preparation/data_sources/fine_tuned_cnn.h5')
deepmir_model.summary()
weights = deepmir_model.get_weights()


# %%
# convert model from keras to pytorch
class DeepMir(nn.Module):
    def __init__(self):
        super(DeepMir, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(3, 3), stride=(1, 1)),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding=(1, 1)),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                      # nn.Dropout(p=0.25),
                                      nn.BatchNorm2d(48),

                                      nn.Conv2d(in_channels=48, out_channels=60, kernel_size=(3, 3), padding=(1, 1)),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=60, out_channels=60, kernel_size=(3, 3), padding=(1, 1)),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                      # nn.Dropout(p=0.25),
                                      nn.BatchNorm2d(60),

                                      nn.Conv2d(in_channels=60, out_channels=72, kernel_size=(3, 3), padding=(1, 1)),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=72, out_channels=72, kernel_size=(3, 3), padding=(1, 1)),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                      # nn.Dropout(p=0.25))
                                      nn.BatchNorm2d(72))

        self.classifier = nn.Sequential(nn.Linear(2 * 12 * 72, 256),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(256, 2),
                                        nn.Softmax(dim=1)
                                        )

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = self.maxpool1(x)
        # x = self.dropout1(x)
        #
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = self.maxpool2(x)
        # x = self.dropout2(x)
        #
        # x = F.relu(self.conv5(x))
        # x = F.relu(self.conv6(x))
        # x = self.maxpool3(x)
        # x = self.dropout3(x)
        #
        # b, c, h, w = x.size()
        # x = x.view(-1, c * h * w)
        #
        # x = F.relu(self.fc1(x))
        # x = self.dropout4(x)
        # x = F.softmax(self.fc2(x), dim=1)
        x = self.features(x)
        b, c, h, w = x.size()
        x = x.view(-1, c * h * w)
        # x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


torch_model = DeepMir()
print(torch_model)
summary(torch_model, (128, 3, 25, 100))

# %%
# load weights from pretrained pytorch model into fine-tuning model
state_dict = torch.load('checkpoints/deepmir_pretrain.pth')['state_dict']
state_dict_torchmodel = torch_model.state_dict()
new_state_dict = {k: state_dict[k] if k in state_dict.keys() else state_dict_torchmodel[k] for k in
                  state_dict_torchmodel.keys()}
torch_model.load_state_dict(new_state_dict)

# # %%
# load weights from keras to pytorch
# def keras_to_pyt(km, pm):
#     weight_dict = dict()
#     for layer in deepmir_model.layers:
#         if type(layer) is keras.layers.Conv2D:
#             weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (3, 2, 0, 1))
#             weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
#         elif type(layer) is keras.layers.Dense:
#             weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (1, 0))
#             weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
#     pyt_state_dict = pm.state_dict()
#     for key_pytorch, key_keras in zip(pyt_state_dict.keys(), weight_dict.keys()):
#         pyt_state_dict[key_pytorch] = torch.from_numpy(weight_dict[key_keras])
#     pm.load_state_dict(pyt_state_dict)
#     return pm
#
#
# # create model from keras sequential model
# pyt_model = keras_to_pyt(deepmir_model, torch_model)
# print(pyt_model.state_dict())


# %%
# first test whether I get good results on the test set to check whether the pytorch model is equal to the keras one
train_tables = []
test_tables = []
for dirname, _, filenames in os.walk('preparation/datasets/modhsa_original/'):
    for filename in filenames:
        if filename == ".DS_Store":
            continue
        if dirname[37:50] == "concept_train" or dirname[37:49] == "concept_test":
            continue
        else:
            entry = pd.DataFrame([os.path.join(dirname, filename)], columns=['path'])
            if dirname[37:42] == "train":
                entry['class_label'] = dirname[43:44]
                train_tables.append(entry)
            else:
                entry['class_label'] = dirname[42:43]
                test_tables.append(entry)

# collect the data and concept information
train_data_tables = pd.concat(train_tables, ignore_index=True)  # create dataframe from list of tables and reset index
test_data_tables = pd.concat(test_tables, ignore_index=True)  # create dataframe from list of tables and reset index
print(train_data_tables['class_label'].value_counts())
print(test_data_tables['class_label'].value_counts())

# %%
pretrain_tables = []
for dirname, _, filenames in os.walk('preparation/datasets/nonhsa_modmirbase/'):
    for filename in filenames:
        if filename == ".DS_Store":
            continue
        entry = pd.DataFrame([os.path.join(dirname, filename)], columns=['path'])
        entry['class_label'] = dirname[39:40]
        pretrain_tables.append(entry)

# collect the data and concept information
pretrain_data_tables = pd.concat(pretrain_tables,
                                 ignore_index=True)  # create dataframe from list of tables and reset index
print(pretrain_data_tables['class_label'].value_counts())


# %%
def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


# dataset definition
class PremiRNADataset(Dataset):
    # load the dataset
    def __init__(self, dataframe):
        # store the inputs and outputs
        self.X = dataframe['path']
        self.y = dataframe['class_label']
        # normalize the image data
        self.transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        image = io.imread(self.X[idx])
        image_array = np.array(image)
        # image = np.asarray(self.X[idx], dtype=np.uint8)
        img = Image.fromarray(image_array, 'RGB')
        image = add_margin(img, 0, 28, 7, 0, (255, 255, 255))

        image_array = np.array(image)
        image_array = image_array.astype('float32')
        image_array_transposed = np.transpose(image_array, [2, 0, 1])
        tensor_image = torch.from_numpy(image_array_transposed)
        tensor_normalized = self.transform(tensor_image)

        return [tensor_normalized, int(self.y[idx])]


# convert pandas df to pytorch dataset
premirna_data_train = PremiRNADataset(train_data_tables)
premirna_data_test = PremiRNADataset(test_data_tables)
premirna_data_pretrain = PremiRNADataset(pretrain_data_tables)
# create a data loader for train and test sets
train_dl = DataLoader(premirna_data_train, batch_size=128, shuffle=True)
test_dl = DataLoader(premirna_data_test, batch_size=128, shuffle=False)
pretrain_dl = DataLoader(premirna_data_pretrain, batch_size=128, shuffle=True)

# define the optimization
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(torch_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-07, weight_decay=0)
optimizer_pretrain = optim.Adam(torch_model.parameters())


# %%
# train the model
def train_model(train_dl, model):
    # enumerate epochs
    for epoch in range(100):
        print(f'epoch {epoch} starting')
        predictions_train, trues_train = list(), list()

        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            print(inputs[0].shape)
            # clear the gradients
            optimizer_pretrain.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer_pretrain.step()

            yhat = yhat.detach().numpy()
            # save the actual label, label 0 in case the first index is a 1, else 1
            yhat = np.argmax(yhat, axis=1)
            predictions_train.extend(yhat)
            targets = np.array(targets)
            trues_train.extend(targets)

        accuracy = accuracy_score(trues_train, predictions_train)
        print(accuracy)

        if epoch == 99:
            torch.save({
                'state_dict': torch_model.state_dict(),
                'optimizer': optimizer_pretrain.state_dict(),
                'epoch': epoch,
                'best_prec1': accuracy,

            }, 'checkpoints/deepmir_bn.pth')

        torch.save(torch_model, 'checkpoints/deepmir_architecture_bn.pth')
        # if epoch == 39:
        #     torch.save({
        #         'state_dict': torch_model.state_dict(),
        #         'optimizer': optimizer_pretrain.state_dict(),
        #         'epoch': epoch,
        #         'best_prec1': accuracy,
        #
        #     }, 'checkpoints/deepmir_pretrain.pth')


train_model(train_dl, torch_model)
# %%
torch_model.eval()


# evaluate the model
def evaluate_model(test_dl, model):
    predictions, trues = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        # save the actual label, in case there is a 1 on the first index, the label is 0, else it is 1
        yhat = np.argmax(yhat, axis=1)
        # store
        predictions.extend(yhat)
        targets = np.array(targets)
        trues.extend(targets)
    predictions_stacked, trues_stacked = np.vstack(predictions), np.vstack(trues)
    # calculate accuracy
    accuracy = accuracy_score(trues_stacked, predictions_stacked)

    return accuracy, predictions, trues


acc, predictions, trues = evaluate_model(test_dl, torch_model)
print('Accuracy: %.3f' % acc)

# %%
# check out the result with the DeepMir model from keras, this lies around 95%....
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=None)

test_gen = test_datagen.flow_from_dataframe(
    test_data_tables,
    directory=None,
    x_col="path",
    y_col="class_label",
    target_size=(25, 100),
    color_mode="rgb",
    class_mode="binary",
    batch_size=128,
    shuffle=False,
    seed=2,
    validate_filenames=True,
)

predict = deepmir_model.predict_generator(test_gen)
predictions_keras = np.argmax(predict, axis=1)
trues_keras = test_data_tables['class_label'].astype(int)
accuracy = accuracy_score(trues_keras, predictions_keras)

# %%
print(predictions)
print(predictions_keras)

# %%
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in torch_model.state_dict():
    print(param_tensor, "\t", torch_model.state_dict()[param_tensor].size())

# %%
print(torch_model.fc1.weight)
print(weights[14])

# %%
img = test_data_tables.iloc[3]['path']
image_array = io.imread(img)
image_array = np.array(image_array.astype('float32'))
image_array_channelsfirst = np.transpose(image_array, [2, 0, 1])
image_array_channelsfirst_4axes = image_array_channelsfirst[np.newaxis, ...]
image_array_channelsfirst_4axes = torch.tensor(image_array_channelsfirst_4axes)
result = torch_model(image_array_channelsfirst_4axes)
# %%
activation = {}


def get_activation(name):
    def hook(output):
        activation[name] = output.detach()

    return hook


torch_model.conv1.register_forward_hook(get_activation('conv1'))
output = torch_model(image_array_channelsfirst_4axes)
out_val_torch = activation['conv1']

# %% keras predict
img = test_data_tables.iloc[3]['path']
image_array = io.imread(img)
image_array = np.array(image_array.astype('float32'))
image_array_4axes = image_array[np.newaxis, ...]
result_keras = deepmir_model.predict(image_array_4axes)[0]
# %%
OutFunc = K.function([deepmir_model.input], [deepmir_model.layers[1].output])
out_val = OutFunc([image_array_4axes])
# %%
print("Keras result: ", result_keras)
print("Torch result: ", result)

# %%
import torchvision.models as models

resnet = models.__dict__['resnet18'](num_classes=2)
summary(resnet)
# %%
state_dict_resnet = torch.load('checkpoints/palindrome_True/RESNET_PREMIRNA_CPT_WHITEN_TRANSFER_12_checkpoint.pth.tar',
                               map_location='cpu')['state_dict']

# %%
from MODELS.iterative_normalization import IterNormRotation as cw_layer


def conv3x3(in_planes: int, out_planes: int, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=dilation)


class DeepMirResNetv2(nn.Module):
    def __init__(self) -> None:
        super(DeepMirResNetv2, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, kernel_size=3, stride=(1, 1))
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(48, 48)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # self.bn1 = nn.BatchNorm2d(48, track_running_stats=True)
        self.cw1 = cw_layer(48, activation_mode='pool_max')
        self.relu3 = nn.ReLU()

        self.conv3 = conv3x3(48, 60)
        self.relu4 = nn.ReLU()
        self.conv4 = conv3x3(60, 60)
        self.relu5 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.bn2 = nn.BatchNorm2d(60, track_running_stats=True)
        self.relu6 = nn.ReLU()

        self.conv5 = conv3x3(60, 72)
        self.relu7 = nn.ReLU()
        self.conv6 = conv3x3(72, 72)
        self.relu8 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.bn3 = nn.BatchNorm2d(72, track_running_stats=True)
        self.relu9 = nn.ReLU()

        self.linear1 = nn.Linear(2 * 12 * 72, 256)
        self.relu10 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(256, 2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool1(out)
        # out = self.bn1(out)
        out = self.cw1(out)
        out = self.relu3(out)

        out = self.conv3(out)
        out = self.relu4(out)
        out = self.conv4(out)
        out = self.relu5(out)
        out = self.pool2(out)
        out = self.bn2(out)
        out = self.relu6(out)

        out = self.conv5(out)
        out = self.relu7(out)
        out = self.conv6(out)
        out = self.relu8(out)
        out = self.pool3(out)
        out = self.bn3(out)
        out = self.relu9(out)

        out = out.contiguous()
        x = out.view(out.size(0), -1)
        x = self.linear1(x)
        x = self.relu10(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


model = DeepMirResNetv2()

# %%
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

#%%
 # pool of square window of size=3, stride=2
m = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
unm = nn.MaxUnpool2d(kernel_size=2, stride=1, padding=0)
input_conv = torch.randn(256, 256, 3, 3)
input = torch.randn(64, 256, 2, 8)
print(input_conv.shape)
output_stage1 = m(input_conv)
print(output_stage1.shape)
output = unm(output_stage1)
print(output.shape)