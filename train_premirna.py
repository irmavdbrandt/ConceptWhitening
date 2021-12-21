import argparse
import shutil
import sys
import time
import torch.nn.parallel
import torch.optim
import torch.utils.data
from plot_functions_cw import *
from helper_functions import *
from PIL import ImageFile
import neptune.new as neptune
from access_keys import neptune_key

run = neptune.init(project='irmavdbrandt/Interpret-rna',
                   api_token=neptune_key,
                   source_files=['train_premirna.py'])

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='PyTorch premiRNA Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='deepmir')
parser.add_argument('--whitened_layers', default='6')
parser.add_argument('--act_mode', default='pool_max')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=56, type=int, metavar='BS', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--concepts', type=str)
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--checkpoint_name', default='', type=str, metavar='PATH', help='path to latest checkpoint '
                                                                                    '(default: none)')
parser.add_argument("--seed", type=int, default=1234, metavar='S', help='randomization seed')
parser.add_argument("--prefix", type=str, required=True, metavar='PFX', help='prefix for logging & checkpoint saving')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('--type_training', type=str, dest='type_training', help='desired type of training (pre-training, '
                                                                            'fine-tuning, CW, evaluation')
parser.add_argument('--foldn_bestmodel', type=str, default=0, help='data fold with best results during training')

os.chdir(sys.path[0])

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')


def main():
    global args, best_acc
    args = parser.parse_args()
    print("args", args)

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # training initialization
    if (args.type_training != 'evaluate') and (args.type_training != 'get_activations'):
        #  specifies which type of training is desired
        if args.type_training == "finetune" or args.type_training == "pretrain":
            # initialize empty model
            model = None
            # if fine-tuning, we want the weights from pre-training
            if args.type_training == 'finetune':
                if args.arch == 'deepmir_v2_bn':
                    # change the model file to the file with the desired pre-training weights
                    model = DeepMir_v2_BN(args, model_file='checkpoints/deepmir_v2_bn/DEEPMIR_v2_pretrain_BN_'
                                                           'checkpoint.pth.tar')
                elif args.arch == 'deepmir_vfinal_bn':
                    # change the model file to the file with the desired pre-training weights
                    # model file below is the one reported in the thesis report
                    model = DeepMir_vfinal_BN(args, model_file='checkpoints/deepmir_vfinal_bn/DEEPMIR_vfinal_'
                                                               'BN_pretrain_checkpoint.pth.tar')
            else:
                # if pre-training, we need to initialize the model with random weights
                if args.arch == 'deepmir_v2_bn':
                    model = DeepMir_v2_BN(args, model_file=None)
                elif args.arch == 'deepmir_vfinal_bn':
                    model = DeepMir_vfinal_BN(args, model_file=None)

            # define optimizer
            optimizer = torch.optim.Adam(model.parameters(), args.lr)

            model = torch.nn.DataParallel(model)
            print('Model architecture: ', model)
            print(f'Number of model parameters: {sum([p.data.nelement() for p in model.parameters()])}')

            # add seeds for reproducibility
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

            # create path links to data directories
            traindir = os.path.join(args.data, 'train')
            test_loader = None
            if args.type_training == 'finetune':
                test_loader = create_data_loader(os.path.join(args.data, 'test'), False)

            # create a balanced data loader for the training set using class weights calculated on the training set
            train_loader = balanced_data_loader(args, traindir)

            # set the best accuracy so far to 0 before starting the training
            best_acc = 0
            for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
                train_loss, train_acc = train_baseline(train_loader, model, criterion, optimizer, epoch)

                # evaluate on validation set
                if args.type_training == 'finetune':
                    acc, val_loss = validate(test_loader, model, criterion)
                else:
                    # if pre-training without test set, evaluate on the training set
                    acc, val_loss = validate(train_loader, model, criterion)

                # neptune logging metrics
                # Log epoch loss
                run[f"training/loss"].log(train_loss)
                # Log epoch accuracy
                run[f"training/acc"].log(train_acc)
                # Log epoch loss
                run[f"validation/loss"].log(val_loss)
                # Log epoch accuracy
                run[f"validation/acc"].log(acc)

                # remember best accuracy (or precision) and save checkpoint for this model
                is_best = acc > best_acc
                best_acc = max(acc, best_acc)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_acc,
                    'optimizer': optimizer.state_dict(),
                }, is_best, args.prefix, fold_n=None)

                print('Best accuracy so far: ', best_acc)
                print('Accuracy current fold: ', acc)

            # stop the neptune logging
            run.stop()

            if args.type_training == "finetune":
                dst = './plot/' + '/' + args.arch + '/'
                if not os.path.exists(dst):
                    os.mkdir(dst)
                # create a data loader for the test set that includes the image paths
                test_loader_with_path = torch.utils.data.DataLoader(
                    ImageFolderWithPaths(os.path.join(args.data, 'test'),
                                         transforms.Compose([transforms.ToTensor(), ])),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

                print("Plot correlation BN layer in non-CW model")
                plot_correlation(dst, args, test_loader_with_path, model, args.whitened_layers)

        elif args.type_training == 'cw':
            args.prefix += '_' + '_'.join(args.whitened_layers.split(','))
            train_accuracies = []
            val_accuracies = []
            correlations = []

            # apply 5-fold cv using the training set splits in the dataset directory
            k_folds = 5

            for fold in range(0, k_folds):
                print(f'now starting fold {fold}')
                # initialize model architecture and possibly weights
                model = None
                if args.arch == 'deepmir_v2_cw':
                    model = DeepMir_v2_Transfer(args, [int(x) for x in args.whitened_layers.split(',')],
                                                model_file='checkpoints/deepmir_v2_bn/DEEPMIR_v2_BN_finetune_'
                                                           'checkpoint.pth.tar')
                elif args.arch == "deepmir_vfinal_cw":
                    model = DeepMir_vfinal_Transfer(args, [int(x) for x in args.whitened_layers.split(',')],
                                                    model_file='checkpoints/deepmir_vfinal_bn/DEEPMIR_'
                                                               'vfinal_BN_finetune_model_best.pth.tar')

                # define optimizer
                optimizer = torch.optim.Adam(model.parameters(), args.lr)

                model = torch.nn.DataParallel(model)
                print('Model architecture: ', model)
                print(f'Number of model parameters: {sum([p.data.nelement() for p in model.parameters()])}')

                # add seeds for reproducibility
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)

                # create path links to data directories
                traindir = os.path.join(args.data, f'train_fold{fold}')
                valdir = os.path.join(args.data, f'val_fold{fold}')
                conceptdir_train = os.path.join(args.data, f'concept_train_fold{fold}')

                # create a balanced data loader for the training set using class weights calculated on the training set
                train_loader = balanced_data_loader(args, traindir)

                # initialize the concept data loader
                concept_loaders = [
                    torch.utils.data.DataLoader(
                        datasets.ImageFolder(os.path.join(conceptdir_train, concept), transforms.Compose([
                            transforms.ToTensor(), ])),
                        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
                    for concept in args.concepts.split(',')
                ]

                # create balanced data loader for the test set using class weights calculated on the test set
                val_loader = create_data_loader(valdir, False)

                # create another data loader for the test set that includes the image paths
                val_loader_with_path = torch.utils.data.DataLoader(
                    ImageFolderWithPaths(valdir, transforms.Compose([transforms.ToTensor(), ])),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

                # neptune parameter configuration
                run['config/dataset/path'] = traindir
                run['config/model'] = type(model).__name__
                run['config/criterion'] = type(criterion).__name__
                run['config/optimizer'] = type(optimizer).__name__
                run['config/lr'] = args.lr
                run['config/batchsize'] = args.batch_size

                print("Start training")
                best_acc = 0
                accuracies = []
                val_losses = []
                train_acc = None
                val_acc = None

                # set settings for early stopping: the model needs to run for 20 epochs before activating the function,
                # initialize an empty counter for the number of epochs without improvement in val loss and set the
                # initial val loss to infinity
                n_epochs_stop = 20
                epochs_no_improve = 0
                min_val_loss = np.Inf

                for epoch in range(args.start_epoch, args.start_epoch + args.epochs):  # 0 used to be args.start_epoch
                    adjust_learning_rate(args, optimizer, epoch)
                    # train for one epoch
                    train_acc, train_loss = train(train_loader, concept_loaders, model, criterion, optimizer, epoch)
                    # evaluate on validation set
                    val_acc, val_loss = validate(val_loader, model, criterion)
                    accuracies.append(val_acc)
                    val_losses.append(val_loss)

                    # Neptune logging
                    # Log fold loss
                    run[f"training/{fold}/loss"].log(train_loss)
                    # Log fold accuracy
                    run[f"training/{fold}/acc"].log(train_acc)
                    # Log fold loss
                    run[f"validation/{fold}/loss"].log(val_loss)
                    # Log fold accuracy
                    run[f"validation/{fold}/acc"].log(val_acc)

                    # remember best accuracy (or precision) and save checkpoint for this model
                    is_best = val_acc > best_acc
                    best_acc = max(val_acc, best_acc)
                    # do not save models before the first 10 epochs (these models tend to not yet have learned any
                    # concepts but do have high accuracy due to the use of pretraining weights)
                    if epoch < args.start_epoch + 5:
                        continue
                    else:
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': model.state_dict(),
                            'best_prec1': best_acc,
                            'optimizer': optimizer.state_dict(),
                        }, is_best, args.prefix, fold_n=fold)

                    print('Best accuracy so far: ', best_acc)
                    print('Accuracy current fold: ', val_acc)

                    # add early stopping criteria: if the loss on the validation set has not improved over the last 10
                    # epochs, stop the training
                    if val_loss < min_val_loss:
                        epochs_no_improve = 0
                        min_val_loss = val_loss

                    else:
                        epochs_no_improve += 1

                    if epoch > args.start_epoch + 10 and epochs_no_improve == n_epochs_stop:
                        print('Early stopping!')
                        print('validation loss not decreased over 20 epochs')
                        break
                    else:
                        continue

                train_accuracies.append(train_acc)
                val_accuracies.append(val_acc)

                print("mean accuracy on training set: ", np.mean(train_accuracies))
                print("std accuracy on training set: ", np.std(train_accuracies))
                print("mean accuracy on validation set: ", np.mean(val_accuracies))
                print("std accuracy on validation set: ", np.std(val_accuracies))
                print('Max accuracy on validation set: ', np.max(val_accuracies), 'index: ', np.argmax(val_accuracies))

                print('Start evaluation for decorrelation and concept learning')
                concept_name = args.concepts.split(',')
                base_dir = './plot/' + '_'.join(concept_name)
                if not os.path.exists(base_dir):
                    os.mkdir(base_dir)
                val_dir = os.path.join(base_dir, 'validation')
                if not os.path.exists(val_dir):
                    os.mkdir(val_dir)
                dir_fold = os.path.join(val_dir, str(fold))
                if not os.path.exists(dir_fold):
                    os.mkdir(dir_fold)

                # create directory where plots will be stored
                dst = './plot/' + '_'.join(args.concepts.split(',')) + '/validation/' + str(fold) + '/' + args.arch \
                      + '/'
                if not os.path.exists(dst):
                    os.mkdir(dst)

                # plot the correlation between the different neurons in the CW layer in a heatmap and collect the
                # correlation values
                mean_correlation = plot_correlation(dst, args, val_loader_with_path, model, args.whitened_layers)
                # Neptune logging: log decorrelation of trained model on fold x
                run[f"validation/correlation"].log(mean_correlation)
                correlations.append(mean_correlation)
                print("mean correlations on validation set: ", np.mean(correlations))
                print("std correlations on validation set: ", np.std(correlations))
                print('Min correlation on validation set: ', np.min(correlations), 'index: ', np.argmin(correlations))

                print("Collect 50 most activated images and plot the top 10")
                plot_concept_top50(args, val_loader_with_path, model, args.whitened_layers, False, args.act_mode, dst)
                plot_top10(args.concepts.split(','), args.whitened_layers, args.type_training, dst)

            # stop the neptune logging
            run.stop()

    # get activations after the cw layer on the complete training and test dataset
    elif args.type_training == 'get_activations':
        # create a data loader for the training set that includes the image paths
        train_loader_with_path = torch.utils.data.DataLoader(
            ImageFolderWithPaths(os.path.join(args.data, 'train'), transforms.Compose([transforms.ToTensor(), ])),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

        # create a data loader for the test set that includes the image paths
        test_loader_with_path = torch.utils.data.DataLoader(
            ImageFolderWithPaths(os.path.join(args.data, 'test'), transforms.Compose([transforms.ToTensor(), ])),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

        model = load_deepmir_model(args, whitened_layer=args.whitened_layers, checkpoint_name=args.checkpoint_name)

        print("Save activation values after CW layer for training set instances")
        get_activations_CWlayer(args, train_loader_with_path, model, args.whitened_layers, args.type_training, 72)
        print("Save activation values after CW layer for test set instances")
        get_activations_CWlayer(args, test_loader_with_path, model, args.whitened_layers, args.type_training, 72)

    elif args.type_training == 'evaluate':
        # create path links to data directories
        testdir = os.path.join(args.data, 'test')
        conceptdir_test = os.path.join(args.data, 'concept_test')

        # create balanced data loader for the test set using class weights calculated on the test set
        test_loader = create_data_loader(testdir, False)

        # create another data loader for the test set that includes the image paths
        test_loader_with_path = torch.utils.data.DataLoader(
            ImageFolderWithPaths(testdir, transforms.Compose([transforms.ToTensor(), ])),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

        model = load_deepmir_model(args, whitened_layer=args.whitened_layers, checkpoint_name=args.checkpoint_name)

        print("Start testing")
        validate(test_loader, model, criterion)
        print("Start Plotting")
        if not os.path.exists('./plot/' + '_'.join(args.concepts.split(','))):
            os.mkdir('./plot/' + '_'.join(args.concepts.split(',')))
        plot_figures(args, model, test_loader_with_path, conceptdir_test)


def create_data_loader(directory, shuffle):
    """
    :param directory: folder (or directory) where data is stored
    :param shuffle: where the data should be shuffled by the loader
    :return: torch data loader object
    """
    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(directory, transforms.Compose([transforms.ToTensor(), ])),
        batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers, pin_memory=False)

    return data_loader


def train(train_loader, concept_loaders, model, criterion, optimizer, epoch):
    """
    :param train_loader: data loader with training images
    :param concept_loaders: data loader with concept images of training set
    :param model: model used for training
    :param criterion: loss function
    :param optimizer: optimizer
    :param epoch: current training epoch
    :return: training script
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input_img, target) in enumerate(train_loader):
        # after 30 images, switch to evaluation mode
        if (i + 1) % 30 == 0:
            model.eval()
        with torch.no_grad():
            # update the gradient matrix G by aligning concepts with the latent space axes
            for concept_index, concept_loader in enumerate(concept_loaders):
                # change to concept aligning mode
                model.module.change_mode(concept_index)
                for j, (X, _) in enumerate(concept_loader):
                    X_var = torch.autograd.Variable(X)
                    model(X_var)
                    break
            model.module.update_rotation_matrix()
            # change to ordinary training mode
            model.module.change_mode(-1)
        # induce training again
        model.train()
        # measure data loading time
        data_time.update(time.time() - end)

        # create autograd variables of the input and target so that hooks (forward and backward) can be used
        # the hooks are used to track the gradient updates in layers that have hooks
        input_var = torch.autograd.Variable(input_img)
        target_var = torch.autograd.Variable(target)
        output = model(input_var)  # compute model predictions_test
        # activate these lines in case of model with 1 predictions_test neuron
        # target_var = target_var.unsqueeze(1)
        # target_var = target_var.float()
        # ###########
        loss = criterion(output, target_var)  # update the loss function
        # measure accuracy and record loss
        [acc] = accuracy(output.data, target, topk=(1,))

        losses.update(loss.data, input_img.size(0))
        top1.update(acc.item(), input_img.size(0))
        # compute gradient and do loss step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Accuracy {top1.val:.3f} ({top1.avg:.3f})')

    print(' * Accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, losses.avg


def validate(test_loader, model, criterion):
    """
    :param test_loader: data loader containing the test/validation set images
    :param model: model used for validation
    :param criterion: loss function
    :return: validation script
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    # initialize empty list for storing predictions
    predictions = []
    with torch.no_grad():
        for i, (input_img, target) in enumerate(test_loader):
            # create autograd variables of the input and target so that hooks (forward and backward) can be used
            # the hooks are used to track the gradient updates in layers that have hooks
            input_var = torch.autograd.Variable(input_img)
            target_var = torch.autograd.Variable(target)
            output = model(input_var)  # compute model predictions_test
            # save the predictions in case we are dealing with the test set (they are used for explainability)
            if (args.type_training == 'evaluate') or (args.type_training == 'activations_tree_train'):
                predictions.append(output.data.detach().numpy())
            # activate these lines in case of model with 1 predictions_test neuron
            # target_var = target_var.unsqueeze(1)
            # target_var = target_var.float()
            # ###########
            loss = criterion(output, target_var)  # update the loss function
            # measure accuracy and record loss
            [acc] = accuracy(output.data, target, topk=(1,))
            losses.update(loss.data, input_img.size(0))
            top1.update(acc.item(), input_img.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(f'Test: [{i}/{len(test_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Accuracy {top1.val:.3f} ({top1.avg:.3f})')

    print(' * Accuracy {top1.avg:.3f}'.format(top1=top1))

    # save the predictions in case we are dealing with the test set (they are used for explainability)
    if args.type_training == 'evaluate':
        dst = './output/predictions/' + '_'.join(args.concepts.split(',')) + '/'
        if not os.path.exists(dst):
            os.mkdir(dst)
        # save predictions for creating decision tree / decision rules
        np.save(dst + 'predictions_test', predictions)
    elif args.type_training == 'get_activations_train':
        dst = './output/predictions/' + '_'.join(args.concepts.split(',')) + '/'
        if not os.path.exists(dst):
            os.mkdir(dst)
        # save predictions for creating decision tree / decision rules
        np.save(dst + 'predictions_train', predictions)

    return top1.avg, losses.avg


def train_baseline(train_loader, model, criterion, optimizer, epoch):
    """
    :param train_loader: data loader with training images
    :param model: model used for training
    :param criterion: loss function
    :param optimizer: optimizer
    :param epoch: current training epoch
    :return: baseline training script used for pretraining and fine-tuning of deepmir model
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input_img, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # create autograd variables of the input and target so that hooks (forward and backward) can be used
        # the hooks are used to track the gradient updates in layers that have hooks
        input_var = torch.autograd.Variable(input_img)
        target_var = torch.autograd.Variable(target)
        output = model(input_var)  # compute model predictions_test
        # activate these lines in case of model with 1 predictions_test neuron
        # target_var = target_var.unsqueeze(1)
        # target_var = target_var.float()
        # ###########
        loss = criterion(output, target_var)  # update the loss function
        # measure accuracy and record loss
        [acc] = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, input_img.size(0))
        top1.update(acc.item(), input_img.size(0))
        # compute gradient and do loss step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Accuracy {top1.val:.3f} ({top1.avg:.3f})')

    return losses.avg, top1.avg


def plot_figures(arguments, model, test_loader_with_path, conceptdir):
    """
    :param arguments: arguments given by user
    :param model: model used for training
    :param test_loader_with_path: data loader with test images (including path)
    :param conceptdir: directory containing concept images from test set
    :return: visualizations (correlation matrix, highly activated images, concept pureness, etc.) of CW results
    """
    concept_name = arguments.concepts.split(',')
    dst = './plot/' + '_'.join(args.concepts.split(',')) + '/' + args.arch + '/'
    if not os.path.exists(dst):
        os.mkdir(dst)

    # if using the ordinary model without cw, just plot the correlation matrix in the BN layer
    if args.arch == "deepmir_v2_cw" or args.arch == "deepmir_vfinal_cw":
        # TODO: add here the pretty labels for the x and y axes of the plots (with newline...)
        # ticklabels_xaxis = ['Large\nterminal\nloop', 'At least\n90% base\npairs and\nwobbles in\nstem',
        #                     'Large\nasymmetric\nbulge\ninstead\nof terminal\nloop', 'Large\nasymmetric\nbulge',
        #                     'U-G-U\nmotif', 'A-U pairs\nmotif']
        ticklabels_xaxis = ['Large asymmetric bulge', 'At least 90% base\npairs and wobbles in\nstem']
        # ticklabels_yaxis = ['Large terminal\nloop', 'At least 90%\nbase pairs and\nwobbles in stem',
        #                     'Large asymmetric\nbulge instead\nof terminal loop', 'Large asymmetric\nbulge',
        #                     'U-G-U motif', 'A-U pairs motif']
        ticklabels_yaxis = ['Large\nasymmetric\nbulge', 'At least\n90% base\npairs and\nwobbles in\nstem']
        print("Plot correlation in CW layer of CW model")
        plot_correlation(dst, args, test_loader_with_path, model, args.whitened_layers)
        print("Collect 50 most activated images and plot the top 10")
        # False is if you want the top50 concept images for the whitened layer and the assigned neuron,
        # otherwise you can say for which layer neuron in that layer you want the top 50
        plot_concept_top50(args, test_loader_with_path, model, args.whitened_layers, False, args.act_mode, dst)
        plot_top10(args.concepts.split(','), args.whitened_layers, args.type_training, dst)
        # use below if you want to get the most activated images for another neuron (and specify which neuron)
        plot_concept_top50(args, test_loader_with_path, model, args.whitened_layers, 4, args.act_mode, dst)
        plot_concept_top50(args, test_loader_with_path, model, args.whitened_layers, 36, args.act_mode, dst)
        print("Plot intra- and inter-concept similarities")
        intra_concept_dot_product_vs_inter_concept_dot_product(args, conceptdir, args.whitened_layers,
                                                               args.concepts.split(','), 'deepmir_vfinal_cw',
                                                               model, ticklabels_xaxis, ticklabels_yaxis)

        print("Plot AUC-concept_purity")
        plot_auc_cw(args, conceptdir, whitened_layers=args.whitened_layers, plot_cpt=concept_name,
                    activation_mode=args.act_mode, concept_labels=ticklabels_xaxis)

        print("Plot receptive field over most activated img")
        saliency_map_concept_cover(args, args.whitened_layers, num_concepts=len(args.concepts.split(',')),
                                   model=model)
        # one below is for checking the other neurons in the layer that are not aligned with concepts
        # nodes variable is for the highly activated neurons one wants to check
        # give the list of neurons from the lowest to the highest first number of the neuron index
        saliency_map_cover_most_activated_neuron(args, args.whitened_layers, 4, [36, 4], model)
        saliency_map_cover_most_activated_neuron(args, args.whitened_layers, 36, [36, 4], model)


def save_checkpoint(state, is_best, prefix, checkpoint_folder='./checkpoints', fold_n=None):
    """
    :param state: model state with weight dictionary
    :param is_best: boolean specifying whether model is the best based on accuracy
    :param prefix: name to be used for stored object
    :param checkpoint_folder: folder where checkpoint needs to be stored
    :param fold_n: current fold in k-fold cross validation
    :return: storage of weights (checkpoint) of model in checkpoint folder
    """
    if args.type_training == 'pretrain' or args.type_training == 'finetune':
        if not os.path.exists(os.path.join(checkpoint_folder, args.arch)):
            os.mkdir(os.path.join(checkpoint_folder, args.arch))
        filename = os.path.join(checkpoint_folder, args.arch, f'{prefix}_checkpoint.pth.tar')
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(checkpoint_folder, args.arch, f'{prefix}_model_best.pth.tar'))
    else:
        concept_name = '_'.join(args.concepts.split(','))
        if not os.path.exists(os.path.join(checkpoint_folder, concept_name)):
            os.mkdir(os.path.join(checkpoint_folder, concept_name))
        filename = os.path.join(checkpoint_folder, concept_name, f'{prefix}_foldn{str(fold_n)}_checkpoint.pth.tar')
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename,
                            os.path.join(checkpoint_folder, concept_name, f'{prefix}_foldn{str(fold_n)}_model_'
                                                                          f'best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(arguments, optimizer, epoch):
    """
    :param arguments: arguments given by user
    :param optimizer: optimizer
    :param epoch: current epoch
    :return: sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    print('old lr', arguments.lr)
    print('start epoch', args.start_epoch)
    lr = arguments.lr * (0.1 ** ((epoch - args.start_epoch) // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('new lr', lr)


def accuracy(output, target, topk=(1,)):
    """
    :param output: model predictions_test (prediction)
    :param target: target value (true)
    :param topk: specification for the number of additional instances that need accuracy calculation (top-k accuracy)
    :return: computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    # for CrossEntropyLoss use below
    _, pred = output.topk(maxk, 1, True, True)
    # in case of 1 predictions_test node and BCEloss use below
    # pred = (predictions_test > 0.5).float()
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def make_weights_for_balanced_classes(images, n_classes):
    """
    :param images: images of dataset
    :param n_classes: number of classes in dataset
    :return: class weight for training data loader
    """
    count = [0] * n_classes
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * n_classes
    N = float(sum(count))
    for i in range(n_classes):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def get_class_weights(images, n_classes):
    """
    :param images: images of dataset
    :param n_classes: number of classes in dataset
    :return: class weights for test data loader
    """
    count = [0] * n_classes
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * n_classes
    N = float(sum(count))
    for i in range(n_classes):
        weight_per_class[i] = N / (2 * float(count[i]))

    return weight_per_class


def balanced_data_loader(arguments, dataset_dir):
    """
    :param arguments: arguments given in training/evaluation initialization
    :param dataset_dir: directory where dataset is stored
    :return: data loader that uses balanced class weights to balance the data that is fed to the model
    """

    dataset = datasets.ImageFolder(dataset_dir, transforms.Compose([transforms.ToTensor(), ]))

    # For unbalanced dataset we create a weighted sampler
    weights = make_weights_for_balanced_classes(dataset.imgs, len(dataset.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    loader = torch.utils.data.DataLoader(dataset, batch_size=arguments.batch_size, shuffle=False,
                                         sampler=sampler, num_workers=arguments.workers, pin_memory=False)

    return loader


if __name__ == '__main__':
    main()
