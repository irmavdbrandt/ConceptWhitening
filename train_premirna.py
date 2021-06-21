import argparse
import shutil
import sys
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from plot_functions import *
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch premiRNA Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='deepmir_cw_bn',
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: deepmir)')
parser.add_argument('--whitened_layers', default='6')
parser.add_argument('--act_mode', default='pool_max')
parser.add_argument('--depth', default=8, type=int, metavar='D', help='model depth')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=56, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--concepts', type=str, required=True)
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument("--seed", type=int, default=1234, metavar='BS', help='input batch size for training (default: 64)')
parser.add_argument("--prefix", type=str, required=True, metavar='PFX', help='prefix for logging & checkpoint saving')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluation only')

os.chdir(sys.path[0])

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')


def main():
    global args, best_prec1
    args = parser.parse_args()
    print("args", args)
    args.prefix += '_' + '_'.join(args.whitened_layers.split(','))

    # add seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # initialize model architecture and possibly weights
    model = None
    if args.arch == "resnet_cw":
        if args.depth == 18:
            model = ResidualNetTransfer(2, args, [int(x) for x in args.whitened_layers.split(',')], arch='resnet18',
                                        layers=[2, 2, 2, 2],
                                        model_file='checkpoints/resnet_premirna_checkpoints'
                                                   '/RESNET_PREMIRNA_BN_checkpoint.pth.tar')
    elif args.arch == "resnet_bn":
        model = ResidualNetBN(2, args, arch='resnet18', layers=[2, 2, 2, 2],
                              model_file='checkpoints/resnet_premirna_checkpoints/RESNET_PREMIRNA_PRETRAIN_checkpoint'
                                         '.pth.tar')
    elif args.arch == "deepmir_cw":
        model = DeepMirTransfer(args, [int(x) for x in args.whitened_layers.split(',')], arch='deepmir_cw',
                                model_file='checkpoints/deepmir_related_checkpoints/nonCW_training/deepmir_cw.pth')
    elif args.arch == "deepmir_bn":
        model = DeepMirBN(args, arch='deepmir_bn', model_file=None)
    elif args.arch == "deepmir_cw_bn":
        model = DeepMirTransfer(args, [int(x) for x in args.whitened_layers.split(',')], arch='deepmir_cw',
                                model_file='checkpoints/deepmir_related_checkpoints/DEEPMIR_PREMIRNA_BN_checkpoint_'
                                           'new.pth.tar')
        # model = DeepMirTransfer(args, [int(x) for x in args.whitened_layers.split(',')], arch='deepmir_cw')
    elif args.arch == "deepmir_resnet_bn":
        model = DeepMirResNetBN(args, arch='deepmir_resnet_bn', model_file='checkpoints/resnet_premirna_checkpoints'
                                                                           '/DEEPMIR_RESNET_PREMIRNA_PRETRAIN_'
                                                                           'checkpoint.pth.tar')
    elif args.arch == 'deepmir_resnet_bn_v2':
        model = DeepMirResNetBNv2(args, arch='deepmir_resnet_bn_v2', model_file='checkpoints/resnet_deepmir_v2/'
                                                                                'DEEPMIR_RESNET_PREMIRNA_v2_pretrain_BN_1_'
                                                                                'checkpoint.pth.tar')
    elif args.arch == 'deepmir_resnet_cw_v2':
        model = DeepMirResNetTransferv2(args, [int(x) for x in args.whitened_layers.split(',')],
                                        arch='deepmir_resnet_cw_v2', model_file='checkpoints/resnet_deepmir_v2/'
                                                                                'DEEPMIR_RESNET_PREMIRNA_v2_BN_1'
                                                                                '_checkpoint.pth.tar')
    elif args.arch == "deepmir_resnet_cw":
        model = DeepMirResNetTransfer(args, [int(x) for x in args.whitened_layers.split(',')], arch='deepmir_resnet_cw',
                                      model_file='checkpoints/DEEPMIR_RESNET_PREMIRNA_BN_2_checkpoint.pth.tar')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    model = torch.nn.DataParallel(model)
    print('Model architecture: ', model)
    print(f'Number of model parameters: {sum([p.data.nelement() for p in model.parameters()])}')

    cudnn.benchmark = True

    # create path links to data directories
    traindir = os.path.join(args.data, 'train')
    testdir = os.path.join(args.data, 'test')
    conceptdir_train = os.path.join(args.data, 'concept_train')
    conceptdir_test = os.path.join(args.data, 'concept_test')

    # create a balanced data loader for the training set using class weights calculated on the training set
    train_loader = balanced_data_loader(args, traindir)

    # initialize the concept data loader
    concept_loaders = [
        torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(conceptdir_train, concept), transforms.Compose([
                transforms.ToTensor(),
            ])),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
        for concept in args.concepts.split(',')
    ]

    # create balanced data loader for the test set using class weights calculated on the test set
    test_dataset = datasets.ImageFolder(testdir, transforms.Compose([
        transforms.ToTensor(),
    ]))
    args.test_weights = get_class_weights(test_dataset.imgs, len(test_dataset.classes))
    print(args.test_weights)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=False)

    # create another data loader for the test set that includes the image paths
    test_loader_with_path = torch.utils.data.DataLoader(
        ImageFolderWithPaths(testdir, transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    # training initialization
    if not args.evaluate:
        print("Start training")
        epoch = None
        best_prec1 = 0   # best top-1 precision = top-1 accuracy
        for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
            adjust_learning_rate(optimizer, epoch)
            # train for one epoch
            if args.arch == "resnet_cw" or args.arch == "resnet_original" or args.arch == "deepmir_cw" or \
                    args.arch == "deepmir_cw_bn" or args.arch == "deepmir_bn" or \
                    args.arch == "resnet_bn" or args.arch == "deepmir_resnet_bn" or args.arch == "deepmir_resnet_cw" \
                    or args.arch == "deepmir_resnet_bn_v2" or args.arch == "deepmir_resnet_cw_v2":
                train(train_loader, concept_loaders, model, criterion, optimizer, epoch)
            elif args.arch == "resnet_baseline":
                train_baseline(train_loader, concept_loaders, model, criterion, optimizer, epoch)
            # evaluate on validation set
            prec1 = validate(test_loader, model, criterion)

            # remember best prec-1 and save checkpoint for this model
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.prefix)
        print(best_prec1)
        validate(test_loader, model, criterion)
    else:
        if args.arch == "deepmir_resnet_cw":
            model = load_deepmir_resnet_model(args, arch=args.arch, whitened_layer=args.whitened_layers)
        elif args.arch == "deepmir_resnet_bn":
            model = load_deepmir_resnet_bn_model(args, whitened_layer=args.whitened_layers)
        elif args.arch == "resnet_cw":
            model = load_resnet_model(args, checkpoint_folder="./checkpoints", whitened_layer=args.whitened_layers)
        elif args.arch == "deepmir_resnet_cw_v2":
            model = load_deepmir_resnet_cw_v2_model(args, checkpoint_folder="./checkpoints",
                                                    whitened_layer=args.whitened_layers)
        elif args.arch == "deepmir_resnet_bn_v2":
            model = load_deepmir_resnet_v2_bn_model(args, whitened_layer=args.whitened_layers)

        # cannot invoke validation before the concept importance is calculated (because that uses a backward pass..)
        print("Start Plotting")
        if not os.path.exists('./plot/' + '_'.join(args.concepts.split(','))):
            os.mkdir('./plot/' + '_'.join(args.concepts.split(',')))
        if len(args.whitened_layers) > 1:
            whitened_layers = [int(x) for x in args.whitened_layers.split(',')]
            for layer in whitened_layers:
                concept_gradient_importance(args, test_loader, layer, arch=args.arch, num_classes=2, model=model)
        else:
            concept_gradient_importance(args, test_loader, layer=args.whitened_layers, arch=args.arch, num_classes=2,
                                        model=model)

        print("Start testing")
        validate(test_loader, model, criterion)
        plot_figures(args, model, test_loader_with_path, train_loader, concept_loaders, conceptdir_test, test_loader,
                     criterion)

        # don't know what to do with the function below?
        # get_representation_distance_to_center(args, test_loader, args.whitened_layers, arch='deepmir_resnet_cw',
        #                                       model=model)


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

    for i, (input, target) in enumerate(train_loader):
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
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        output = model(input_var)  # compute model output
        loss = criterion(output, target_var)  # update the loss function
        # measure accuracy and record loss
        [prec1] = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, input.size(0))
        top1.update(prec1.item(), input.size(0))
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
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')


def validate(test_loader, model, criterion):
    """
    :param test_loader: data loader containing the test set images
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
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            # create autograd variables of the input and target so that hooks (forward and backward) can be used
            # the hooks are used to track the gradient updates in layers that have hooks
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            output = model(input_var)  # compute model output
            loss = criterion(output, target_var)  # update the loss function
            # measure accuracy and record loss
            [prec1] = accuracy(output.data, target, topk=(1,))
            losses.update(loss.data, input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(f'Test: [{i}/{len(test_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


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

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # create autograd variables of the input and target so that hooks (forward and backward) can be used
        # the hooks are used to track the gradient updates in layers that have hooks
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        output = model(input_var)  # compute model output
        loss = criterion(output, target_var)  # update the loss function
        # measure accuracy and record loss
        [prec1] = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, input.size(0))
        top1.update(prec1.item(), input.size(0))
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
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')


def plot_figures(args, model, test_loader_with_path, train_loader, concept_loaders, conceptdir, test_loader,
                 criterion):
    """
    :param args: arguments given by user
    :param model: model used for training
    :param test_loader_with_path: data loader with test images (including path)
    :param train_loader: data loader for training images
    :param concept_loaders: data loader with concept images from training set
    :param conceptdir: directory containing concept images from test set
    :param test_loader: data loader with test images (without path)
    :param criterion: loss function
    :return: visualizations (correlation matrix, highly activated images, concept pureness, etc.) of CW results
    """
    concept_name = args.concepts.split(',')

    if not os.path.exists('./plot/' + '_'.join(concept_name)):
        os.mkdir('./plot/' + '_'.join(concept_name))

    if args.arch == "deepmir_resnet_bn" or args.arch == "deepmir_resnet_bn_v2":
        print("Plot correlation")
        plot_correlation_BN(args, test_loader_with_path, model, layer=args.whitened_layers)
    elif args.arch == "deepmir_resnet_cw" or args.arch == "resnet_cw" or args.arch == "deepmir_resnet_cw_v2":
        if len(args.whitened_layers) > 1:
            # split the layers in the whitened_layers string to get the individual layers
            whitened_layers = [int(x) for x in args.whitened_layers.split(',')]
            for layer in whitened_layers:
                print("Plot correlation")
                plot_correlation(args, test_loader_with_path, model, layer=str(layer))
                print("Plot intra- and inter-concept similarities")
                intra_concept_dot_product_vs_inter_concept_dot_product(args, conceptdir, str(layer),
                                                                       plot_cpt=args.concepts.split(','),
                                                                       arch='deepmir_resnet_cw',
                                                                       model=model)
                print("Plot concept importance for overall classifier")
                concept_permutation_importance(args, test_loader, layer, criterion, arch=args.arch,
                                               num_concepts=len(args.concepts.split(',')), model=model)
            print("Plot top50 activated images")
            # False is if you want the top50 concept images for the whitened layer and the assigned neuron,
            # otherwise you can say for which layer neuron in that layer you want the top 50
            plot_concept_top50(args, test_loader_with_path, model, args.whitened_layers, False,
                               activation_mode=args.act_mode)
            plot_top10(args, plot_cpt=args.concepts.split(','), layer=args.whitened_layers)
            print("Plot 2d slice of representation")
            plot_concept_representation(args, test_loader_with_path, model, args.whitened_layers,
                                        plot_cpt=[concept_name[0], concept_name[1]], activation_mode=args.act_mode)
            print("Plot trajectory")
            plot_trajectory(args, test_loader_with_path, args.whitened_layers, plot_cpt=[concept_name[0],
                                                                                         concept_name[1]], model=model)
            print("Plot AUC-concept_purity")
            plot_auc_cw(args, conceptdir, whitened_layers=args.whitened_layers, plot_cpt=concept_name,
                        activation_mode=args.act_mode)
            print("AUC plotting")
            plot_auc(args, plot_cpt=concept_name)

        else:
            print("Plot correlation")
            plot_correlation(args, test_loader_with_path, model, layer=args.whitened_layers)
            # plot_concept_top50(args, test_loader_with_path, model, args.whitened_layers, 10,
            #                    activation_mode=args.act_mode)
            # plot_top10(args, plot_cpt=args.concepts.split(','), layer=args.whitened_layers)
            print("Plot top50 activated images")
            # False is if you want the top50 concept images for the whitened layer and the assigned neuron,
            # otherwise you can say for which layer neuron in that layer you want the top 50
            plot_concept_top50(args, test_loader_with_path, model, args.whitened_layers, False,
                               activation_mode=args.act_mode)
            plot_top10(args, plot_cpt=args.concepts.split(','), layer=args.whitened_layers)
            print("Plot 2d slice of representation")
            plot_concept_representation(args, test_loader_with_path, model, args.whitened_layers,
                                        plot_cpt=[concept_name[0], concept_name[1]], activation_mode=args.act_mode)
            print("Plot intra- and inter-concept similarities")
            intra_concept_dot_product_vs_inter_concept_dot_product(args, conceptdir, args.whitened_layers,
                                                                   plot_cpt=args.concepts.split(','),
                                                                   arch='deepmir_resnet_cw',
                                                                   model=model)

            print("Plot concept importance for overall classifier")
            concept_permutation_importance(args, test_loader, args.whitened_layers, criterion, arch=args.arch,
                                           num_concepts=len(args.concepts.split(',')), model=model)

            # print("Plot receptive field over most activated img")
            # saliency_map_concept_cover_2(args, test_loader, args.whitened_layers, arch=args.arch,
            #                              dataset=None, num_concepts=len(args.concepts.split(',')), model=model)


def save_checkpoint(state, is_best, prefix, checkpoint_folder='./checkpoints'):
    """
    :param state: model state with weight dictionary
    :param is_best: boolean specifying whether model is the best based on accuracy
    :param prefix: name to be used for stored object
    :param checkpoint_folder: folder where checkpoint needs to be stored
    :return: storage of weights (checkpoint) of model in checkpoint folder
    """
    concept_name = '_'.join(args.concepts.split(','))
    if not os.path.exists(os.path.join(checkpoint_folder, concept_name)):
        os.mkdir(os.path.join(checkpoint_folder, concept_name))
    filename = os.path.join(checkpoint_folder, concept_name, '%s_checkpoint.pth.tar' % prefix)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_folder, concept_name, '%s_model_best.pth.tar' % prefix))


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


def adjust_learning_rate(optimizer, epoch):
    """
    :param optimizer: optimizer
    :param epoch: current epoch
    :return: sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """
    :param output: model output (prediction)
    :param target: target value (true)
    :param topk: specification for the number of additional instances that need accuracy calculation (top-k accuracy)
    :return: computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
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


def balanced_data_loader(args, dataset_dir):
    """
    :param args: arguments given in training/evaluation initialization
    :param dataset_dir: directory where dataset is stored
    :return: data loader that uses balanced class weights to balance the data that is fed to the model
    """
    dataset = datasets.ImageFolder(dataset_dir, transforms.Compose([
        transforms.ToTensor(),
    ]))

    # For unbalanced dataset we create a weighted sampler
    weights = make_weights_for_balanced_classes(dataset.imgs, len(dataset.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         sampler=sampler, num_workers=args.workers, pin_memory=False)

    return loader


if __name__ == '__main__':
    main()

