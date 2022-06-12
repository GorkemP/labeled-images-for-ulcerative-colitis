# Created by Gorkem Polat at 24.02.2021
# contact: polatgorkem@gmail.com

import os
import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import torchvision.transforms as transforms
import time
import wandb
from dataset.ucmayo4 import UCMayo4
from utils import provider
from utils.provider import get_regression_accuracy_with_boundaries, setup_reproducability, get_dataset_mean_and_std, \
    get_batch_size_for_model
import argparse

setup_reproducability(35)

parser = argparse.ArgumentParser(description="Arguments for the training.")

parser.add_argument("--train_dir", type=str, required=True, help="path to training set.")
parser.add_argument("--val_dir", type=str, required=True, help="path to validation set.")
parser.add_argument("--model_name", type=str, default="ResNet18",
                    choices=["ResNet18", "ResNet50", "VGG16_bn", "DenseNet121", "Inception_v3", "MobileNet_v3_large"],
                    help="Name of the CNN architecture.")
parser.add_argument("--optimizer", type=str, choices=["Adam", "SGD"], default="Adam",
                    help="Name of the optimization function.")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.0002, help="learning rate.")
parser.add_argument("-wd", "--weight_decay", type=float, default=0., help="weight decay.")
parser.add_argument("-est", "--early_stopping_threshold", type=int, default=25,
                    help="early stopping threshold to terminate training.")
parser.add_argument("--num_epoch", type=int, default=200, help="Max number of epochs to train.")
parser.add_argument("--use_lrscheduling", choices=["True", "False"], default="True",
                    help="if given, training does not use LR scheduling.")
parser.add_argument("-lrsp", "--LRscheduling_patience", type=int, default=15,
                    help="learning rate scheduling patience to decrease learning rate.")
parser.add_argument("-lrsf", "--LRscheduling_factor", type=float, default=0.2,
                    help="learning rate scheduling scaling factor when decrease learning rate.")
parser.add_argument("--use_pretrained_weights", choices=["True", "False"], default="True",
                    help="if True, weights start from pretrained weights on imagenet dataset.")
parser.add_argument("--enable_wandb", choices=["True", "False"], default="True",
                    help="if True, logs training details into wandb platform. Wandb settings should be performed before using this option.")
args = parser.parse_args()

for k, v in vars(args).items():
    print(k, ":", v)

model_name = args.model_name
batch_size = get_batch_size_for_model(model_name)
optimizer_name = args.optimizer
use_lrscheduling = args.use_lrscheduling == "True"

learning_rate = args.learning_rate
weight_decay = args.weight_decay
best_threshold = 0.0001
num_epoch = args.num_epoch
best_acc = 0
num_worker = 4
early_stop_counter = 0
early_stopping_thresh = args.early_stopping_threshold
LRScheduling_patience = args.LRscheduling_patience
lrs_factor = args.LRscheduling_factor
num_classes = 1
use_multiGPU = False
use_weighted_sampler = True
pretrained_weights = args.use_pretrained_weights == "True"
enable_wandb = args.enable_wandb == "True"

print("\nCreate weights directory for checkpoints!")
dirName = "weights"
try:
    os.makedirs(dirName)
    print("Directory ", dirName, " Created ")
except FileExistsError:
    print("Directory ", dirName, " already exists")

if enable_wandb:
    wandb.init(project="ulcerative-colitis-classification", save_code=True)
    wandb.run.name = os.path.basename(__file__)[:-3] + "_" + wandb.run.name.split("-")[2]
    wandb.run.save()

    config = wandb.config
    config.model = model_name
    config.dataset = "70_15_15"
    config.lr = learning_rate
    config.wd = weight_decay
    config.bs = batch_size
    config.num_worker = num_worker
    config.optimizer = optimizer_name

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

train_dir = args.train_dir
val_dir = args.val_dir

channel_means, channel_stds = get_dataset_mean_and_std(train_dir)
normalize = transforms.Normalize(mean=channel_means,
                                 std=channel_stds)

if model_name == "Inception_v3":
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation((-180, 180)),
                                          transforms.Resize((299, 299)),
                                          transforms.ToTensor(),
                                          normalize])
else:
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation((-180, 180)),
                                          transforms.ToTensor(),
                                          normalize])
train_dataset = UCMayo4(train_dir, transform=train_transform)

if use_weighted_sampler:
    weighted_sampler = provider.weighted_random_sampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                               sampler=weighted_sampler,
                                               num_workers=num_worker,
                                               pin_memory=True)
else:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_worker,
                                               pin_memory=True)

if model_name == "Inception_v3":
    val_transform = transforms.Compose([transforms.Resize((299, 299)),
                                        transforms.ToTensor(),
                                        normalize])
else:
    val_transform = transforms.Compose([transforms.ToTensor(),
                                        normalize])

val_dataset = UCMayo4(val_dir, transform=val_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker,
                                         pin_memory=True)


def train_inception(model, device, train_loader, criterion, optimizer):
    model.train()
    training_loss = 0.0
    correct = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        output, aux_output = model(data)
        loss1 = criterion(output, target)
        loss2 = criterion(aux_output, target)
        loss = loss1 + 0.4 * loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prediction = output.argmax(dim=1, keepdim=True)
        correct += prediction.eq(target.view_as(prediction)).sum().item()

        training_loss += loss.item()

    training_loss /= len(train_loader)
    correct /= len(train_loader.dataset)

    return training_loss, correct


def train(model, device, train_loader, criterion, optimizer):
    model.train()
    training_loss = 0.0
    correct = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device).float()

        output = model(data)
        output.squeeze_(1)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output_classified = get_regression_accuracy_with_boundaries(output, target, [0.5, 1.5, 2.5])
        correct += output_classified.eq(target).sum().item()
        training_loss += loss.item()

    training_loss /= len(train_loader)
    correct /= len(train_loader.dataset)

    return training_loss, correct


def validation(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device).float()

            output = model(data)
            output.squeeze_(1)
            loss = criterion(output, target)

            output_classified = get_regression_accuracy_with_boundaries(output, target, [0.5, 1.5, 2.5])
            correct += output_classified.eq(target).sum().item()
            val_loss += loss.item()

    val_loss /= len(val_loader)
    correct /= len(val_loader.dataset)

    return val_loss, correct


model = provider.initialize_model(model_name, pretrained_weights, num_classes)

if use_multiGPU:
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

model.to(device)

experiment_signature = "R " + model_name + " lr=" + str(learning_rate) + " reg=" + str(weight_decay) + " bs=" + str(
        batch_size)
print("model: " + experiment_signature + " num_worker: " + str(num_worker))

if optimizer_name == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
elif optimizer_name == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
else:
    raise Exception("Undefined optimizer name")
if use_lrscheduling:
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=lrs_factor, patience=LRScheduling_patience,
                                               threshold=best_threshold,
                                               verbose=True)

criterion = nn.MSELoss()
last_epoch = 0
for epoch in range(num_epoch):
    last_epoch = epoch

    start = time.time()
    if model_name == "Inception_v3":
        train_loss, train_accuracy = train_inception(model, device, train_loader, criterion, optimizer)
    else:
        train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer)
    elapsed = time.time() - start

    val_loss, val_accuracy = validation(model, device, val_loader, criterion)
    if use_lrscheduling:
        scheduler.step(val_accuracy)

    print("epoch: {:3.0f}".format(epoch + 1) + " | time: {:3.0f} sec".format(
            elapsed) + " | Average batch-process time: {:4.3f} sec".format(
            elapsed / len(train_loader)) + " | Train acc: {:4.2f}".format(
            train_accuracy * 100) + " | Val acc: {:4.2f}".format(
            val_accuracy * 100) + " | Train loss: {:6.4f}".format(
            train_loss) + " | Val loss: {:6.4f}".format(
            val_loss))

    if enable_wandb:
        wandb.log(
                {"epoch"     : epoch + 1,
                 "lr"        : optimizer.param_groups[0]['lr'],
                 'train loss': train_loss,
                 'val loss'  : val_loss,
                 'train acc' : train_accuracy,
                 'val acc'   : val_accuracy})

    if val_accuracy > best_acc * (1 + best_threshold):
        early_stop_counter = 0
        best_acc = val_accuracy
        print("overwriting the best model!")
        if enable_wandb:
            wandb.run.summary["best accuracy"] = best_acc
        torch.save(model.state_dict(), "weights/best_R_" + model_name + '.pth.tar')
    else:
        early_stop_counter += 1

    if early_stop_counter >= early_stopping_thresh:
        print("Early stopping at: " + str(epoch))
        break

if enable_wandb:
    wandb.run.finish()

print("------ Training finished ------")
print("Best validation set accuracy: " + str(best_acc * 100))
