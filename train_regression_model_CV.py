import os
import numpy as np
import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import torchvision.transforms as transforms
import time
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
import wandb
from dataset.ucmayo4 import UCMayo4
from utils.metrics import get_mean_sensitivity_specificity
from utils import provider
from utils.provider import get_test_results_regression, get_regression_accuracy_with_boundaries, \
    setup_reproducability, get_dataset_mean_and_std, write_metric_results_to_file, get_batch_size_for_model
from sklearn.metrics import classification_report, precision_recall_fscore_support

setup_reproducability(35)

model_name = "ResNet18"
batch_size = get_batch_size_for_model(model_name)
optimizer_name = "Adam"
use_lrscheduling = True

learning_rate = 0.0002
weight_decay = 0
best_threshold = 0.0001
num_epoch = 200
num_worker = 4
early_stopping_thresh = 25
lr_scheduler_patience = 10
num_classes = 1
real_num_classes = 4
use_multiGPU = False
enable_wandb = True
use_weighted_sampler = True
pretrained_weights = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)


def train_inception(model, device, train_loader, criterion, optimizer):
    model.train()
    training_loss = 0.0
    correct = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device).float()

        output, aux_output = model(data)
        output.squeeze_(1)
        aux_output.squeeze_(1)

        loss1 = criterion(output, target)
        loss2 = criterion(aux_output, target)
        loss = loss1 + 0.4 * loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output_classified = get_regression_accuracy_with_boundaries(output, target, [0.5, 1.5, 2.5])
        correct += output_classified.eq(target).sum().item()
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


def get_test_set_results(id, test_dir, normalize):
    if model_name == "Inception_v3":
        test_transform = transforms.Compose([transforms.Resize((299, 299)),
                                             transforms.ToTensor(),
                                             normalize])
    else:
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             normalize])

    test_dataset = UCMayo4(test_dir, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_worker,
                                              pin_memory=True)

    model = provider.initialize_model(model_name, False, num_classes)
    model.load_state_dict(torch.load("weights/best_R_" + model_name + '_' + str(id) + '.pth.tar'))
    model.to(device)

    y_true, y_pred, r_true, r_pred = get_test_results_regression(model, test_loader, device, [0.5, 1.5, 2.5])

    return y_true, y_pred, r_true, r_pred


def run_experiment(experiment_id: int, train_dir: str, val_dir: str, normalize, best_acc=0, early_stop_counter=0):
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

    print("Standard model is running!")
    model = provider.initialize_model(model_name, pretrained_weights, num_classes)

    if use_multiGPU:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

    model.to(device)

    experiment_signature = "R " + model_name + " ID=" + str(experiment_id) + " lr=" + str(
            learning_rate) + " reg=" + str(weight_decay) + " bs=" + str(
            batch_size)
    print("model: " + experiment_signature + " worker: " + str(num_worker))

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise Exception("Undefined optimizer name")

    if use_lrscheduling:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=lr_scheduler_patience,
                                                   threshold=best_threshold,
                                                   verbose=False)
    criterion = nn.MSELoss()

    for epoch in range(num_epoch):

        if model_name == "Inception_v3":
            train_loss, train_accuracy = train_inception(model, device, train_loader, criterion, optimizer)
        else:
            train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer)
        val_loss, val_accuracy = validation(model, device, val_loader, criterion)
        if use_lrscheduling:
            scheduler.step(val_accuracy)

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
            if enable_wandb:
                wandb.run.summary["best accuracy"] = best_acc
            torch.save(model.state_dict(), "weights/best_R_" + model_name + '_' + str(experiment_id) + '.pth.tar')
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stopping_thresh:
            print("Early stopping at: " + str(epoch))
            break

    print("Experiment: " + str(experiment_id) + ", best validation set accuracy: " + str(best_acc * 100))


experiment_start_time = time.time()
if __name__ == "__main__":
    if enable_wandb:
        group_id = wandb.util.generate_id()

    CV_fold_path = "/home/ws2080/Desktop/data/ucmayo4/cross_validation_folds_splitted_test"
    CV_fold_folders = [x for x in os.listdir(CV_fold_path) if x.startswith("fold")]
    CV_fold_folders = sorted(CV_fold_folders)
    number_of_experiments = len(CV_fold_folders)

    kappa_scores = []
    weighted_kappa_scores = []
    accuracies = []
    sensitivities = []
    specificities = []

    macro_precisions = []
    macro_recalls = []
    macro_f1s = []

    class_precisions = np.zeros([number_of_experiments, real_num_classes])
    class_recalls = np.zeros([number_of_experiments, real_num_classes])
    class_f1s = np.zeros([number_of_experiments, real_num_classes])

    precisions_r = []
    recalls_r = []
    f1s_r = []

    kappa_scores_r = []
    accuracies_r = []
    sensitivities_r = []
    specificities_r = []

    for i in range(number_of_experiments):
        id = i + 1
        print("\n--------------> Starting Fold " + str(id))
        if enable_wandb:
            wandb.init(project="ulcerative-colitis-classification",
                       group=model_name + "_CV_R" + "_" + group_id,
                       save_code=True,
                       reinit=True)
            wandb.run.name = "epoch_" + str(id)
            wandb.run.save()

            config = wandb.config
            config.exp = os.path.basename(__file__)[:-3]
            config.model = model_name
            config.dataset = "final_dataset"
            config.lr = learning_rate
            config.wd = weight_decay
            config.bs = batch_size
            config.num_worker = num_worker
            config.optimizer = optimizer_name

        train_dir = os.path.join(CV_fold_path, CV_fold_folders[i], "train")
        val_dir = os.path.join(CV_fold_path, CV_fold_folders[i], "val")
        test_dir = os.path.join(CV_fold_path, "test")

        channel_means, channel_stds = get_dataset_mean_and_std(train_dir)
        normalize = transforms.Normalize(mean=channel_means,
                                         std=channel_stds)

        run_experiment(id, train_dir, val_dir, normalize)
        y_true, y_pred, r_true, r_pred = get_test_set_results(id, test_dir, normalize)

        prf1_4classes = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1, 2, 3])
        prf1_remission = precision_recall_fscore_support(r_true, r_pred, average="binary")

        cm_4class = confusion_matrix(y_true, y_pred)
        cm_remission = confusion_matrix(r_true, r_pred)

        class_precisions[i] = prf1_4classes[0]
        macro_precision = prf1_4classes[0].mean()
        macro_precisions.append(macro_precision)

        class_recalls[i] = prf1_4classes[1]
        macro_recall = prf1_4classes[1].mean()
        macro_recalls.append(macro_recall)

        class_f1s[i] = prf1_4classes[2]
        macro_f1 = prf1_4classes[2].mean()
        macro_f1s.append(macro_f1)

        # 4-class analysis
        all_kappa_score = cohen_kappa_score(y_true, y_pred)
        kappa_scores.append(all_kappa_score)

        all_kappa_score_weighted = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        weighted_kappa_scores.append(all_kappa_score_weighted)

        accuracy = accuracy_score(y_true, y_pred)
        accuracies.append(accuracy)

        mean_sensitivity, mean_specificity = get_mean_sensitivity_specificity(y_true, y_pred)
        sensitivities.append(mean_sensitivity), specificities.append(mean_specificity)

        # Remission analysis
        remission_kappa_score = cohen_kappa_score(r_true, r_pred)
        kappa_scores_r.append(remission_kappa_score)

        accuracy_r = accuracy_score(r_true, r_pred)
        accuracies_r.append(accuracy_r)

        precisions_r.append(prf1_remission[0])
        recalls_r.append(prf1_remission[1])
        f1s_r.append(prf1_remission[2])

        cr_r = classification_report(r_true, r_pred, output_dict=True)
        sensitivities_r.append(cr_r["0"]["recall"]), specificities_r.append(cr_r["1"]["recall"])

        if enable_wandb:

            wandb.run.summary["m_precision"] = macro_precision
            wandb.run.summary["m_recall"] = macro_recall
            wandb.run.summary["m_f1"] = macro_f1

            wandb.run.summary["kappa"] = all_kappa_score
            wandb.run.summary["qw_kappa"] = all_kappa_score_weighted

            wandb.run.summary["accuracy"] = accuracy
            wandb.run.summary["m_sensitivity"] = mean_sensitivity
            wandb.run.summary["m_specificity"] = mean_specificity

            wandb.run.summary["remission_accuracy"] = accuracy_r
            wandb.run.summary["remission_kappa"] = remission_kappa_score

            wandb.run.summary["remission_precision"] = prf1_remission[0]
            wandb.run.summary["remission_recall"] = prf1_remission[1]
            wandb.run.summary["remission_f1"] = prf1_remission[2]

            wandb.run.summary["remission_sensitivity"] = cr_r["0"]["recall"]
            wandb.run.summary["remission_specificity"] = cr_r["1"]["recall"]

            if id == number_of_experiments:
                write_metric_results_to_file(wandb.run.dir, accuracies, kappa_scores, weighted_kappa_scores,
                                             sensitivities, specificities,
                                             macro_precisions,
                                             macro_recalls, macro_f1s, class_precisions, class_recalls, class_f1s,
                                             accuracies_r, kappa_scores_r, sensitivities_r, specificities_r,
                                             precisions_r,
                                             recalls_r, f1s_r)
            wandb.run.finish()
