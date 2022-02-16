# Created by Gorkem Polat at 12.02.2022
# contact: polatgorkem@gmail.com

import torch
import torchvision.transforms as transforms
from dataset.ucmayo4 import UCMayo4
from utils.metrics import get_mean_sensitivity_specificity
from utils.provider import get_test_results_regression, get_dataset_mean_and_std
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, accuracy_score
from utils import provider
import argparse

parser = argparse.ArgumentParser(description="Arguments for the inference.")

parser.add_argument("--train_dir", type=str, required=True, help="path to training set.")
parser.add_argument("--test_dir", type=str, required=True, help="path to validation set.")
parser.add_argument("--model_name", type=str, required=True,
                    choices=["ResNet18", "ResNet50", "VGG16_bn", "DenseNet121", "Inception_v3", "MobileNet_v3_large"],
                    help="Name of the CNN architecture.")
parser.add_argument("--checkpoint", type=str, required=True, help="path to checkpoint file (pretrained weights).")


args = parser.parse_args()

num_worker = 4
state_dict_name = args.checkpoint
model_name = args.model_name

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

train_dir = args.train_dir
test_dir = args.test_dir

channel_means, channel_stds = get_dataset_mean_and_std(train_dir)
normalize = transforms.Normalize(mean=channel_means,
                                 std=channel_stds)

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

model = provider.initialize_model(model_name, False, 1)
model.load_state_dict(torch.load(state_dict_name))
model.to(device)

y_true, y_pred, r_true, r_pred = get_test_results_regression(model, test_loader, device, [0.5, 1.5, 2.5])

QWK_kappa_score = cohen_kappa_score(y_true, y_pred, weights="quadratic")
cm_all = confusion_matrix(y_true, y_pred)
provider.plot_confusion_matrix(cm_all, ["Mayo 0", "Mayo 1", "Mayo 2", "Mayo 3"],
                               "confusion matrix - " + state_dict_name, normalize=False)
accuracy = accuracy_score(y_true, y_pred)
mean_sensitivity, mean_specificity = get_mean_sensitivity_specificity(y_true, y_pred)

print("------------ Classification Report (All) ------------\n")
print("Average Accuracy: {:0.4f}".format(accuracy))
print("QWK score: " + str(QWK_kappa_score))
print("Average Sensitivity: {:0.4f}".format(mean_sensitivity))
print("Average Specificity: {:0.4f}".format(mean_specificity))

remission_kappa_score = cohen_kappa_score(r_true, r_pred)
cm_remission = confusion_matrix(r_true, r_pred)
cr_remission = classification_report(r_true, r_pred, target_names=["Remission", "Non Remission"], output_dict=True)
provider.plot_confusion_matrix(cm_remission, ["Remission", "Non Remission"], "confusion matrix - " + state_dict_name,
                               normalize=False)

print("------------ Classification Report (Remission) ------------\n")
print("Average Accuracy: {:0.4f}".format(cr_remission["accuracy"]))
print("Remission kappa score: " + str(remission_kappa_score))
print("Average Sensitivity: {:0.4f}".format(cr_remission["Remission"]["recall"]))
print("Average Specificity: {:0.4f}".format(cr_remission["Non Remission"]["recall"]))