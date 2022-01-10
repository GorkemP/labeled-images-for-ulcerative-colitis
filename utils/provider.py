import numpy as np
import random
import cv2
import os
import matplotlib.pyplot as plt
import itertools
import torch
import time
from scipy.stats import ranksums
from torch import save

from dataset.ucmayo4 import UCMayo4
from math import sqrt
from statistics import mean, pstdev
import numpy as np


def get_batch_size_for_model(model_name=""):
    if model_name == "ResNet18":
        batch_size = 64
    elif model_name == "ResNet50":
        batch_size = 32
    elif model_name == "VGG16_bn":
        batch_size = 12
    elif model_name == "DenseNet121":
        batch_size = 16
    elif model_name == "Inception_v3":
        batch_size = 32
    elif model_name == "mobilenet_v3_large":
        batch_size = 32
    else:
        batch_size = 16

    return batch_size


def plot_confusion_matrix_and_save(cm,
                                   target_names,
                                   path,
                                   title='Confusion matrix',
                                   cmap=None,
                                   normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    # plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.1 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(path)
    plt.show()


def save_confusion_matrix(cm,
                          target_names,
                          path,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    # plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.1 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(path)
    # plt.show()
    plt.close()


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def plot_confusion_matrix_TR(cm,
                             target_names,
                             title='Confusion matrix',
                             cmap=None,
                             normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin\ndoğruluk={:0.4f}; hata={:0.4f}'.format(accuracy, misclass))
    plt.show()


def plot_confusion_matrix_2(cm,
                            target_names,
                            title='Confusion matrix',
                            cmap=None,
                            normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:4.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel("Predicted label")
    plt.show()


def plot_confusion_matrix_2_and_save(cm,
                                     target_names,
                                     path,
                                     title='Confusion matrix',
                                     save_dpi=600,
                                     cmap=None,
                                     normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(9, 7), dpi=save_dpi)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:4.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel("Predicted label\naccuracy={:0.4f}; misclass={:0.4f}".format(accuracy, misclass))
    plt.show()
    plt.savefig(path, dpi=save_dpi)


def weighted_random_sampler(dataset):
    class_sample_counts = [0 for x in range(dataset.number_of_class)]
    train_targets = []

    for x in dataset:
        train_targets.append(x[1])
        class_sample_counts[x[1]] += 1

    weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    samples_weights = weights[train_targets]

    sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weights,
                                                     num_samples=len(samples_weights))
    return sampler


def initialize_model(model_name, pretrained, num_classes):
    import torch
    import torchvision.models as models

    model = None

    if model_name == "VGG16_bn":
        if pretrained:
            model = models.vgg16_bn(pretrained=True)
        else:
            model = models.vgg16_bn()
        in_features = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(in_features, num_classes)

    elif model_name == "ResNet18":
        if pretrained:
            model = models.resnet18(pretrained=True)
        else:
            model = models.resnet18()
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)

    elif model_name == "ResNet50":
        if pretrained:
            model = models.resnet50(pretrained=True)
        else:
            model = models.resnet50()
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)

    elif model_name == "ResNet152":
        if pretrained:
            model = models.resnet152(pretrained=True)
        else:
            model = models.resnet152()
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)

    elif model_name == "DenseNet121":
        if pretrained:
            model = models.densenet121(pretrained=True)
        else:
            model = models.densenet121()
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, num_classes)

    elif model_name == "Inception_v3":
        if pretrained:
            model = models.inception_v3(pretrained=True, transform_input=False)
        else:
            model = models.inception_v3(transform_input=False)
        in_features = model.fc.in_features
        aux_in_features = model.AuxLogits.fc.in_features

        model.AuxLogits.fc = torch.nn.Linear(aux_in_features, num_classes)
        model.fc = torch.nn.Linear(in_features, num_classes)

    elif model_name == "mobilenet_v3_large":
        if pretrained:
            model = models.mobilenet_v3_large(pretrained=True)
        else:
            model = models.mobilenet_v3_large()
        in_features = model.classifier[3].in_features
        model.classifier[3] = torch.nn.Linear(in_features, num_classes)

    else:
        print("Invalid model name!")
        exit()

    return model

def initialize_corn_model(model_name, pretrained, num_classes):
    import torch
    import torchvision.models as models

    model = None

    if model_name == "VGG16_bn":
        if pretrained:
            model = models.vgg16_bn(pretrained=True)
        else:
            model = models.vgg16_bn()
        in_features = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(in_features, num_classes-1)

    elif model_name == "ResNet18":
        if pretrained:
            model = models.resnet18(pretrained=True)
        else:
            model = models.resnet18()
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes-1)

    elif model_name == "ResNet50":
        if pretrained:
            model = models.resnet50(pretrained=True)
        else:
            model = models.resnet50()
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes-1)

    elif model_name == "ResNet152":
        if pretrained:
            model = models.resnet152(pretrained=True)
        else:
            model = models.resnet152()
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes-1)

    elif model_name == "DenseNet121":
        if pretrained:
            model = models.densenet121(pretrained=True)
        else:
            model = models.densenet121()
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, num_classes-1)

    elif model_name == "Inception_v3":
        if pretrained:
            model = models.inception_v3(pretrained=True, transform_input=False)
        else:
            model = models.inception_v3(transform_input=False)
        in_features = model.fc.in_features
        aux_in_features = model.AuxLogits.fc.in_features

        model.AuxLogits.fc = torch.nn.Linear(aux_in_features, num_classes-1)
        model.fc = torch.nn.Linear(in_features, num_classes-1)

    elif model_name == "mobilenet_v3_large":
        if pretrained:
            model = models.mobilenet_v3_large(pretrained=True)
        else:
            model = models.mobilenet_v3_large()
        in_features = model.classifier[3].in_features
        model.classifier[3] = torch.nn.Linear(in_features, num_classes-1)

    else:
        print("Invalid model name!")
        exit()

    return model

def label_from_logits_corn(logits):
    probas = torch.sigmoid(logits)
    probas = torch.cumprod(probas, dim=1)
    predict_levels = probas > 0.5
    predicted_labels = torch.sum(predict_levels, dim=1)
    return predicted_labels

def get_remission_test_results(model, data_loader, device):
    model.eval()

    y_true = []
    y_probs = []
    y_pred = []

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            y_true.append(target.item())

            output = model(data)
            output_prob = output.sigmoid().item()

            y_probs.append(output_prob)
            prediction = round(output_prob)
            y_pred.append(prediction)

    return y_true, y_probs, y_pred


def get_test_results_classification(model, data_loader, device, calculate_remission=True,
                                    nonremission_scores=[2, 3]):
    model.eval()

    y_true = []
    y_probs = []
    y_pred = []

    r_true = []
    r_probs = []
    r_pred = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            y_true.append(target.item())

            output = model(data)

            y_probs.append(output.softmax(1).tolist()[0])
            prediction = output.argmax(dim=1, keepdim=True)[0][0].item()
            y_pred.append(prediction)

            if calculate_remission:
                # Remission calculation
                if target.item() in nonremission_scores:
                    r_true.append(1)
                else:
                    r_true.append(0)

                if prediction in nonremission_scores:
                    r_pred.append(1)
                else:
                    r_pred.append(0)

                r_probs.append(output.softmax(1).squeeze(0)[nonremission_scores].sum().item())

    if calculate_remission:
        return y_true, y_probs, y_pred, r_true, r_probs, r_pred
    else:
        return y_true, y_probs, y_pred

def get_test_results_classification_for_corn_loss_model(model, data_loader, device, calculate_remission=True,
                                    nonremission_scores=[2, 3]):
    model.eval()

    y_true = []
    y_probs = []
    y_pred = []

    r_true = []
    r_probs = []
    r_pred = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            y_true.append(target.item())

            output = model(data)

            # TODO fix here, I may need probabilities later
            # probas = torch.sigmoid(output)
            # probas = torch.cumprod(probas, dim=1)
            # y_probs.append(probas.tolist()[0])

            prediction = label_from_logits_corn(output)
            y_pred.append(prediction.item())

            if calculate_remission:
                # Remission calculation
                if target.item() in nonremission_scores:
                    r_true.append(1)
                else:
                    r_true.append(0)

                if prediction in nonremission_scores:
                    r_pred.append(1)
                else:
                    r_pred.append(0)

                # TODO fix here, I may need probabilities later
                # r_probs.append(probas.squeeze(0)[nonremission_scores].sum().item())

    if calculate_remission:
        return y_true, y_probs, y_pred, r_true, r_probs, r_pred
    else:
        return y_true, y_probs, y_pred

def get_test_results_regression(model, data_loader, device, boundaries, nonremission_scores=[2, 3]):
    model.eval()

    y_true = []
    y_pred = []
    r_true = []
    r_pred = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            y_true.append(target.item())

            output = model(data)
            output.squeeze_(1)
            prediction = get_regression_accuracy_with_boundaries(output, target, boundaries)
            y_pred.append(int(prediction.item()))

            # Remission calculation
            if target.item() in nonremission_scores:
                r_true.append(1)
            else:
                r_true.append(0)

            if prediction in nonremission_scores:
                r_pred.append(1)
            else:
                r_pred.append(0)

    return y_true, y_pred, r_true, r_pred


def get_regression_accuracy_with_boundaries(output, target, boundaries):
    output_classified = torch.zeros_like(output)
    for output_index in range(len(output)):
        for i in range(len(boundaries)):
            if i == 0:
                if output[output_index] < boundaries[i]:
                    output_classified[output_index] = 0
                    break
            elif i == len(boundaries) - 1:
                if boundaries[i] < output[output_index]:
                    output_classified[output_index] = i + 1
                else:
                    output_classified[output_index] = i
                break
            elif boundaries[i - 1] < output[output_index] and output[output_index] < boundaries[i]:
                output_classified[output_index] = i
                break

    return output_classified


def mixup_data(x, y, device, alpha=1.0, ):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_averaged_featuremap(channels: np.ndarray, shape: tuple):
    featuremap = channels.sum(0)
    featuremap = featuremap - np.min(featuremap)
    featuremap = featuremap / np.max(featuremap)
    featuremap_image = np.uint8(255 * featuremap)
    featuremap_image = cv2.resize(featuremap_image, (shape[1], shape[0]))

    return featuremap_image


def get_CAM(channels: np.ndarray, weights: np.ndarray, shape: tuple, id: int):
    class_weights = weights[id, :]
    class_weights = class_weights[:, np.newaxis, np.newaxis]
    CAM = channels * class_weights
    CAM = CAM.sum(0)
    CAM = CAM - np.min(CAM)
    CAM_img = CAM / (1e-7 + np.max(CAM))
    CAM_img = np.float32(cv2.resize(CAM_img, (shape[1], shape[0])))
    output_CAM = np.uint8(255 * CAM_img)

    return output_CAM


def get_CAM_with_bias(channels: np.ndarray, last_fc: torch.nn.Linear, shape: tuple, id: int):
    weights = last_fc.weight.data.cpu().numpy()
    bias = last_fc.bias.data.cpu().numpy()

    class_weights = weights[id, :]
    bias = bias[id]

    class_weights = class_weights[:, np.newaxis, np.newaxis]

    CAM = channels * class_weights + bias
    CAM = CAM.sum(0)
    CAM = CAM - np.min(CAM)
    CAM_img = CAM / (1e-7 + np.max(CAM))
    CAM_img = np.float32(cv2.resize(CAM_img, (shape[1], shape[0])))
    output_CAM = np.uint8(255 * CAM_img)

    return output_CAM


def get_CAM_clip_results(channels: np.ndarray, last_fc: torch.nn.Linear, shape: tuple, id: int, use_bias=False):
    weights = last_fc.weight.data.cpu().numpy()
    if use_bias:
        bias = last_fc.bias.data.cpu().numpy()

    class_weights = weights[id, :]
    if use_bias:
        bias = bias[id]

    class_weights = class_weights[:, np.newaxis, np.newaxis]

    if use_bias:
        CAM = channels * class_weights + bias
    else:
        CAM = channels * class_weights

    CAM = CAM.sum(0)
    CAM = np.clip(CAM, 0, CAM.max())
    CAM = CAM - np.min(CAM)
    CAM_img = CAM / (1e-7 + np.max(CAM))
    CAM_img = np.float32(cv2.resize(CAM_img, (shape[1], shape[0])))
    output_CAM = np.uint8(255 * CAM_img)

    return output_CAM


def get_CAM_clip_weights(channels: np.ndarray, last_fc: torch.nn.Linear, shape: tuple, id: int, use_bias=False):
    weights = last_fc.weight.data.cpu().numpy()

    if use_bias:
        bias = last_fc.bias.data.cpu().numpy()

    class_weights = weights[id, :]
    if use_bias:
        bias = bias[id]

    class_weights = class_weights[:, np.newaxis, np.newaxis]
    class_weights = np.clip(class_weights, 0, class_weights.max())

    if use_bias:
        CAM = channels * class_weights + bias
    else:
        CAM = channels * class_weights
    CAM = CAM.sum(0)
    CAM = CAM - np.min(CAM)
    CAM_img = CAM / (1e-7 + np.max(CAM))
    CAM_img = np.float32(cv2.resize(CAM_img, (shape[1], shape[0])))
    output_CAM = np.uint8(255 * CAM_img)

    return output_CAM


def setup_reproducability(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_dataset_mean_and_std(dataset_dir):
    trainingSet = UCMayo4(dataset_dir)
    R_total = 0
    G_total = 0
    B_total = 0

    total_count = 0
    for image, _ in trainingSet:
        image = np.asarray(image)
        total_count = total_count + image.shape[0] * image.shape[1]

        R_total = R_total + np.sum(image[:, :, 0])
        G_total = G_total + np.sum(image[:, :, 1])
        B_total = B_total + np.sum(image[:, :, 2])

    R_mean = R_total / total_count
    G_mean = G_total / total_count
    B_mean = B_total / total_count

    R_total = 0
    G_total = 0
    B_total = 0

    total_count = 0
    for image, _ in trainingSet:
        image = np.asarray(image)
        total_count = total_count + image.shape[0] * image.shape[1]

        R_total = R_total + np.sum((image[:, :, 0] - R_mean) ** 2)
        G_total = G_total + np.sum((image[:, :, 1] - G_mean) ** 2)
        B_total = B_total + np.sum((image[:, :, 2] - B_mean) ** 2)

    R_std = sqrt(R_total / total_count)
    G_std = sqrt(G_total / total_count)
    B_std = sqrt(B_total / total_count)

    return [R_mean / 255, G_mean / 255, B_mean / 255], [R_std / 255, G_std / 255, B_std / 255]


def write_metric_results_to_file(wandb_rund_dir, accuracies=None, kappa_scores=None, weighted_kappa_scores=None,
                                 sensitivities=None, specificities=None, macro_precisions=None,
                                 macro_recalls=None, macro_f1s=None, class_precisions=None, class_recalls=None,
                                 class_f1s=None,
                                 accuracies_r=None, kappa_scores_r=None, sensitivities_r=None, specificities_r=None,
                                 precisions_r=None, recalls_r=None, f1s_r=None):
    results = []
    results.append("\n------------- 4-class Score -------------\n")
    if accuracies is not None:
        results.append("Accuracies: " + str(accuracies))
        results.append("Accuracy mean: " + str(mean(accuracies)))
        results.append("Accuracy stddev: " + str(pstdev(accuracies)))
        results.append("")
    if kappa_scores is not None:
        results.append("Kappas: " + str(kappa_scores))
        results.append("kappa mean: " + str(mean(kappa_scores)))
        results.append("kappa stddev: " + str(pstdev(kappa_scores)))
        results.append("")
    if weighted_kappa_scores is not None:
        results.append("QWK: " + str(weighted_kappa_scores))
        results.append("QWK mean: " + str(mean(weighted_kappa_scores)))
        results.append("QWK stddev: " + str(pstdev(weighted_kappa_scores)))
        results.append("")
    if sensitivities is not None:
        results.append("Sensitivities: " + str(sensitivities))
        results.append("Sensitivity mean: " + str(mean(sensitivities)))
        results.append("Sensitivity stddev: " + str(pstdev(sensitivities)))
        results.append("")
    if specificities is not None:
        results.append("Specificities: " + str(specificities))
        results.append("Specificity mean: " + str(mean(specificities)))
        results.append("Specificity stddev: " + str(pstdev(specificities)))
        results.append("")
    if macro_precisions is not None:
        results.append("Macro precision: " + str(macro_precisions))
        results.append("Macro precision mean: " + str(mean(macro_precisions)))
        results.append("Macro precision stddev: " + str(pstdev(macro_precisions)))
        results.append("")
    if macro_recalls is not None:
        results.append("Macro Recall: " + str(macro_recalls))
        results.append("Macro Recall mean: " + str(mean(macro_recalls)))
        results.append("Macro Recall stddev: " + str(pstdev(macro_recalls)))
        results.append("")
    if macro_f1s is not None:
        results.append("Macro f1: " + str(macro_f1s))
        results.append("Macro f1 mean: " + str(mean(macro_f1s)))
        results.append("Macro f1 stddev: " + str(pstdev(macro_f1s)))
        results.append("")
    if class_precisions is not None:
        results.append("Mayo-0 precision: " + str(class_precisions[:, 0]))
        results.append("Mayo-0 precision mean: " + str(mean(class_precisions[:, 0])))
        results.append("Mayo-0 precision stddev: " + str(pstdev(class_precisions[:, 0])))
        results.append("")
    if class_recalls is not None:
        results.append("Mayo-0 recall: " + str(class_recalls[:, 0]))
        results.append("Mayo-0 recall mean: " + str(mean(class_recalls[:, 0])))
        results.append("Mayo-0 recall stddev: " + str(pstdev(class_recalls[:, 0])))
        results.append("")
    if class_f1s is not None:
        results.append("Mayo-0 f1: " + str(class_f1s[:, 0]))
        results.append("Mayo-0 f1 mean: " + str(mean(class_f1s[:, 0])))
        results.append("Mayo-0 f1 stddev: " + str(pstdev(class_f1s[:, 0])))
        results.append("")
    if class_precisions is not None:
        results.append("Mayo-1 precision: " + str(class_precisions[:, 1]))
        results.append("Mayo-1 precision mean: " + str(mean(class_precisions[:, 1])))
        results.append("Mayo-1 precision stddev: " + str(pstdev(class_precisions[:, 1])))
        results.append("")
    if class_recalls is not None:
        results.append("Mayo-1 recall: " + str(class_recalls[:, 1]))
        results.append("Mayo-1 recall mean: " + str(mean(class_recalls[:, 1])))
        results.append("Mayo-1 recall stddev: " + str(pstdev(class_recalls[:, 1])))
        results.append("")
    if class_f1s is not None:
        results.append("Mayo-1 f1: " + str(class_f1s[:, 1]))
        results.append("Mayo-1 f1 mean: " + str(mean(class_f1s[:, 1])))
        results.append("Mayo-1 f1 stddev: " + str(pstdev(class_f1s[:, 1])))
        results.append("")
    if class_precisions is not None:
        results.append("Mayo-2 precision: " + str(class_precisions[:, 2]))
        results.append("Mayo-2 precision mean: " + str(mean(class_precisions[:, 2])))
        results.append("Mayo-2 precision stddev: " + str(pstdev(class_precisions[:, 2])))
        results.append("")
    if class_recalls is not None:
        results.append("Mayo-2 recall: " + str(class_recalls[:, 2]))
        results.append("Mayo-2 recall mean: " + str(mean(class_recalls[:, 2])))
        results.append("Mayo-2 recall stddev: " + str(pstdev(class_recalls[:, 2])))
        results.append("")
    if class_f1s is not None:
        results.append("Mayo-2 f1: " + str(class_f1s[:, 2]))
        results.append("Mayo-2 f1 mean: " + str(mean(class_f1s[:, 2])))
        results.append("Mayo-2 f1 stddev: " + str(pstdev(class_f1s[:, 2])))
        results.append("")
    if class_precisions is not None:
        results.append("Mayo-3 precision: " + str(class_precisions[:, 3]))
        results.append("Mayo-3 precision mean: " + str(mean(class_precisions[:, 3])))
        results.append("Mayo-3 precision stddev: " + str(pstdev(class_precisions[:, 3])))
        results.append("")
    if class_recalls is not None:
        results.append("Mayo-3 recall: " + str(class_recalls[:, 3]))
        results.append("Mayo-3 recall mean: " + str(mean(class_recalls[:, 3])))
        results.append("Mayo-3 recall stddev: " + str(pstdev(class_recalls[:, 3])))
        results.append("")
    if class_f1s is not None:
        results.append("Mayo-3 f1: " + str(class_f1s[:, 3]))
        results.append("Mayo-3 f1 mean: " + str(mean(class_f1s[:, 3])))
        results.append("Mayo-3 f1 stddev: " + str(pstdev(class_f1s[:, 3])))
        results.append("")

    results.append("\n------------- Remission -------------\n")
    if accuracies_r is not None:
        results.append("Accuracies_r: " + str(accuracies_r))
        results.append("Accuracies_r mean: " + str(mean(accuracies_r)))
        results.append("Accuracies_r stddev: " + str(pstdev(accuracies_r)))
        results.append("")
    if kappa_scores_r is not None:
        results.append("kappa_r: " + str(kappa_scores_r))
        results.append("kappa_r mean: " + str(mean(kappa_scores_r)))
        results.append("kappa_r stddev: " + str(pstdev(kappa_scores_r)))
        results.append("")
    if sensitivities_r is not None:
        results.append("sensitivities_r: " + str(sensitivities_r))
        results.append("sensitivities_r mean: " + str(mean(sensitivities_r)))
        results.append("sensitivities_r stddev: " + str(pstdev(sensitivities_r)))
        results.append("")
    if specificities_r is not None:
        results.append("specificities_r: " + str(specificities_r))
        results.append("specificities_r mean: " + str(mean(specificities_r)))
        results.append("specificities_r stddev: " + str(pstdev(specificities_r)))
        results.append("")
    if precisions_r is not None:
        results.append("precisions_r: " + str(precisions_r))
        results.append("precisions_r mean: " + str(mean(precisions_r)))
        results.append("precisions_r stddev: " + str(pstdev(precisions_r)))
        results.append("")
    if recalls_r is not None:
        results.append("recalls_r: " + str(recalls_r))
        results.append("recalls_r mean: " + str(mean(recalls_r)))
        results.append("recalls_r stddev: " + str(pstdev(recalls_r)))
        results.append("")
    if f1s_r is not None:
        results.append("f1s_r: " + str(f1s_r))
        results.append("f1s_r mean: " + str(mean(f1s_r)))
        results.append("f1s_r stddev: " + str(pstdev(f1s_r)))

    results_separated = map(lambda x: x + "\n", results)
    file = open(os.path.join(wandb_rund_dir, "results.txt"), "w")
    file.writelines(results_separated)
    file.close()
