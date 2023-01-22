import copy
import numpy as np
import os
from pathlib import Path
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

from utils.balancedDataset import BalancedDataset
from utils.const import *
from utils.helperFunctions import setSeed, getScores, getSubDirs
from utils.tasks import currentTask

# ITERABLE PARAMETERS

# Ratio between classes cat and dog
BALANCES = [[50, 50], [40, 60], [30, 70], [20, 80]]

# Models to train
MODEL_NAMES = ["alexnet", "resnet", "vgg"]

# OTHER PARAMETERS
NUM_CLASSES = 2  # Binary Classification
NUM_WORKERS = 0
PIN_MEMORY = True

# Batch size for training (change depending on how much memory you have)
BATCH_SIZE = 128

# Early stopping
NUM_EPOCHS = 500  # Number of epochs to train for
PATIENCE_ES = 20  # Patience for early stopping
DELTA_ES = 0.0001  # Delta for early stopping

# Flag for feature extracting. When False, we finetune the whole model,
# when True we only update the reshaped layer params
FEATURE_EXTRACT = False

LEARNING_RATE = 0.001  # The learning rate of the optimizer
MOMENTUM = 0.9  # The momentum of the optimizer

### HELPER FUNCTIONS ###


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, delta=0, patience=10):
    since = time.time()
    last_since = time.time()

    scores_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

    best_score = None
    counter = 0

    for epoch in range(num_epochs):
        print('[💪 EPOCH] {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        epoch_score = None

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            labels_outputs = torch.tensor([]).to(DEVICE, non_blocking=True)
            labels_targets = torch.tensor([]).to(DEVICE, non_blocking=True)

            # Iterate over data
            setSeed()
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                labels_outputs = torch.cat([labels_outputs, preds], dim=0)
                labels_targets = torch.cat([labels_targets, labels], dim=0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc, epoch_prec, epoch_rec, epoch_f1 = getScores(
                labels_targets, labels_outputs, complete=False)

            print('[🗃️ {}] Loss: {:.4f} Acc: {:.4f} Pre: {:.4f} Rec: {:.4f} F-Score: {:.4f}'.format(
                phase.upper(), epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_f1))

            time_elapsed = time.time() - last_since
            last_since = time.time()
            print("\t[🕑] {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60))

            if phase == 'val':
                epoch_score = epoch_f1

                # Deep copy the model
                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    best_model_wts = copy.deepcopy(model.state_dict())

                # Store scores history
                scores_history.append({
                    "loss": epoch_loss,
                    "acc": epoch_acc.cpu().numpy(),
                    "precision": epoch_prec.cpu().numpy(),
                    "recall": epoch_rec.cpu().numpy(),
                    "f1": epoch_f1.cpu().numpy()
                })

        if best_score is None:
            best_score = epoch_score
        elif epoch_score <= best_score + delta:
            counter += 1
            print("\t[⚠️ EARLY STOPPING] {}/{}".format(counter, patience))
            if counter >= patience:
                break
        else:
            best_score = epoch_score
            counter = 0

        print()

    time_elapsed = time.time() - since
    print()
    print('[🕑 TRAINING COMPLETE] {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('[🥇 BEST SCORE] F-Score: {:4f}'.format(best_f1))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, scores_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def evaluateModel(model, dataloader):
    model.eval()
    labelsOutputs = torch.tensor([]).to(DEVICE, non_blocking=True)

    for inputs, labels in dataloader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        labelsOutputs = torch.cat([labelsOutputs, preds], dim=0)

    return labelsOutputs


def evaluateModelF1(model, dataloader):
    model.eval()
    labelsOutputs = torch.tensor([]).to(DEVICE, non_blocking=True)
    labelsTargets = torch.tensor([]).to(DEVICE, non_blocking=True)

    for inputs, labels in dataloader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        labelsOutputs = torch.cat([labelsOutputs, preds], dim=0)
        labelsTargets = torch.cat([labelsTargets, labels], dim=0)

    acc, precision, recall, f1 = getScores(
        labelsTargets, labelsOutputs, complete=False)

    return {
        "acc": acc.cpu().numpy(),
        "precision": precision.cpu().numpy(),
        "recall": recall.cpu().numpy(),
        "f1": f1.cpu().numpy()
    }


def evaluateModelsOnDataset(datasetFolder, datasetInfo):
    modelsEvals = []

    # Setup for normalization
    dataTransform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZATION_PARAMS[0], NORMALIZATION_PARAMS[1])
    ])

    testDataset = BalancedDataset(
        datasetFolder, transform=dataTransform, use_cache=True, check_images=False)

    setSeed()
    testDataLoader = DataLoader(
        testDataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

    # Evaluate every model
    for root, _, fnames in sorted(os.walk(MODELS_DIR, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)

            try:
                modelData = torch.load(path)
            except:
                continue

            modelDataset = modelData["dataset"]
            modelName = modelData["model_name"]
            modelPercents = "/".join([str(x)
                                     for x in modelData["balance"]])

            print()
            print("[🧮 EVALUATING] {} - {} {}".format(
                modelDataset,
                modelName,
                modelPercents
            ))

            modelToTest = modelData["model"]
            modelToTest = modelToTest.to(DEVICE, non_blocking=True)

            scores = evaluateModelF1(modelToTest, testDataLoader)

            modelsEvals.append({
                "source_dataset": datasetInfo["dataset"],
                "target_model": modelName,
                "target_dataset": modelDataset,
                "target_balancing": modelPercents,
                "baseline_f1": scores["f1"]
            })

            print("\tAcc: {:.4f}".format(scores["acc"]))
            print("\tPre: {:.4f}".format(scores["precision"]))
            print("\tRec: {:.4f}".format(scores["recall"]))
            print("\tF-Score: {:.4f}".format(scores["f1"]))

            torch.cuda.empty_cache()

    return modelsEvals


### ITERATING MODELS AND BALANCES ###
setSeed()

for dataset_dir in sorted(getSubDirs(DATASETS_DIR)):
    for model_name in sorted(MODEL_NAMES):
        for balance in sorted(BALANCES):
            print(
                f'\n\n[🤖 MODEL] {dataset_dir} - {model_name} - {balance}\n\n')

            data_dir = os.path.join(DATASETS_DIR, dataset_dir)

            current_dir = os.getcwd()
            curr_append = os.path.join(os.path.join(
                MODELS_DIR, dataset_dir), model_name)

            model_save_path = os.path.join(current_dir, curr_append)
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            model_save_name = "{}_{}".format(
                model_name, "_".join(str(b) for b in balance))
            model_save_path = os.path.join(model_save_path, model_save_name)

            if os.path.exists(model_save_path + ".pt"):
                print('\t[✅ SKIPPING] ALREADY TRAINED')
                continue

            # Initialize the model for this run
            model_ft, input_size = initialize_model(
                model_name, NUM_CLASSES, FEATURE_EXTRACT, use_pretrained=True)

            # Data resize and normalization
            data_transforms = {
                "train": transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(NORMALIZATION_PARAMS[0], NORMALIZATION_PARAMS[1])
                ]),
                "val": transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(NORMALIZATION_PARAMS[0], NORMALIZATION_PARAMS[1])
                ]),
            }

            # Create training and validation datasets
            image_datasets = {x: BalancedDataset(os.path.join(data_dir, x),
                                                 transform=data_transforms[x],
                                                 balance=balance,
                                                 check_images=False,
                                                 use_cache=True) for x in ["train", "val"]}

            # Check the sizes of the created datasets
            for x in ["train", "val"]:
                print()

                print("[🗃️ {}]".format(x.upper()))
                for cls in image_datasets[x].classes:
                    cls_index = image_datasets[x].class_to_idx[cls]
                    num_cls = np.count_nonzero(
                        np.array(image_datasets[x].targets) == cls_index)
                    print("[🧮 # ELEMENTS] {}: {}".format(cls, num_cls))

            # Create training and validation dataloaders
            setSeed()
            dataloaders_dict = {x: torch.utils.data.DataLoader(
                image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY) for x in ["train", "val"]}

            model_ft = model_ft.to(DEVICE, non_blocking=True)

            # Gather the parameters to be optimized/updated in this run. If we are
            #  finetuning we will be updating all parameters. However, if we are
            #  doing feature extract method, we will only update the parameters
            #  that we have just initialized, i.e. the parameters with requires_grad
            #  is True.
            params_to_update = model_ft.parameters()
            if FEATURE_EXTRACT:
                params_to_update = []
                for name, param in model_ft.named_parameters():
                    if param.requires_grad == True:
                        params_to_update.append(param)

            # Observe that all parameters are being optimized
            optimizer_ft = optim.SGD(
                params_to_update, lr=LEARNING_RATE, momentum=MOMENTUM)

            # Setup the loss fxn
            criterion = nn.CrossEntropyLoss()

            # Train and evaluate
            setSeed()
            model_ft, scores_history = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,
                                                   num_epochs=NUM_EPOCHS, is_inception=False,
                                                   delta=DELTA_ES, patience=PATIENCE_ES)

            torch.save({
                'model': model_ft,
                'task': currentTask,
                'dataset': dataset_dir,
                'learning_rate': LEARNING_RATE,
                'momentum': MOMENTUM,
                'balance': balance,
                'model_name': model_name,
                'batch_size': BATCH_SIZE,
                'num_epochs': NUM_EPOCHS,
                'criterion': criterion,
                'optimizer': optimizer_ft,
                'scores_history': scores_history,
                'delta_es': DELTA_ES,
                'patience_es': PATIENCE_ES
            }, model_save_path + ".pt")

            print("[💾 SAVED]", dataset_dir, model_name,
                  "/".join(str(b) for b in balance))


### GENERATING PREDICTIONS ###
print("\n\n" + "-" * 50)
print("\n[🧠 GENERATING MODEL PREDICTIONS]")

predictions = []

for dataset in sorted(getSubDirs(DATASETS_DIR)):
    print("\n" + "-" * 15)
    print("[🗃️ DATASET] {}\n".format(dataset))

    datasetDir = os.path.join(DATASETS_DIR, dataset)
    testDir = os.path.join(datasetDir, "test")

    toTensor = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZATION_PARAMS[0], NORMALIZATION_PARAMS[1])
    ])

    testDataset = BalancedDataset(
        testDir, transform=toTensor, use_cache=False, check_images=False)

    testDataLoader = DataLoader(testDataset, batch_size=16, shuffle=False)

    for root, _, fnames in sorted(os.walk(MODELS_DIR)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)

            modelData = torch.load(path)

            modelDataset = modelData["dataset"]
            modelName = modelData["model_name"]

            modelBalance = "/".join(str(x) for x in modelData["balance"])

            print("[🎖️ EVALUATING]", modelData["model_name"], modelBalance)

            modelToTest = modelData["model"]
            modelToTest = modelToTest.to(DEVICE, non_blocking=True)

            outputs = evaluateModel(modelToTest, testDataLoader)

            for (image, label), output in zip(testDataset.imgs, outputs):
                predictions.append(
                    {
                        "task": currentTask,
                        "model": modelData["model_name"],
                        "model_dataset": modelData["dataset"],
                        "balance": modelBalance,
                        "dataset": dataset,
                        "image": Path(image),
                        "name": Path(image).name,
                        "label": label,
                        "prediction": int(output.cpu().numpy())
                    }
                )

predictionsDF = pd.DataFrame(predictions)

if not os.path.exists(os.path.dirname(MODEL_PREDICTIONS_PATH)):
    os.makedirs(os.path.dirname(MODEL_PREDICTIONS_PATH))

predictionsDF.to_csv(MODEL_PREDICTIONS_PATH)


print("\n\n" + "-" * 50)
print("\n[🧠 MODELS EVALUATION - BASELINE]")

modelsEvals = []

# Evaluate models on test folders
for dataset in sorted(getSubDirs(DATASETS_DIR)):
    print("\n" + "-" * 15)
    print("[🗃️ TEST DATASET] {}".format(dataset))

    datasetDir = os.path.join(DATASETS_DIR, dataset)
    testDir = os.path.join(datasetDir, "test")

    advDatasetInfo = {
        "dataset": dataset,
        "math": None,
        "attack": None,
        "balancing": None,
        "model": None,
    }

    evals = evaluateModelsOnDataset(testDir, advDatasetInfo)
    modelsEvals.extend(evals)

modelsEvalsDF = pd.DataFrame(modelsEvals)

if not os.path.exists(os.path.dirname(BASELINE_PATH)):
    os.makedirs(os.path.dirname(BASELINE_PATH))

modelsEvalsDF.to_csv(BASELINE_PATH)