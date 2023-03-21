import numpy as np
import os
import random

import torch
from torchvision import transforms

from .const import *


def getSubDirs(dir):
    return [x for x in os.listdir(dir) if os.path.isdir(os.path.join(dir, x))]


def setSeed(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def getScores(labels, predicted, complete=True):
    labels = torch.tensor(labels).to(DEVICE, non_blocking=True)
    predicted = torch.tensor(predicted).to(DEVICE, non_blocking=True)

    acc = torch.sum(predicted == labels) / len(predicted)

    tp = (labels * predicted).sum()
    fp = ((1 - labels) * predicted).sum()
    fn = (labels * (1 - predicted)).sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1 = 2 * (precision * recall) / (precision + recall)

    if not complete:
        return acc, precision, recall, f1

    asr = torch.sum(predicted != labels) / len(predicted)

    len_0 = 0
    len_1 = 0
    n_0 = 0
    n_1 = 0
    for pred, lab in zip(predicted, labels):
        if int(lab) == 0:
            len_0 += 1
            if int(pred) == 1:
                n_0 += 1
        else:
            len_1 += 1
            if int(pred) == 0:
                n_1 += 1

    asr_0 = n_0 / len_0
    asr_1 = n_1 / len_1

    return acc, precision, recall, f1, asr, asr_0, asr_1


def evaluateModel(model, dataloader, dataset, modelInfo, dfMath):
    balance = "/".join([str(x) for x in modelInfo["balance"]])
    # Detecting samples that were misclassified by the base model
    # In this way we evaluate only the ASR
    misclassified = dfMath.loc[currentTask,
                                modelInfo["model_name"],
                                modelInfo["dataset"],
                                balance,
                                dataset]
    misclassified = misclassified[misclassified["label"]
                                    != misclassified["prediction"]]
    misclassified = np.array(misclassified["name"])

    model.eval()
    labelsOutputs = []
    labelsTargets = []

    for inputs, labels, paths in dataloader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        for pred, label, path in zip(preds, labels, paths):
            if os.path.basename(path) not in misclassified:
                labelsOutputs.append(pred)
                labelsTargets.append(label)

    acc, precision, recall, f1, asr, asr_0, asr_1 = getScores(
        labelsTargets, labelsOutputs)

    return {
        "acc": acc.cpu().numpy(),
        "precision": precision.cpu().numpy(),
        "recall": recall.cpu().numpy(),
        "f1": f1.cpu().numpy(),
        "asr": asr.cpu().numpy(),
        "asr_0": asr_0,
        "asr_1": asr_1
    }


def saveMathAdversarials(dataloader, classes, attack, saveDir):
    saved = 0

    for images, labels, paths in dataloader:
        adversarials = attack(images, labels)

        for adversarial, label, fName in zip(adversarials, labels, paths):
            image = transforms.ToPILImage()(adversarial).convert("RGB")
            path = os.path.join(saveDir, classes[label])

            if not os.path.exists(path):
                os.makedirs(path)

            imageName = os.path.basename(fName)
            image.save(os.path.join(path, imageName), "JPEG")
            saved += 1

            if saved % 20 == 0:
                print(f"\t[💾 SAVED] #{saved} images")
