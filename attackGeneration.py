import numpy as np
import os

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchattacks import FGSM, DeepFool, BIM, RFGSM, PGD, TIFGSM
from torchmetrics import StructuralSimilarityIndexMeasure
from torchvision import transforms

from utils.balancedDataset import BalancedDataset
from utils.nonMathAttacks import NonMathAttacks
from utils.tasks import currentTask


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Ranges and step for attack epsilon

attacks_names_math = [
    'BIM',
    'DeepFool',
    'FGSM',
    'PGD',
    'RFGSM',
    'TIFGSM'
]

eps_init_math = {
    "BIM": 0.01,
    "FGSM": 0.01,
    "DeepFool": 10,
    "RFGSM": 0.01,
    "PGD": 0.01,
    "TIFGSM": 0.01
}

steps_math = {
    "BIM": 0.01,
    "FGSM": 0.01,
    "DeepFool": 1,
    "RFGSM": 0.01,
    "PGD": 0.01,
    "TIFGSM": 0.01
}

eps_thresholds_math = {
    "BIM": 0.3,
    "FGSM": 0.3,
    "DeepFool": 100,
    "RFGSM": 0.3,
    "PGD": 0.3,
    "TIFGSM": 0.3
}

nonMathAttacks = NonMathAttacks()

eps_init_non_math = {
    "GaussianNoise": 0.005,
    "BoxBlur": 0.5,
    "Sharpen": 1,
    "InvertColor": 1,
    "GreyScale": 1,
    "SplitMergeRGB": 1,
    "SaltPepper": 0.005,
    "RandomBlackBox": 10
}

steps_non_math = {
    "GaussianNoise": 0.005,
    "BoxBlur": 0.5,
    "Sharpen": 0,
    "InvertColor": 0,
    "GreyScale": 0,
    "SplitMergeRGB": 0,
    "SaltPepper": 0.005,
    "RandomBlackBox": 10
}

eps_thresholds_non_math = {
    "GaussianNoise": 0.1,
    "BoxBlur": 10,
    "Sharpen": 1,
    "InvertColor": 1,
    "GreyScale": 1,
    "SplitMergeRGB": 1,
    "SaltPepper": 0.1,
    "RandomBlackBox": 200
}

attack_names_non_math = [
    "GaussianNoise",
    "BoxBlur",
    "Sharpen",
    "InvertColor",
    "GreyScale",
    "SplitMergeRGB",
    "SaltPepper",
    "RandomBlackBox"
]


# Parameters

datasetToFolder = {"bing": "bing", "google": "google"}

shuffleDataset = False  # Shuffle the dataset

inputSize = 224  # Specified for alexnet, resnet, vgg
datasetSize = 150  # Reduce the size of the dataset

adversarialDir = "./adversarialSamples/" + currentTask

if not os.path.exists(os.path.join(os.getcwd(), adversarialDir)):
    os.makedirs(os.path.join(os.getcwd(), adversarialDir))

datasetsDir = "./datasets/" + currentTask
modelsDir = "./models/" + currentTask

modelPredictions_path = './results/models/predictions/predictions_' + currentTask + '.csv'
dfMath = pd.read_csv(modelPredictions_path, index_col=[
                     "task", "model", "model_dataset", "balance", "dataset"]).sort_index()

# Setting seed for reproducibility

SEED = 151836


def setSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


setSeed(SEED)

# Helper functions


def getSubDirs(dir):
    return [x for x in os.listdir(dir) if os.path.isdir(os.path.join(dir, x))]


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


def getBestScores(hist, key, min=False):
    scores = [x[key] for x in hist]

    if min:
        i = np.argmin(np.array(scores))
    else:
        i = np.argmax(np.array(scores))

    return hist[i], i


def getScores(labels, predicted):
    labels = torch.tensor(labels).to(device, non_blocking=True)
    predicted = torch.tensor(predicted).to(device, non_blocking=True)

    acc = torch.sum(predicted == labels) / len(predicted)

    tp = (labels * predicted).sum()
    fp = ((1 - labels) * predicted).sum()
    fn = (labels * (1 - predicted)).sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

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

    asr_0 = n_0/len_0
    asr_1 = n_1/len_1

    f1 = 2 * (precision * recall) / (precision + recall)

    return acc, precision, recall, f1, asr, asr_0, asr_1


def evaluateModel(model, dataloader, dataset, modelInfo):
    balance = "/".join([str(x) for x in modelInfo["balance"]])
    missclassified = dfMath.loc[currentTask,
                                modelInfo["model_name"],
                                modelInfo["dataset"],
                                balance,
                                dataset]
    missclassified = missclassified[missclassified["label"]
                                    != missclassified["prediction"]]
    missclassified = np.array(missclassified["name"])

    model.eval()
    labelsOutputs = []
    labelsTargets = []

    for inputs, labels, paths in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        for pred, label, path in zip(preds, labels, paths):
            if os.path.basename(path) not in missclassified:
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


modelsEvals = []

datasetsToGenerate = getSubDirs(datasetsDir)

i = 0

csv_data = []

print("[🧠 MATH ATTACK GENERATION]\n")

for attack_name in attacks_names_math:
    csv_data = []
    for dataset in sorted(datasetsToGenerate):

        print("\n" + "-" * 15)
        print("[🗃️  SOURCE DATASET] {}\n".format(dataset))

        datasetDir = os.path.join(datasetsDir, dataset)
        testDir = os.path.join(datasetDir, "test")

        datasetAdvDir = os.path.join(adversarialDir, dataset)
        mathAttacksDir = os.path.join(datasetAdvDir, "math")

        if not os.path.exists(mathAttacksDir):
            os.makedirs(mathAttacksDir)

        toTensor = transforms.Compose([transforms.ToTensor()])
        toNormalizedTensor = transforms.Compose([
            transforms.Resize(inputSize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        for root, _, fnames in sorted(os.walk(os.path.join(modelsDir, datasetToFolder[dataset]), followlinks=True)):
            for fname in sorted(fnames):
                eps = eps_init_math[attack_name]
                effective = False
                asr_history = []

                path = os.path.join(root, fname)

                modelData = torch.load(path, map_location=torch.device('cpu'))

                modelDataset = modelData["dataset"]
                modelName = modelData["model_name"]

                torch.cuda.empty_cache()

                modelPercents = "_".join([str(x)
                                         for x in modelData["balance"]])
                model = modelData["model"].to(device)

                # Test dataset without normalization (for generating samples)
                originalTestDataset = BalancedDataset(
                    testDir, transform=toTensor, datasetSize=datasetSize, use_cache=False, check_images=False, with_path=True)

                setSeed(SEED)
                originalTestDataLoader = DataLoader(
                    originalTestDataset, batch_size=16, num_workers=0, shuffle=shuffleDataset)

                # Test dataset with normalization (for evaluation)
                testDataset = BalancedDataset(
                    testDir, transform=toNormalizedTensor, datasetSize=datasetSize, use_cache=False, check_images=False, with_path=True)

                setSeed(SEED)
                testDataLoader = DataLoader(
                    testDataset, batch_size=16, num_workers=0, shuffle=shuffleDataset)

                justCreated = False

                while not effective:
                    attacks = {
                        "BIM": BIM(model, eps=eps),
                        "FGSM": FGSM(model, eps=eps),
                        "DeepFool": DeepFool(model, overshoot=eps),
                        "PGD": PGD(model, eps=eps),
                        "RFGSM": RFGSM(model, eps=eps),
                        "TIFGSM": TIFGSM(model, eps=eps)
                    }
                    for attack in attacks:
                        if attack == attack_name:
                            attacker = attacks[attack]

                            attackDir = os.path.join(
                                mathAttacksDir, attack)
                            saveDir = os.path.join(
                                attackDir, modelName + "/" + modelPercents)

                            if not os.path.exists(saveDir):
                                os.makedirs(saveDir)

                            print("\n[⚔️  ADVERSARIAL] {} @ {} - {} - {} {}".format(
                                attack,
                                eps,
                                modelDataset,
                                modelName,
                                modelPercents
                            ))

                            setSeed(SEED)
                            saveMathAdversarials(
                                originalTestDataLoader, originalTestDataset.classes, attacker, saveDir)

                            advDatasetInfo = {
                                "dataset": dataset,
                                "math": True,
                                "attack": attack,
                                "balancing": modelPercents.replace("_", "/"),
                                "model": modelName,
                            }

                            # Load the adversarial images created with normalization
                            advDataset = BalancedDataset(
                                saveDir, transform=toNormalizedTensor, datasetSize=datasetSize, use_cache=False, check_images=False, with_path=True)

                            setSeed(SEED)
                            advDataLoader = DataLoader(
                                advDataset, batch_size=16, num_workers=0, shuffle=shuffleDataset)

                            evals = evaluateModel(
                                model, advDataLoader, dataset, modelData)

                            modelsEvals.extend(evals)
                            asr = evals['asr']
                            asr_0 = evals['asr_0']
                            asr_1 = evals['asr_1']
                            asr_history.append(asr)

                            # Estimating SSIM

                            ssims = []
                            ssim_measure = StructuralSimilarityIndexMeasure(
                                data_range=1.0)

                            for (advBatch, _, advPaths), (testBatch, _, testPaths) in zip(advDataLoader, testDataLoader):
                                ssims.append(ssim_measure(
                                    advBatch, testBatch))

                            mean_ssim = sum(ssims)/len(ssims)

                            print('\n\t[🖼️  SSIM]: {}'.format(
                                round(float(mean_ssim), 2)))
                            print('\t[🎯 ASR]: {}'.format(
                                round(float(asr), 2)))
                            print('\t\t[ASR_0]: {}'.format(
                                round(float(asr_0), 2)))
                            print('\t\t[ASR_1]: {}\n'.format(
                                round(float(asr_1), 2)))

                            csv_data.append({
                                'attack': attack,
                                'task': currentTask,
                                'model': modelName,
                                'balance': modelPercents,
                                'dataset': modelDataset,
                                'ssim': mean_ssim.item(),
                                'eps': eps,
                                'asr': asr,
                                'asr_0': asr_0,
                                'asr_1': asr_1
                            })

                            if eps >= eps_thresholds_math[attack_name]:
                                effective = True
                            else:
                                eps += steps_math[attack_name]
                                eps = round(eps, 2)

                            i += 1

                            torch.cuda.empty_cache()

    data_df = pd.DataFrame(csv_data)
    data_df.to_csv('./results/attacks/history/' + currentTask + '/' + attack + '.csv')


print("\n\n[🧠 NON-MATH ATTACK GENERATION]\n")

for attack_name in attack_names_non_math:
    csv_data = []
    for dataset in sorted(datasetsToGenerate):

        print("\n" + "-" * 15)
        print("[🗃️  SOURCE DATASET] {}\n".format(dataset))

        datasetDir = os.path.join(datasetsDir, dataset)
        testDir = os.path.join(datasetDir, "test")

        datasetAdvDir = os.path.join(adversarialDir, dataset)
        nonMathAttacksDir = os.path.join(datasetAdvDir, "nonMath")

        if not os.path.exists(nonMathAttacksDir):
            os.makedirs(nonMathAttacksDir)

        toTensor = transforms.Compose([transforms.ToTensor()])
        toNormalizedTensor = transforms.Compose([
            transforms.Resize(inputSize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        for root, _, fnames in sorted(os.walk(os.path.join(modelsDir, datasetToFolder[dataset]), followlinks=True)):
            for fname in sorted(fnames):
                eps = eps_init_non_math[attack_name]
                effective = False
                asr_history = []

                path = os.path.join(root, fname)

                modelData = torch.load(path, map_location=torch.device('cpu'))

                modelDataset = modelData["dataset"]
                modelName = modelData["model_name"]

                # if not modelName in modelsGenerateOnly:
                torch.cuda.empty_cache()
                # continue

                modelPercents = "_".join([str(x)
                                         for x in modelData["balance"]])
                model = modelData["model"].to(device)

                # Test dataset without normalization (for generating samples)
                originalTestDataset = BalancedDataset(
                    testDir, transform=toTensor, datasetSize=datasetSize, use_cache=False, check_images=False, with_path=True)

                setSeed(SEED)
                originalTestDataLoader = DataLoader(
                    originalTestDataset, batch_size=16, num_workers=0, shuffle=shuffleDataset)

                # Test dataset with normalization (for evaluation)
                testDataset = BalancedDataset(
                    testDir, transform=toNormalizedTensor, datasetSize=datasetSize, use_cache=False, check_images=False, with_path=True)

                setSeed(SEED)
                testDataLoader = DataLoader(
                    testDataset, batch_size=16, num_workers=0, shuffle=shuffleDataset)

                es_count = 0
                finishNext = False

                while not effective:
                    attacks = {
                        "GaussianNoise": nonMathAttacks.gaussianNoise,
                        "BoxBlur": nonMathAttacks.boxBlur,
                        "Sharpen": nonMathAttacks.sharpen,
                        "InvertColor": nonMathAttacks.invertColor,
                        "GreyScale": nonMathAttacks.greyscale,
                        "SplitMergeRGB": nonMathAttacks.splitMergeRGB,
                        "SaltPepper": nonMathAttacks.saltAndPepper,
                        "RandomBlackBox": nonMathAttacks.randomBlackBox,
                    }
                    for attack in attacks:
                        if attack == attack_name:
                            print("\n[⚔️  ATTACKS] {} @ {} - {} - {} {}".format(
                                attack,
                                eps,
                                modelDataset,
                                modelName,
                                modelPercents
                            ))
                            for path, cls in sorted(testDataset.imgs):
                                clsName = testDataset.classes[cls]

                                imageName = os.path.basename(path)

                                image = Image.open(path).convert("RGB")

                                attacker = attacks[attack]

                                attackDir = os.path.join(
                                    nonMathAttacksDir, attack)
                                saveDir = os.path.join(attackDir, modelName)
                                saveDir2 = os.path.join(saveDir, modelPercents)
                                saveDir = os.path.join(saveDir2, clsName)

                                if not os.path.exists(saveDir):
                                    os.makedirs(saveDir)

                                outImage = image.copy()
                                if attack != 'Sharpen' and attack != 'InvertColor' and attack != 'GreyScale' and attack != 'SplitMergeRGB':
                                    outImage = attacker(outImage, amount=eps)
                                else:
                                    outImage = attacker(outImage)
                                    effective = True
                                outImage.save(os.path.join(
                                    saveDir, imageName), "JPEG")

                            print(f"\t[💾 IMAGES SAVED]")

                            advDatasetInfo = {
                                "dataset": dataset,
                                "math": True,
                                "attack": attack,
                                "balancing": modelPercents.replace("_", "/"),
                                "model": modelName,
                            }

                            # Load the adversarial images created with normalization
                            advDataset = BalancedDataset(
                                saveDir2, transform=toNormalizedTensor, datasetSize=datasetSize, use_cache=False, check_images=False, with_path=True)
                            setSeed(SEED)
                            advDataLoader = DataLoader(
                                advDataset, batch_size=16, num_workers=0, shuffle=shuffleDataset)

                            # If finishNext that means that we have already found
                            # the best eps for our task. We will then only generate
                            # those adversarial samples again with no need of
                            # evaluating its performance
                            if not finishNext:
                                evals = evaluateModel(
                                    model, advDataLoader, dataset, modelData)
                                modelsEvals.extend(evals)
                                asr = evals['asr']
                                asr_0 = evals['asr_0']
                                asr_1 = evals['asr_1']
                                asr_history.append(asr)

                                # Estimating SSIM

                                ssims = []
                                ssim_measure = StructuralSimilarityIndexMeasure(
                                    data_range=1.0)

                                for (advBatch, _, advPaths), (testBatch, _, testPaths) in zip(advDataLoader, testDataLoader):
                                    ssims.append(ssim_measure(
                                        advBatch, testBatch))

                                mean_ssim = sum(ssims)/len(ssims)

                                print('\n\t[🖼️  SSIM]: {}'.format(
                                    round(float(mean_ssim), 2)))
                                print('\t[🎯 ASR]: {}'.format(
                                    round(float(asr), 2)))
                                print('\t\t[ASR_0]: {}'.format(
                                    round(float(asr_0), 2)))
                                print('\t\t[ASR_1]: {}\n'.format(
                                    round(float(asr_1), 2)))

                                csv_data.append({
                                    'attack': attack,
                                    'task': currentTask,
                                    'model': modelName,
                                    'balance': modelPercents,
                                    'dataset': modelDataset,
                                    'ssim': mean_ssim.item(),
                                    'eps': eps,
                                    'asr': asr,
                                    'asr_0': asr_0,
                                    'asr_1': asr_1
                                })

                            if eps >= eps_thresholds_non_math[attack_name]:
                                effective = True
                            else:
                                eps += steps_non_math[attack_name]
                                eps = round(eps, 3)

                            i += 1

                            torch.cuda.empty_cache()

    data_df = pd.DataFrame(csv_data)
    if not os.path.exists('./results/attacks/history/' + currentTask + '/'):
        os.makedirs('./results/attacks/history/' + currentTask + '/')
    data_df.to_csv('./results/attacks/history/' + currentTask + '/' + attack + '.csv')