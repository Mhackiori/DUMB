import numpy as np
import os

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchattacks import FGSM, DeepFool, BIM, RFGSM, PGD, TIFGSM
import torchvision
from torchvision import transforms

from utils.balancedDataset import BalancedDataset
from utils.nonMathAttacks import NonMathAttacks
from utils.tasks import currentTask


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters

nonMathAttacks = NonMathAttacks()

datasetToFolder = {"bing": "bing", "google": "google"}

shuffleDataset = False  # Shuffle the dataset

inputSize = 224  # Specified for alexnet, resnet, vgg
datasetSize = 150  # Reduce the size of the dataset

adversarialDir = "./adversarialSamples/" + currentTask

if not os.path.exists(os.path.join(os.getcwd(), adversarialDir)):
    os.makedirs(os.path.join(os.getcwd(), adversarialDir))

datasetsDir = "./datasets/" + currentTask
modelsDir = "./models/" + currentTask
adversarialsDir = "./adversarialSamples/" + currentTask

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


def evaluateModelsOnDataset(datasetFolder, datasetInfo):
    global modelsDir, inputSize

    modelsEvals = []

    # Get the images and calculate mean and standard deviation
    imageDataset = torchvision.datasets.ImageFolder(
        datasetFolder, transform=transforms.Compose([transforms.ToTensor()]))

    for cls in imageDataset.classes:
        cls_index = imageDataset.class_to_idx[cls]
        num_cls = np.count_nonzero(
            np.array(imageDataset.targets) == cls_index)

        print("\t[🧮 # ELEMENTS] {}: {}".format(cls, num_cls))

    # Setup for normalization
    dataTransform = transforms.Compose([
        transforms.Resize(inputSize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    testDataset = BalancedDataset(
        datasetFolder, transform=dataTransform, use_cache=False, check_images=False, with_path=True)

    setSeed(SEED)
    testDataLoader = DataLoader(
        testDataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
 

    # Evaluate every model
    for root, _, fnames in sorted(os.walk(modelsDir, followlinks=True)):
        for fname in sorted(fnames):
            # Putting testDataset here to keep paths
            # testDataset = BalancedDataset(
            #     datasetFolder, transform=dataTransform, use_cache=True, check_images=False, with_path=True)

            # setSeed(SEED)
            # testDataLoader = DataLoader(
            #     testDataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
            
            modelPath = os.path.join(root, fname)

            try:
                modelData = torch.load(modelPath)
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
            modelToTest = modelToTest.to(device, non_blocking=True)

            scores = evaluateModel(modelToTest, testDataLoader, modelDataset, modelData)

            modelsEvals.append({
                "source_dataset": datasetInfo["dataset"],
                "isMath": datasetInfo["math"],
                "attack": datasetInfo["attack"],
                "source_model": datasetInfo["model"],
                "source_balancing": datasetInfo["balancing"],

                "target_model": modelName,
                "target_dataset": modelDataset,
                "target_balancing": modelPercents,
                "asr": scores["asr"],
                "asr_0": scores["asr_0"],
                "asr_1": scores["asr_1"]
            })

            print("\t[ASR]: {:.4f}".format(scores["asr"]))
            print("\t\t[ASR_0]: {:.4f}".format(scores["asr_0"]))
            print("\t\t[ASR_1]: {:.4f}\n".format(scores["asr_1"]))

            torch.cuda.empty_cache()

    return modelsEvals


modelsEvals = []

datasetsToGenerate = getSubDirs(datasetsDir)

i = 0

attacks_names = [
    'BIM',
    'BoxBlur',
    'DeepFool',
    'FGSM',
    'GaussianNoise',
    'GreyScale',
    'InvertColor',
    'PGD',
    'RandomBlackBox',
    'RFGSM',
    'SaltPepper',
    'Sharpen',
    'SplitMergeRGB',
    'TIFGSM'
]

attacks_names_math = [
    'BIM',
    'DeepFool',
    'FGSM',
    'PGD',
    'RFGSM',
    'TIFGSM'
]

attack_names_static = [
    'GreyScale',
    'InvertColor',
    'Sharpen',
    'SplitMergeRGB'
]

print("[🧠 GENERATING BEST EPS FOR EACH ATTACK]\n")

for attack_name in attacks_names:
    for dataset in sorted(datasetsToGenerate):

        print("\n" + "-" * 15)
        print("[🗃️  SOURCE DATASET] {}\n".format(dataset))

        datasetDir = os.path.join(datasetsDir, dataset)
        testDir = os.path.join(datasetDir, "test")

        datasetAdvDir = os.path.join(adversarialDir, dataset)
        mathAttacksDir = os.path.join(datasetAdvDir, "math")
        nonMathAttacksDir = os.path.join(datasetAdvDir, "nonMath")

        if not os.path.exists(mathAttacksDir):
            os.makedirs(mathAttacksDir)
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

                # Loading best epsilon value for this model
                csv_path = './results/attacks/history/' + currentTask + '/' + attack_name + '.csv'
                best_df = pd.read_csv(csv_path, index_col='Unnamed: 0')

                df_atk = best_df[best_df['model'] == modelName]
                df_atk = df_atk[df_atk['dataset'] == modelDataset]
                df_atk = df_atk[df_atk['balance'] == modelPercents]

                epss = list(df_atk['eps'])
                asrs = list(df_atk['asr'])
                ssims = list(df_atk['ssim'])

                best = []
                for j in range(len(epss)):
                    best.append(asrs[j] + ssims[j])

                maxx = max(best)
                best_index = best.index(maxx)
                eps = epss[best_index]

                attacks = {
                    "BIM": BIM(model, eps=eps),
                    "BoxBlur": nonMathAttacks.boxBlur,
                    "FGSM": FGSM(model, eps=eps),
                    "GaussianNoise": nonMathAttacks.gaussianNoise,
                    "GreyScale": nonMathAttacks.greyscale,
                    "InvertColor": nonMathAttacks.invertColor,
                    "DeepFool": DeepFool(model, overshoot=eps),
                    "PGD": PGD(model, eps=eps),
                    "RandomBlackBox": nonMathAttacks.randomBlackBox,
                    "RFGSM": RFGSM(model, eps=eps),
                    "SaltPepper": nonMathAttacks.saltAndPepper,
                    "Sharpen": nonMathAttacks.sharpen,
                    "SplitMergeRGB": nonMathAttacks.splitMergeRGB,
                    "TIFGSM": TIFGSM(model, eps=eps)
                }

                for attack in attacks:
                    if attack == attack_name:
                        # Mathematical attacks
                        if attack in attacks_names_math:
                            attacker = attacks[attack]

                            attackDir = os.path.join(
                                mathAttacksDir, attack)
                            saveDir = os.path.join(
                                attackDir, modelName + "/" + modelPercents)

                            if not os.path.exists(saveDir):
                                os.makedirs(saveDir)

                            print("[⚔️  ADVERSARIAL] {} @ {} - {} - {} {}".format(
                                attack,
                                eps,
                                modelDataset,
                                modelName,
                                modelPercents
                            ))

                            setSeed(SEED)
                            saveMathAdversarials(
                                originalTestDataLoader, originalTestDataset.classes, attacker, saveDir)
                        # Non mathematical attacks of which a parameter have been grid-searched
                        elif attack not in attack_names_static:
                            print("[⚔️  ADVERSARIAL] {} @ {} - {} - {} {}".format(
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
                                outImage = attacker(outImage, amount=eps)
                                outImage.save(os.path.join(
                                    saveDir, imageName), "JPEG")

                            print(f"\t[💾 IMAGES SAVED]")


print("\n\n[🧠 ATTACKS EVALUATION]\n")

modelsEvals = []

for attack in sorted(attacks_names):
    modelsEvals = []
    # Evaluate models on math attacks folders
    for dataset in sorted(getSubDirs(adversarialsDir)):
        datasetDir = os.path.join(adversarialsDir, dataset)
        mathAdvDir = os.path.join(datasetDir, "math")
        nonMathAdvDir = os.path.join(datasetDir, "nonMath")

        if not os.path.exists(mathAdvDir):
            continue

        if attack in attacks_names_math:
            attackDir = os.path.join(mathAdvDir, attack)
            isMath = True
        else:
            attackDir = os.path.join(nonMathAdvDir, attack)
            isMath = False

        for advModel in sorted(getSubDirs(attackDir)):
            advModelDir = os.path.join(attackDir, advModel)

            for advBalancing in sorted(getSubDirs(advModelDir)):
                advDatasetDir = os.path.join(advModelDir, advBalancing)

                print("\n" + "-" * 15)
                print("[🗃️ ADVERSARIAL DATASET] {}/{}/{}/{}".format(dataset,
                      attack, advModel, advBalancing))

                advDatasetInfo = {
                    "dataset": dataset,
                    "math": isMath,
                    "attack": attack,
                    "balancing": advBalancing.replace("_", "/"),
                    "model": advModel,
                }

                evals = evaluateModelsOnDataset(advDatasetDir, advDatasetInfo)
                modelsEvals.extend(evals)

    ModelsEvalsDF = pd.DataFrame(modelsEvals)
    if not os.path.exists('./results/attacks/evaluation/' + currentTask + '/'):
        os.makedirs('./results/attacks/evaluation/' + currentTask + '/')
    csv_path_name = './results/attacks/evaluation/' + currentTask + '/evaluations_' + attack + '.csv'
    ModelsEvalsDF.to_csv(csv_path_name)