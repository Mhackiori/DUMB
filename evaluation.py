import numpy as np
import os
import sys

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchattacks import FGSM, DeepFool, BIM, RFGSM, PGD, Square, TIFGSM
import torchvision
from torchvision import transforms

from utils.balancedDataset import BalancedDataset
from utils.const import *
from utils.helperFunctions import *
from utils.nonMathAttacks import NonMathAttacks

import warnings
warnings.filterwarnings("ignore")

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

# Parameters

NON_MATH_ATTACKS = NonMathAttacks()

SHUFFLE_DATASET = False  # Shuffle the dataset

# Parameters for best eps estimation
ALPHA = 0.6
BETA = 1 - ALPHA

# If true, maximize gamma function
# If false, take eps which gives maximum ASR when SSIM is over a threshold
useGamma = False
threshold = 0.4

if not os.path.exists(os.path.join(os.getcwd(), ADVERSARIAL_DIR)):
    os.makedirs(os.path.join(os.getcwd(), ADVERSARIAL_DIR))

dfMath = pd.read_csv(MODEL_PREDICTIONS_PATH, index_col=[
                     "task", "model", "model_dataset", "balance", "dataset"]).sort_index()

# Setting seed for reproducibility

setSeed()

# Helper functions


def evaluateModelsOnDataset(datasetFolder, datasetInfo):
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
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZATION_PARAMS[0], NORMALIZATION_PARAMS[1])
    ])

    testDataset = BalancedDataset(
        datasetFolder, transform=dataTransform, use_cache=False, check_images=False, with_path=True)

    setSeed()
    testDataLoader = DataLoader(
        testDataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

    # Evaluate every model
    for root, _, fnames in sorted(os.walk(MODELS_DIR, followlinks=True)):
        for fname in sorted(fnames):

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
            modelToTest = modelToTest.to(DEVICE, non_blocking=True)

            scores = evaluateModel(
                modelToTest, testDataLoader, modelDataset, modelData, dfMath)

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

datasetsToGenerate = getSubDirs(DATASETS_DIR)

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
    'SplitMergeRGB',
    'Square',
    'TIFGSM'
]

attacks_names_math = [
    'BIM',
    'DeepFool',
    'FGSM',
    'PGD',
    'RFGSM',
    'Square',
    'TIFGSM'
]

attack_names_static = [
    'GreyScale',
    'InvertColor',
    'SplitMergeRGB'
]

print("[🧠 GENERATING BEST EPS FOR EACH ATTACK]\n")

best_eps_data = []

for attack_name in attacks_names:
    for dataset in sorted(datasetsToGenerate):

        print("\n" + "-" * 15)
        print("[🗃️  SOURCE DATASET] {}\n".format(dataset))

        datasetDir = os.path.join(DATASETS_DIR, dataset)
        testDir = os.path.join(datasetDir, "test")

        datasetAdvDir = os.path.join(ADVERSARIAL_DIR, dataset)
        mathAttacksDir = os.path.join(datasetAdvDir, "math")
        nonMathAttacksDir = os.path.join(datasetAdvDir, "nonMath")

        if not os.path.exists(mathAttacksDir):
            os.makedirs(mathAttacksDir)
        if not os.path.exists(nonMathAttacksDir):
            os.makedirs(nonMathAttacksDir)

        toTensor = transforms.Compose([transforms.ToTensor()])
        toNormalizedTensor = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_PARAMS[0], NORMALIZATION_PARAMS[1])
        ])

        for root, _, fnames in sorted(os.walk(os.path.join(MODELS_DIR, dataset), followlinks=True)):
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
                model = modelData["model"].to(DEVICE)

                # Test dataset without normalization (for generating samples)
                originalTestDataset = BalancedDataset(
                    testDir, transform=toTensor, datasetSize=DATASET_SIZE, use_cache=False, check_images=False, with_path=True)

                setSeed()
                originalTestDataLoader = DataLoader(
                    originalTestDataset, batch_size=16, num_workers=0, shuffle=SHUFFLE_DATASET)

                # Test dataset with normalization (for evaluation)
                testDataset = BalancedDataset(
                    testDir, transform=toNormalizedTensor, datasetSize=DATASET_SIZE, use_cache=False, check_images=False, with_path=True)

                setSeed()
                testDataLoader = DataLoader(
                    testDataset, batch_size=16, num_workers=0, shuffle=SHUFFLE_DATASET)

                # Loading best epsilon value for this model
                best_df = pd.read_csv(os.path.join(
                    HISTORY_DIR, attack_name + '.csv'), index_col='Unnamed: 0')

                df_atk = best_df[best_df['model'] == modelName]
                df_atk = df_atk[df_atk['dataset'] == modelDataset]
                df_atk = df_atk[df_atk['balance'] == modelPercents]

                epss = list(df_atk['eps'])
                asrs = list(df_atk['asr'])
                ssims = list(df_atk['ssim'])

                best = []
                max_eps_idx = 0
                for j in range(len(epss)):
                    if useGamma:
                        best.append((ALPHA * asrs[j]) + (BETA * ssims[j]))
                    else:
                        if ssims[j] > threshold and asrs[j] >= asrs[max_eps_idx]:
                            eps = epss[j]
                            max_eps_idx = j

                if useGamma:
                    maxx = max(best)
                    best_index = best.index(maxx)
                    eps = epss[best_index]

                attacks = {
                    "BIM": BIM(model, eps=eps),
                    "BoxBlur": NON_MATH_ATTACKS.boxBlur,
                    "FGSM": FGSM(model, eps=eps),
                    "GaussianNoise": NON_MATH_ATTACKS.gaussianNoise,
                    "GreyScale": NON_MATH_ATTACKS.greyscale,
                    "InvertColor": NON_MATH_ATTACKS.invertColor,
                    "DeepFool": DeepFool(model, overshoot=eps),
                    "PGD": PGD(model, eps=eps),
                    "RandomBlackBox": NON_MATH_ATTACKS.randomBlackBox,
                    "RFGSM": RFGSM(model, eps=eps),
                    "SaltPepper": NON_MATH_ATTACKS.saltAndPepper,
                    "SplitMergeRGB": NON_MATH_ATTACKS.splitMergeRGB,
                    "Square": Square(model, eps=eps),
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

                            print("\n[⚔️  ADVERSARIAL] {} @ {} - {} - {} {}".format(
                                attack,
                                eps,
                                modelDataset,
                                modelName,
                                modelPercents
                            ))

                            setSeed()
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
                            best_eps_data.append({
                                'attack': attack_name,
                                'model': modelName,
                                'dataset': modelDataset,
                                'balance': modelPercents,
                                'best_eps': eps
                            })
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

eps_df = pd.DataFrame(best_eps_data)
eps_df.to_csv(os.path.join(HISTORY_DIR, 'all_eps.csv'))


print("\n\n[🧠 ATTACKS EVALUATION]\n")

modelsEvals = []

for attack in sorted(attacks_names):
    modelsEvals = []
    # Evaluate models on math attacks folders
    for dataset in sorted(getSubDirs(ADVERSARIAL_DIR)):
        datasetDir = os.path.join(ADVERSARIAL_DIR, dataset)
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

    modelsEvalsDF = pd.DataFrame(modelsEvals)

    if not os.path.exists(EVALUATIONS_DIR):
        os.makedirs(EVALUATIONS_DIR)

    modelsEvalsDF.to_csv(os.path.join(
        EVALUATIONS_DIR, 'evaluations_' + attack + '.csv'))