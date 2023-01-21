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
from utils.helperFunctions import *
from utils.const import *

# Ranges and step for attack epsilon

attacksParams = {
    "math": {
        "BIM": {"init": 0.01, "steps": 0.01, "threshold": 0.3},
        "FGSM": {"init": 0.01, "steps": 0.01, "threshold": 0.3},
        "DeepFool": {"init": 10, "steps": 1, "threshold": 100},
        "RFGSM": {"init": 0.01, "steps": 0.01, "threshold": 0.3},
        "PGD": {"init": 0.01, "steps": 0.01, "threshold": 0.3},
        "TIFGSM": {"init": 0.01, "steps": 0.01, "threshold": 0.3}
    },
    "nonmath": {
        "GaussianNoise": {"init": 0.005,    "steps": 0.005,    "threshold": 0.1},
        "BoxBlur": {"init": 0.5,    "steps": 0.5,    "threshold": 10},
        "Sharpen": {"init": 1,    "steps": 0,    "threshold": 1},
        "InvertColor": {"init": 1,    "steps": 0,    "threshold": 1},
        "GreyScale": {"init": 1,    "steps": 0,    "threshold": 1},
        "SplitMergeRGB": {"init": 1,    "steps": 0,    "threshold": 1},
        "SaltPepper": {"init": 0.005,    "steps": 0.005,    "threshold": 0.1},
        "RandomBlackBox": {"init": 10,    "steps": 10,    "threshold": 200}
    }
}

# attacks["math"].keys()

nonMathAttacks = NonMathAttacks()

# Parameters

SHUFFLE_DATASET = False  # Shuffle the dataset

if not os.path.exists(os.path.join(os.getcwd(), ADVERSARIAL_DIR)):
    os.makedirs(os.path.join(os.getcwd(), ADVERSARIAL_DIR))

dfMath = pd.read_csv(MODEL_PREDICTIONS_PATH, index_col=[
                     "task", "model", "model_dataset", "balance", "dataset"]).sort_index()

# Helper functions

modelsEvals = []

datasetsToGenerate = getSubDirs(DATASETS_DIR)

i = 0

csv_data = []

print("[🧠 MATH ATTACK GENERATION]\n")

for attack_name in attacksParams["math"].keys():
    currentAttackParams = attacksParams["math"][attack_name]

    csv_data = []
    for dataset in sorted(datasetsToGenerate):

        print("\n" + "-" * 15)
        print("[🗃️  SOURCE DATASET] {}\n".format(dataset))

        datasetDir = os.path.join(DATASETS_DIR, dataset)
        testDir = os.path.join(datasetDir, "test")

        datasetAdvDir = os.path.join(ADVERSARIAL_DIR, dataset)
        mathAttacksDir = os.path.join(datasetAdvDir, "math")

        if not os.path.exists(mathAttacksDir):
            os.makedirs(mathAttacksDir)

        toTensor = transforms.Compose([transforms.ToTensor()])
        toNormalizedTensor = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_PARAMS)
        ])

        for root, _, fnames in sorted(os.walk(os.path.join(MODELS_DIR, dataset), followlinks=True)):
            for fname in sorted(fnames):
                eps = currentAttackParams["init"]
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

                            setSeed()
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
                                saveDir, transform=toNormalizedTensor, datasetSize=DATASET_SIZE, use_cache=False, check_images=False, with_path=True)

                            setSeed()
                            advDataLoader = DataLoader(
                                advDataset, batch_size=16, num_workers=0, shuffle=SHUFFLE_DATASET)

                            evals = evaluateModel(
                                model, advDataLoader, dataset, modelData, dfMath)

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

                            if eps >= currentAttackParams["threshold"]:
                                effective = True
                            else:
                                eps += currentAttackParams["steps"]
                                eps = round(eps, 2)

                            i += 1

                            torch.cuda.empty_cache()

    data_df = pd.DataFrame(csv_data)

    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)

    data_df.to_csv(os.path.join(HISTORY_DIR, attack_name + '.csv'))


print("\n\n[🧠 NON-MATH ATTACK GENERATION]\n")

for attack_name in attacksParams["nonmath"].keys():
    currentAttackParams = attacksParams["nonmath"][attack_name]

    csv_data = []
    for dataset in sorted(datasetsToGenerate):

        print("\n" + "-" * 15)
        print("[🗃️  SOURCE DATASET] {}\n".format(dataset))

        datasetDir = os.path.join(DATASETS_DIR, dataset)
        testDir = os.path.join(datasetDir, "test")

        datasetAdvDir = os.path.join(ADVERSARIAL_DIR, dataset)
        nonMathAttacksDir = os.path.join(datasetAdvDir, "nonMath")

        if not os.path.exists(nonMathAttacksDir):
            os.makedirs(nonMathAttacksDir)

        toTensor = transforms.Compose([transforms.ToTensor()])
        toNormalizedTensor = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_PARAMS)
        ])

        for root, _, fnames in sorted(os.walk(os.path.join(MODELS_DIR, dataset), followlinks=True)):
            for fname in sorted(fnames):
                eps = currentAttackParams["init"]
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
                                saveDir2, transform=toNormalizedTensor, datasetSize=DATASET_SIZE, use_cache=False, check_images=False, with_path=True)
                            setSeed()
                            advDataLoader = DataLoader(
                                advDataset, batch_size=16, num_workers=0, shuffle=SHUFFLE_DATASET)

                            # If finishNext that means that we have already found
                            # the best eps for our task. We will then only generate
                            # those adversarial samples again with no need of
                            # evaluating its performance
                            if not finishNext:
                                evals = evaluateModel(
                                    model, advDataLoader, dataset, modelData, dfMath)
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

                            if eps >= currentAttackParams["threshold"]:
                                effective = True
                            else:
                                eps += currentAttackParams["steps"]
                                eps = round(eps, 3)

                            i += 1

                            torch.cuda.empty_cache()

    data_df = pd.DataFrame(csv_data)

    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)

    data_df.to_csv(os.path.join(HISTORY_DIR, attack_name + '.csv'))
