<div id="top"></div>
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/luca-martinelli-09/learn-the-art">
    <img src="https://i.postimg.cc/P5m5r2sX/cat-no-bg.png" alt="Logo" width="150" height="150">
  </a>

  <h1 align="center">Enhancing Adversarial Transferability Landscape with the Rainbow Table</h1>

  <p align="center">
    <a href=""><strong>Paper in progress »</strong></a>
    <br />
    <br />
    <a href="https://github.com/MarcoAlecci">Marco Alecci</a>
    ·
    <a href="https://www.math.unipd.it/~conti/">Mauro Conti</a>
    ·
    <a href="https://github.com/Mhackiori">Francesco Marchiori</a>
    ·
    <a href="https://github.com/luca-martinelli-09">Luca Martinelli</a>
    ·
    <a href="https://github.com/pajola">Luca Pajola</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary><strong>Table of Contents</strong></summary>
  <ol>
    <li>
      <a href="#abstract">Abstract</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#modelTrainer">Model Trainer</a></li>
        <li><a href="#attackGeneration">Attacks Generation</a></li>
        <li><a href="#evaluation">Evaluation</a></li>
      </ul>
    </li>
    <li>
      <a href="#datasets">Datasets</a>
    </li>
    <li>
      <a href="#contribution">Contribution</a>
    </li>
  </ol>
</details>


<div id="abstract"></div>

## 🧩 Abstract

Work in progress

<p align="right"><a href="#top">(back to top)</a></p>
<div id="usage"></div>

## ⚙️ Usage

First, start by cloning the repository.

```bash
git clone https://github.com/Mhackiori/Adversarial-Transferability.git
cd Adversarial-Transferability
```

Then, install the required Python packages by running:

```bash
pip install -r requirements.txt
```

You now need to add the datasets in the repository. You can do this by downloading the folder [here]() and dropping it in this repository.

To replicate the results in our paper, you need to execute the scripts in a specific order, or you can execute them one aftern another by running:

```bash
python3 modelTrainer.py && python3 attackGeneration.py && python3 evaluation.py
```

<p align="right"><a href="#top">(back to top)</a></p>
<div id="modelTrainer"></div>

### 💪🏽 Model Trainer

With [`modelTrainer.py`](https://github.com/Mhackiori/Adversarial-Transferability/blob/main/modelTrainer.py) we are training three different model architectures on one of the tasks, which can be selected by changing the value of `currentTask` in [`tasks.py`](https://github.com/Mhackiori/Adversarial-Transferability/blob/main/utils/tasks.py). The three model architectures that we consider are the following.

|          Name          | Paper                                                        |
| :--------------------: | ------------------------------------------------------------ |
|**AlexNet**| One weird trick for parallelizing convolutional neural networks ([Krizhevskyet al., 2014](https://arxiv.org/abs/1404.5997)) |
|**ResNet**| Deep residual learning for image recognition ([He et al., 2016](https://arxiv.org/abs/1512.03385)) |
|**VGG**| Very deep convolutional networks for large-scale image recognition ([Simonyan et al., 2015](https://arxiv.org/abs/1409.1556)) |

The script will automatically handle the different balancing scenarios of the dataset and train 4 models for each architecture. The four different balancings affect only the training set and are described as follows:

* **[50/50]**: 3500 images for `class_0`, 3500 images for `class_1`
* **[40/60]**: 2334 images for `class_0`, 3500 images for `class_1`
* **[30/70]**: 1500 images for `class_0`, 3500 images for `class_1`
* **[20/80]**: 875 images for `class_0`, 3500 images for `class_1`

Once the 24 models are trained (2 datasets * 3 architectures * 4 dataset balancing), the same script will evaluate all of them in the same test set. First, we will save the [`predictions`](https://github.com/Mhackiori/Adversarial-Transferability/tree/main/results/models/predictions) of each of the images. This data will be used to create adversarial sample only from images that are correctly classified by the model. Then, we will save the evaluation of the models on the test set, which gives us a [`baseline`](https://github.com/Mhackiori/Adversarial-Transferability/tree/main/results/models/baseline) to evaluate the difficulty of the task.

<p align="right"><a href="#top">(back to top)</a></p>
<div id="attackGeneration"></div>

### ⚔️ Attacks Generation

With [`attackGeneration.py`](https://github.com/Mhackiori/Adversarial-Transferability/blob/main/attackGeneration.py) we are generating the attacks through the [Torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch) library and the [Pillow](https://pillow.readthedocs.io/en/stable/) library. Indeed, we divide the attacks in two main categories:

* **Mathematical Attacks**: these are the adversarial attacks that is possible to find in the literature. In particular, they are:

|              Name               | Paper                                                                                                                                                     |
|:-------------------------------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
|       **BIM**<br />(Linf)       | Adversarial Examples in the Physical World ([Kurakin et al., 2016](https://arxiv.org/abs/1607.02533))                                                     |
|     **DeepFool**<br />(L2)      | DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks ([Moosavi-Dezfooli et al., 2016](https://arxiv.org/abs/1511.04599))                   |
|      **FGSM**<br />(Linf)       | Explaining and harnessing adversarial examples ([Goodfellow et al., 2014](https://arxiv.org/abs/1412.6572))                                               |
|       **PGD**<br />(Linf)       | Towards Deep Learning Models Resistant to Adversarial Attacks ([Mardry et al., 2017](https://arxiv.org/abs/1706.06083))                                   |
|      **RFGSM**<br />(Linf)      | Ensemble Adversarial Traning: Attacks and Defences ([Tramèr et al., 2017](https://arxiv.org/abs/1705.07204))                                              |
|     **TIFGSM**<br />(Linf)      | Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks ([Dong et al., 2019](https://arxiv.org/abs/1904.02884))            |
* **Non Mathematical Attacks**: these are just visual modifications of the image obtained through filters or other means. In the next Figure a few examples are shown.
![Non Mathematical Attacks](https://i.postimg.cc/c1m5b4Tm/cat-Non-Math-Attacks.jpg "Non Mathematical Attacks")

All mathematical attacks and most of non mathematical attacks include some kind of parameter $\epsilon$ that can be used to tune the level of perturbation added to the image. In order to find the best tradeoff between Attack Success Rate (ASR) and visual quality, we generate each attack with different $\epsilon$ values and save its ASR on the same model on which it has been generated and its Structural Similarity Index Measure (SSIM).

<p align="right"><a href="#top">(back to top)</a></p>
<div id="evaluation"></div>

### 📋 Evaluation

After generating attacks at different $\epsilon$, we decide the best value for this parameter by maximizing the sum of the ASR and the SSIM. With [`evaluation.py`](https://github.com/Mhackiori/Adversarial-Transferability/blob/main/evaluation.py) we are evaluating all the adversarial samples that we generated on all the different models that we trained.

<p align="right"><a href="#top">(back to top)</a></p>
<div id="datasets"></div>

## 📚 Datasets

We created our own datasets by downloading images of cats and dogs from different sources ([Bing](https://bing.com) and [Google](https://google.com)). All images are then processed in order to be 300x300 in RGB using PIL. Images for each class are then sorted in the following format:

* **Trainig Set**: 3500 images
* **Validation Set**: 500 images
* **Test Set**: 1000 images

<p align="right"><a href="#top">(back to top)</a></p>
<div id="contribution"></div>

## 🤝 Contribution

In this work, we enhance the transferability landscape by proposing the rainbow table, a list of 9 scenarios that might occur during an attack by including the three variables of the dataset, class balance, and model architecture.

| Case | Condition                                              | Attack Scenario                                                                                                                                                                                                                                                                           |
|------|--------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| C1   | $D_a = D_v$<br />$B_a = B_v$<br />$M_a = M_v$          | White-box setting: the ideal case for an attacker. We identified two potential attack scenarios. (i) Attackers legally or illegally gain information about the victims' system. (ii) Attackers and victims use the state-of-the-art.                                                      |
| C2   | $D_a = D_v$<br />$B_a \neq B_v$<br />$M_a = M_v$       | Attackers and victims use the state-of-the-art. Furthermore, victims modify the training distribution to boost the model's performance. This scenario can occur, especially, with imbalanced datasets.                                                                                    |
| C3   | $D_a = D_v$<br />$B_a = B_v$<br />$M_a \neq M_v$       | Attackers and victims use standard datasets to train their models. However, here the models' architectures differ.                                                                                                                                                                        |
| C4   | $D_a = D_v$<br />$B_a \neq B_v$<br />$M_a \neq M_v$    | Attackers and victims use standard datasets to train their models, while models' architectures differ. Furthermore, victims adopt data augmentation or preprocessing techniques that alter the class distribution. This scenario can occur, especially, with imbalanced datasets.         |
| C5   | $D_a \neq D_v$<br />$B_a = B_v$<br />$M_a = M_v$       | Attackers and victims use different datasets to accomplish the same classification task.  The class distribution can be equal, especially in balanced datasets. Similarly, models can be equal if they both adopt the state-of-the-art.                                                   |
| C6   | $D_a \neq D_v$<br />$B_a \neq B_v$<br />$M_a = M_v$    | Attackers and victims use different datasets to accomplish the same classification task. Datasets have different class distributions because they are inherently generated in different ways (e.g., see hate speech datasets example) or because the attackers or victims augmented them. |
| C7   | $D_a \neq D_v$<br />$B_a = B_v$<br />$M_a \neq M_v$    | Attackers and victims use different datasets to accomplish the same classification task. Datasets classes distribution matches. Attackers and victims use different models' architecture.                                                                                                 |
| C8   | $D_a \neq D_v$<br />$B_a \neq B_v$<br />$M_a \neq M_v$ | Black-box settings: the worst-case scenario for an attacker. Attackers do not match the victims' dataset, class distribution, and model' architecture.                                                                                                                                    |

<p align="right"><a href="#top">(back to top)</a></p>