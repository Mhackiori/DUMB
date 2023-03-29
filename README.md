<div id="top"></div>
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Mhackiori/Adversarial-Transferability">
    <img src="https://i.postimg.cc/P5m5r2sX/cat-no-bg.png" alt="Logo" width="150" height="150">
  </a>

  <h1 align="center">Your Attack Is Too DUMB</h1>

  <p align="center">Formalizing Attacker Scenarios for Adversarial Transferability
    <br />
    <a href=""><strong>Paper in progress »</strong></a>
    <br />
    <br />
    <a href="https://github.com/MarcoAlecci">Marco Alecci</a>
    ·
    <a href="https://www.math.unipd.it/~conti/">Mauro Conti</a>
    ·
    <a href="https://www.math.unipd.it/~fmarchio/">Francesco Marchiori</a>
    ·
    <a href="https://lucamartinelli.eu.org/">Luca Martinelli</a>
    ·
    <a href="https://sites.google.com/view/lucapajola/home">Luca Pajola</a>
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
        <ul>
          <li><a href="#mathematical">Mathematical Attacks</a></li>
          <li><a href="#nonmathematical">Non Mathematical Attacks</a></li>
          <li><a href="#tuning">Parameter Tuning</a></li>
        </ul>
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

>Evasion attacks are a threat to machine learning models, where adversaries attempt to affect classifiers by injecting malicious samples. An alarming side-effect of evasion attacks is their ability to transfer among different models: this property is called _transferability_. Therefore, an attacker can produce adversarial samples on a custom model to conduct the attack on a victim's organization later. Although literature widely discusses how adversaries can transfer their attacks, their experimental settings are limited and far from reality. For instance, many experiments consider both attacker and defender sharing the same dataset, balance level (i.e., how the ground truth is distributed), and model architecture. In this work, we propose the **DUMB** attacker model. This framework allows analyzing if evasion attacks fail to transfer when the training conditions of surrogate and victim models differ. DUMB considers the following conditions: **D**ataset so**U**rces, **M**odel architecture, and the **B**alance of the ground truth. We then propose a novel testbed to evaluate many state-of-the-art evasion attacks with DUMB; the testbed consists of three computer vision tasks with two distinct datasets each, four types of balance levels, and three model architectures. Our analysis, which generated 13K tests over 14 distinct attacks, led to numerous novel findings in the scope of transferable attacks with surrogate models. In particular, mismatches between attackers and victims in terms of dataset source, balance levels, and model architecture lead to non-negligible loss of attack performance.

<p align="right"><a href="#top">(back to top)</a></p>
<div id="usage"></div>

## ⚙️ Usage

First, start by cloning the repository.

```bash
git clone https://github.com/Mhackiori/Adversarial-Transferability.git
cd Adversarial-Transferability
```
<sup>NOTE: if you're accessing this data from the anonymized repository, the above command might not work. Instead, you can download the repository from [here](https://figshare.com/s/2e80e7a5c603577fc38a).</sup>

Then, install the required Python packages by running:

```bash
pip install -r requirements.txt
```

You now need to add the datasets in the repository. You can do this by downloading the zip file [here](https://figshare.com/s/c5a05acaa9f75a3b05bf) and extracting it in this repository.

To replicate the results in our paper, you need to execute the scripts in a specific order (`modelTrainer.py`, `attackGeneration.py` and `evaluation.py`), or you can execute them one after another by running the dedicated shell script.

```bash
chmod +x ./run.sh && ./run.sh
```

If instead you want to run each script one by one, you will need to specify the task through an environment variable.

* **TASK=0**: `[bike, motorbike]`
* **TASK=1**: `[cat, dog]`
* **TASK=2**: `[man, woman]`

```bash
export TASK=0 && python3 modelTrainer.py
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

<div id="mathematical"></div>

#### **🔢 Mathematical Attacks**
These are the adversarial attacks that is possible to find in the literature. In particular, we use: BIM, DeepFool, FGSM, PGD, RFGSM, and TIFGSM. In the next Table, we include their correspondent papers and the parameter that we use for fine-tuning the attack. We also show the range in which the parameter has been tested and the step for the search (see <a href="#tuning">Parameter Tuning</a>).

| Name              | Paper                                                                                                                                          | Parameter  |
|:-----------------:|------------------------------------------------------------------------------------------------------------------------------------------------|:----------:|
| **BIM**<br />(Linf)    | Adversarial Examples in the Physical World ([Kurakin et al., 2016](https://arxiv.org/abs/1607.02533))                                          | $\epsilon \in [0.01, 0.3]$<br />@ 0.01 step |
| **DeepFool**<br />(L2) | DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks ([Moosavi-Dezfooli et al., 2016](https://arxiv.org/abs/1511.04599))        | Overshoot $\in [10, 100]$<br />@ 1 step |
| **FGSM**<br />(Linf)   | Explaining and harnessing adversarial examples ([Goodfellow et al., 2014](https://arxiv.org/abs/1412.6572))                                    | $\epsilon \in [0.01, 0.3]$<br />@ 0.01 step |
| **PGD**<br />(Linf)    | Towards Deep Learning Models Resistant to Adversarial Attacks ([Mardry et al., 2017](https://arxiv.org/abs/1706.06083))                        | $\epsilon \in [0.01, 0.3]$<br />@ 0.01 step |
| **RFGSM**<br />(Linf)  | Ensemble Adversarial Traning: Attacks and Defences ([Tramèr et al., 2017](https://arxiv.org/abs/1705.07204))                                   | $\epsilon \in [0.01, 0.3]$<br />@ 0.01 step |
| **Square**<br />(Linf, L2)    | Square Attack: a query-efficient black-box adversarial attack via random search ([Andriushchenko et al., 2019](https://arxiv.org/abs/1912.00049))                        | $\epsilon \in [0.1, 0.3]$<br />@ 0.05 step |
| **TIFGSM**<br />(Linf) | Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks ([Dong et al., 2019](https://arxiv.org/abs/1904.02884)) | $\epsilon \in [0.01, 0.3]$<br />@ 0.01 step |

<sup>NOTE: if you're accessing this data from the anonymized repository, the above formulae might not be displayed correctly. This issue might be present in other places in this repository, but can be solved by rendering the Markdown code locally after downloading the repository.</sup>

<div id="nonmathematical"></div>

#### **🎨 Non Mathematical Attacks**
These are just visual modifications of the image obtained through filters or other means. A description of the individual attacks is provided in the next Table. If present, we also specify which parameter we use to define the level of perturbation added to the image.  We also show the range in which the parameter has been tested and the step for the search (see <a href="#tuning">Parameter Tuning</a>).

| Name                         | Description                                                                                                                                                                                                                                                                                                                                         | Parameter   |
|:----------------------------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------:|
| **Box Blur**                     | By applying this filter it is possible to blur the image by setting each pixel to the average value of the pixels in a square box extending radius pixels in each direction                                                                                                                                                                         | $r \in [0.5, 10]$<br />@ 0.5 step     |
| **Gaussian Noise**               | A statistical noise having a probability density function equal to normal distribution                                                                                                                                                                                                                                                              | $\sigma \in [0.005, 0.1]$<br />@ 0.005 step    |
| **Grayscale Filter**             | To get a grayscale image, the color information from each RGB channel is removed, leaving only the luminance values. Grayscale images contain only shades of gray and no color because maximum luminance is white and zero luminance is black, so everything in between is a shade of gray                                                          | N.A.        |
| **Invert Color**                 | An image negative is produced by subtracting each pixel from the maximum intensity value, so for color images, colors are replaced by their complementary colors                                                                                                                                                                                    | N.A.        |
| **Random Black Box**             | We draw a black square in a random position inside the central portion of the image in order to cover some crucial information                                                                                                                                                                                                                      | Square size $\in [10, 200]$<br />@ 10 step |
| **Salt & Pepper**              | An image can be altered by setting a certain amount of the pixels in the image either black or white. The effect is similar to sprinkling white and black dots-salt and pepper-ones in the image                                                                                                                                                    | Amount $\in [0.05, 0.1]$<br />@ 0.005 step     |

In the next Figure we show an example for each of the non-mathematical attacks.
![Non Mathematical Attacks](https://i.postimg.cc/c1m5b4Tm/cat-Non-Math-Attacks.jpg "Non Mathematical Attacks")

<div id="tuning"></div>

#### **👾 Parameter Tuning**

All mathematical attacks and most of non mathematical attacks include some kind of parameter $\epsilon$ that can be used to tune the level of perturbation added to the image. In order to find the best tradeoff between Attack Success Rate (ASR) and visual quality, we generate each attack with different $\epsilon$ values and save its ASR on the same model on which it has been generated and its Structural Similarity Index Measure (SSIM).

<p align="right"><a href="#top">(back to top)</a></p>
<div id="evaluation"></div>

### 📋 Evaluation

After generating attacks at different $\epsilon$, we decide the best value for this parameter by maximizing the sum of the ASR and the SSIM. The optimization is given by the following equation:

$$\gamma = \arg \max_s \frac{1}{n}\sum_{i=1}^n f(x_i)\neq f(x_i^a) \mathrm{\quad subject \ to \quad} \frac{1}{n}\sum_{i=1}^n SSIM(x_i, x_i^a) \geq \alpha$$

In the notation, $f$ is the model owned by the attacker and used during the optimization process, $x^a$ is the adversarial samples derived by $\mathcal{A}(f, x; s)$, and $\mathcal{A}$ is the adversarial procedure with parameter $s$.

With [`evaluation.py`](https://github.com/Mhackiori/Adversarial-Transferability/blob/main/evaluation.py) we are evaluating all the adversarial samples that we generated on all the different models that we trained.

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

In this work, we enhance the transferability landscape by proposing the DUMB attacker model, a list of 8 scenarios that might occur during an attack by including the three variables of the dataset, class balance, and model architecture.

| Case | Condition                                              | Attack Scenario                                                                                                                                                                                                                                                                           |
|------|--------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| C1   | $D_a = D_v$<br />$B_a = B_v$<br />$M_a = M_v$          | The ideal case for an attacker. We identified two potential attack scenarios. (i) Attackers legally or illegally gain information about the victims' system. (ii) Attackers and victims use the state-of-the-art.                                            |
| C2   | $D_a = D_v$<br />$B_a \neq B_v$<br />$M_a = M_v$       | Attackers and victims use state-of-the-art datasets and model architecture. However, victims modify the class balance to boost the model's performance. This scenario can occur especially with imbalanced datasets.                                                                                    |
| C3   | $D_a = D_v$<br />$B_a = B_v$<br />$M_a \neq M_v$       | Attackers and victims use standard datasets to train their models. However, there is a mismatch in the model architecture. This scenario might occur when state-of-the-art presents many comparable models. Or similarly, the victims choose a specific model based on computational constraints.                                                                                                                                                                        |
| C4   | $D_a = D_v$<br />$B_a \neq B_v$<br />$M_a \neq M_v$    | Attackers and victims use standard datasets to train their models, while models' architectures differ. Furthermore, victims adopt data augmentation or preprocessing techniques that alter the ground truth distribution (balancing). This scenario can occur especially with imbalanced datasets.         |
| C5   | $D_a \neq D_v$<br />$B_a = B_v$<br />$M_a = M_v$       | Attackers and victims use different datasets to accomplish the same classification task. The ground truth distribution can be equal, especially in tasks that are inherently balanced. Similarly, models can be equal if they both adopt the state-of-the-art.                                                   |
| C6   | $D_a \neq D_v$<br />$B_a \neq B_v$<br />$M_a = M_v$    | Attackers and victims use different datasets to accomplish the same classification task. Datasets have different balancing because they are inherently generated in different ways (e.g., see hate speech datasets example) or because the attackers or victims augmented them. Attackers and victims use the same state-of-the-art architecture.  |
| C7   | $D_a \neq D_v$<br />$B_a = B_v$<br />$M_a \neq M_v$    | Attackers and victims use different datasets to accomplish the same classification task. Datasets ground truth distribution matches. Attackers and victims use different models' architecture.                                                                                                 |
| C8   | $D_a \neq D_v$<br />$B_a \neq B_v$<br />$M_a \neq M_v$ | The worst-case scenario for an attacker. Attackers do not match the victims' dataset, balancing, and model architecture.                                                                                                                                      |

<p align="right"><a href="#top">(back to top)</a></p>