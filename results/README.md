# 📊 Results

In this folder, you will find the data generated from the execution of the three scripts. In particular, we divide our processing in two different steps.

<!-- TABLE OF CONTENTS -->
<details>
  <summary><strong>Folder Structure</strong></summary>
  <ol>
    <li>
      <a href="#models">Models</a>
      <ul>
        <li><a href="#baseline">Baseline</a></li>
        <li><a href="#predictions">Predictions</a></li>
        <li><a href="#similarity">Similarity</a></li>
      </ul>
    </li>
    <li>
      <a href="#attacks">Attacks</a>
      <ul>
        <li><a href="#history">history</a></li>
        <li><a href="#evaluation">Evaluation</a></li>
      </ul>
    </li>
  </ol>
</details>

<div id="models"></div>

## 🤖 Models

In the [`models`](https://github.com/Mhackiori/Adversarial-Transferability/tree/main/results/models) folder, you will find two different subfolders regarding the baseline evaluation of the models and the predictions for each of the images in the test set.

<div id="baseline"></div>

### 🧪 Baseline

In the [`baseline`](https://github.com/Mhackiori/Adversarial-Transferability/tree/main/results/models/baseline) folder, you will find one `csv` file for each of the tasks. These files are produced when running [`modelTrainer.py`](https://github.com/Mhackiori/Adversarial-Transferability/blob/main/modelTrainer.py), which, after training each of the 24 models for each of the tasks, evaluates them on both test sets (`bing` test set and `google` test set). This gives us a baseline evaluation of the models in order to understand the complexity of the different tasks.

<div id="predictions"></div>

### 🔮 Predictions

In the [`baseline`](https://github.com/Mhackiori/Adversarial-Transferability/tree/main/results/models/predictions) folder, you will find one `csv` file for each of the tasks. These files are produced when running [`modelTrainer.py`](https://github.com/Mhackiori/Adversarial-Transferability/blob/main/modelTrainer.py) and contains the predictions computed by every trained model for every image in the test sets. This will be used in the attack generation step in order to evaluate the ASR only on those samples that were correctly classified by the target model, and thus ignore the ones that were already misclassified.

<div id="similarity"></div>

### 👥 Similarity

In the [`similarity`](https://github.com/Mhackiori/Adversarial-Transferability/tree/main/results/models/similarity) folder, you will find one `csv` file for each of the tasks. These files are produced when running [`modelTrainer.py`](https://github.com/Mhackiori/Adversarial-Transferability/blob/main/modelTrainer.py) and contains the euclidian distance similarity of the embeddings of the two classes computed by the pretrained models (not fine tuned). Through the analysis of these values, we can justify the results obtained in [`baseline`](https://github.com/Mhackiori/Adversarial-Transferability/tree/main/results/models/predictions) and formalize the complexity of the different tasks.

<div id="attacks"></div>

## ⚔️ Attacks

In the [`models`](https://github.com/Mhackiori/Adversarial-Transferability/tree/main/results/attacks) folder, you will find two different subfolders regarding the generation of the attacks and their evaluation.

<div id="history"></div>

### 📜 History

In the [`history`](https://github.com/Mhackiori/Adversarial-Transferability/tree/main/results/attacks/history) folder, you will find for each task a list of `csv` files containing information gathered during the [parameter estimation](https://github.com/Mhackiori/Adversarial-Transferability#tuning) phase of the attack generation. For each combination of model, dataset and balance, we generate the attack samples at different $\epsilon$ values and evaluate the ASR on the same model. Furthermore, we evaluate the ASR on both the classes of the task and the mean SSIM value of the sample with respect to the original test set.

<div id="evaluation"></div>

### 🔍 Evaluation

In the [`history`](https://github.com/Mhackiori/Adversarial-Transferability/tree/main/results/attacks/evaluation) folder, you will find for each task a list of `csv` files containing information on the effectiveness of the attacks. In particular, we evaluate each of the attack samples on all the models (i.e., each combination of model, training dataset and balancing) in order to evaluate their transferability. 