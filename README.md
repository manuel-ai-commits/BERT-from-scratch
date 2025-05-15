# **🧠 BERT from Scratch — PyTorch Implementation**

This repository provides a complete from-scratch implementation of BERT (Bidirectional Encoder Representations from Transformers) in PyTorch. It supports both **pretraining** and **fine-tuning** phases. Both for cuda and mps.
- For **pretraining**, the model is trained on the **Wikipedia** dataset, standard training corpus for BERT architectures. Trained using the standard BERT objectives: **Masked Language Modeling (MLM)** and **Next Sentence Prediction (NSP)**.
- For **fine-tuning**, the model can be adapted for downstream tasks such as **sentence pair classification**, using datasets like **Quora Question Pairs** (for duplicate question detection) or **Wikipedia** (for NSP-based binary classification). More can be added as long as they respect the dataset getitem sctructure of the others, find in the datasets.py

This implementation is modular and configurable via a single config.yaml file and integrates with **Weights & Biases (wandb)](https://wandb.ai)** for experiment tracking.


## **🔍 What is BERT?**

**BERT** is a transformer-based model developed by Google that learns deep contextualized word representations by jointly conditioning on both left and right context in all layers. It is pretrained using two unsupervised tasks:
- **Masked Language Modeling (MLM):** Randomly masks some tokens and learns to predict them.
- **Next Sentence Prediction (NSP):** Predicts if a sentence B follows sentence A in a pair.

Once pretrained, BERT can be fine-tuned for downstream tasks like classification, question answering, etc.

## **⚙️ Configuration and Setup**

All settings are controlled via the config.yaml file.

### 🧪 **Weights & Biases Setup (optional, but recommended)**

To enable experiment tracking:

1. Create an account at [Weights & Biases](https://wandb.ai).
2. 	Run wandb login to link your local environment.
3. 	Set your project name in config.yaml under:

project_name: "BERT"

<pre>
```config.yaml
project_name: "BERT"
```
</pre>


### **📌 Choose a Mode**

Set the mode by changing the following flag:

<pre>
```config.yaml
pretraining: True   # Pretraining mode
```
</pre>
- True → Pretraining
- False → Fine-tuning


### **🧠 Pretraining Mode**

Set pretraining: True and configure the following under the input section:

<pre>
```config.yaml
dataset_name: "wikipedia"
dataset_version: "20220301.en"
dataset_vocab: "wikipedia"
vocab_size: 30522
train_corpus_lines: 100000
on_memory: True
```
</pre>

Other key parameters:
- seq_len: Sequence length (e.g., 30 or 128)
- batch_size: Usually 32–128
- epochs: Number of training epochs
- learning_rate, weight_decay, betas: Optimizer settings

The model will be trained on **Masked Language Modeling** and **Next Sentence Prediction**.

### **🏁 Fine-tuning Mode**

Set pretraining: False to switch to **fine-tuning**.

Then, configure:

<pre>
```config.yaml
fine_tune:
task: "classification"               # downstream task
num_classes: 2                       # adjust for your dataset
pretrained_weights_path: ./models/pretrained.bin  # weights from pretraining
freeze_bert: True                    # whether to freeze BERT during fine-tuning
```
</pre>

## Run your code

run the code using

<pre>
python main.py
</pre>

_Enjoy!_ Feedbacks are appreciated 




