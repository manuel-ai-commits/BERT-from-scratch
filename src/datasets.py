from torch.utils.data import Dataset
import tqdm
import random
import torch 


class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding


        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                if corpus_lines is not None:
                    self.lines = [
                        line[:-1].split("\t")
                        for i, line in enumerate(tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines))
                        if i < corpus_lines
                    ]
                else:
                    self.lines = [
                        line[:-1].split("\t")
                        for line in tqdm.tqdm(f, desc="Loading Dataset")
                    ]
                    self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1, t2, is_next_label = self.random_sent(item)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t2 = t2_random + [self.vocab.eos_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "label": is_next_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        if self.on_memory:
            # print("ITEM", item)
            # print("ITEM 1", self.lines[item][0])
            # print("ITEM 2", self.lines[item][1])
            # print(self.lines[item][0], self.lines[item][1])
            return self.lines[item][0], self.lines[item][1]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t1, t2 = line[:-1].split("\t")
            return t1, t2

    def get_random_line(self):
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))][1]

        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return line[:-1].split("\t")[1]


class QuoraDataset(Dataset):
    def __init__(self, vocab, seq_len, corpus_lines=None):
        self.vocab = vocab
        self.seq_len = seq_len
        self.corpus_lines = corpus_lines

        # Load full dataset
        full_dataset = load_dataset("quora")["train"]

        # Limit the number of lines if specified
        if corpus_lines is not None:
            self.dataset = full_dataset.select(range(min(corpus_lines, len(full_dataset))))
        else:
            self.dataset = full_dataset


        # Check column names and an example from the training set
        # print("Columns:", self.dataset.column_names)
        # print("Example:", self.dataset[0])
    def __len__(self):
        return len(self.dataset)

    def tokenize(self, sentence):
        tokens = sentence.split()
        token_ids = [self.vocab.sos_index]
        for token in tokens:
            token_ids.append(self.vocab.stoi.get(token, self.vocab.unk_index))
        token_ids.append(self.vocab.eos_index)
        return token_ids

    def __getitem__(self, idx):
        q1 = self.dataset[idx]["questions"]["text"][0]
        q2 = self.dataset[idx]["questions"]["text"][1]
        label = self.dataset[idx]["is_duplicate"]

        q1_ids = self.tokenize(q1)
        q2_ids = self.tokenize(q2)

        input_ids = (q1_ids + q2_ids)[:self.seq_len]
        segment_ids = ([1] * len(q1_ids) + [2] * len(q2_ids))[:self.seq_len]

        # Padding
        padding_length = self.seq_len - len(input_ids)
        input_ids += [self.vocab.pad_index] * padding_length
        segment_ids += [self.vocab.pad_index] * padding_length

        return {
            "bert_input": torch.tensor(input_ids, dtype=torch.long),
            "segment_label": torch.tensor(segment_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long)
        }
    

import os
import re
from datasets import load_dataset
from nltk.tokenize import sent_tokenize

def build_corpus_if_missing(dataset_version, dataset_path, train_split):
    # Paths for corpus files
    train_path = os.path.join(dataset_path, f"wikipedia_train.tsv")
    test_path = os.path.join(dataset_path, f"wikipedia_test.tsv")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("✅ Corpus files already exist. Skipping creation.")
        return 

    print("📚 Building corpus from dataset...")

    # Load data from HuggingFace dataset
    text_full = load_dataset(
        "wikipedia",
        dataset_version,
        trust_remote_code=True,
    )["train"]
    
    n_articles = len(text_full)
    train_size = int(n_articles * float(train_split.strip('%')) / 100)
    text_train = text_full.select(range(train_size))
    text_test = text_full.select(range(train_size, n_articles))


    def process_and_write(dataset, output_file):
        with open(output_file, "w", encoding="utf-8") as out_f:
            for article in dataset:
                text = article["text"]
                paragraphs = re.split(r"\n\s*\n", text)
                sentences = []
                for para in paragraphs:
                    para = para.strip()
                    if para.istitle() or len(para.split()) < 3 or not re.search(r"[a-zA-Z]{3,}", para):
                        continue
                    filtered = [
                        s.strip()
                        for s in sent_tokenize(para)
                        if len(s.strip().split()) >= 4
                        and not s.strip().istitle()
                        and re.search(r"[a-zA-Z]{4,}", s)
                    ]
                    sentences.extend(filtered)
                if len(sentences) < 2:
                    continue
                for i in range(0, len(sentences) - 1, 2):
                    s1, s2 = sentences[i].strip(), sentences[i + 1].strip()
                    if not s1 or not s2 or len(s1.split()) < 2 or len(s2.split()) < 2:
                        continue
                    if re.match(r"^\d{4}\b", s1) or re.match(r"^\d{4}\b", s2):
                        continue
                    if any(s.startswith(x) for s in (s1, s2) for x in ["Order", "Series"]):
                        continue
                    if "\n" in s1 or "\n" in s2:
                        continue
                    out_f.write(f"{s1}\t{s2}\n")

    process_and_write(text_train, train_path)
    process_and_write(text_test, test_path)

    print(f"✅ Corpus written to: {train_path} and {test_path}")
    return 