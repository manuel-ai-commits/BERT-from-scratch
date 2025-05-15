import os
import re

from datasets import load_dataset
from nltk.tokenize import sent_tokenize

def preprocess_text(opt, dataset = "wikipedia"):
    # Check if we are using HuggingFace dataset or custom path
    
    if dataset == "wikipedia":
        # Load Wikipedia dataset and split into train and test
        text_full = load_dataset(opt.input.dataset_name, opt.input.dataset_version, split=f"train[:{opt.input.n_articles}]", trust_remote_code=True)
        train_size = int(opt.input.n_articles * float(opt.input.train_split.strip('%')) / 100)
        text_train = text_full.select(range(train_size))
        text_test = text_full.select(range(train_size, opt.input.n_articles))
        

        
        # Initialize output files for train and test
        output_train_file = os.path.join(opt.input.dataset_path, "bert_corpus_train.tsv")
        output_test_file = os.path.join(opt.input.dataset_path, "bert_corpus_test.tsv")
        

        def process_and_write(dataset, output_file):
            with open(output_file, "w", encoding="utf-8") as out_f:
                for article in dataset:
                    text = article["text"]
                    paragraphs = re.split(r"\n\s*\n", text)
                    sentences = []
                    for para in paragraphs:
                        para = para.strip()
                        if (
                            para.istitle()
                            or len(para.split()) < 3
                            or not re.search(r"[a-zA-Z]{3,}", para)
                        ):
                            continue
                        filtered = [
                            s.strip() for s in sent_tokenize(para)
                            if len(s.strip().split()) >= 4
                            and not s.strip().istitle()
                            and re.search(r"[a-zA-Z]{4,}", s)
                        ]
                        sentences.extend(filtered)

                    if len(sentences) < 2:
                        print(f"Skipping article due to insufficient sentences: {article['title']}")
                        continue

                    # Generate sentence pairs (NSP task) and write to output file
                    for i in range(0, len(sentences) - 1, 2):
                        s1 = sentences[i].strip()
                        s2 = sentences[i + 1].strip()

                        if not s1.strip() or not s2.strip() or len(s1.split()) < 2 or len(s2.split()) < 2:
                            continue

                        if re.match(r"^\d{4}\b", s1) or re.match(r"^\d{4}\b", s2):
                            continue
                        if s1.startswith("Order") or s2.startswith("Order") or s1.startswith("Series") or s2.startswith("Series"):
                            continue
                        if "\n" in s1 or "\n" in s2:
                            continue

                        out_f.write(f"{s1}\t{s2}\n")
        process_and_write(text_train, output_train_file)
        process_and_write(text_test, output_test_file)

        print(f"✅ Finished writing sentence pairs for s to {output_train_file} and {output_test_file}")
    elif dataset == "quora":
        text_full = load_dataset("quora", trust_remote_code=True)["train"]
        print(text_full)
        train_size = int(opt.input.n_articles * float(opt.input.train_split.strip('%')) / 100)
        text_train = text_full.select(range(train_size))
        text_test = text_full.select(range(train_size, opt.input.n_articles))
        print(text_train)
        # Initialize output files for train and test
        output_train_file = os.path.join(opt.input.dataset_path, "bert_corpus_train.tsv")
        output_test_file = os.path.join(opt.input.dataset_path, "bert_corpus_test.tsv")
        
        def process_and_write_personal(dataset, output_file):
            """
            Converts dataset entries like:
            {"question1": "text1", "question2": "text2"}
            into lines formatted as: "text1 \t text2"
            """
            with open(output_file, "w", encoding="utf-8") as out_f:
                for example in dataset:
                    print(example)
                    q1 = example.get("question1", "").strip()
                    q2 = example.get("question2", "").strip()
                    print(q1, q2)
                    if q1 and q2:
                        out_f.write(f"{q1}\t{q2}\n")

        
        process_and_write_personal(text_train, output_train_file)
        process_and_write_personal(text_test, output_test_file)
        
        