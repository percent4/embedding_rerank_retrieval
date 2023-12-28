# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: add_corpus.py
# @time: 2023/12/25 19:46
import json
import pandas as pd

with open('../data/doc_qa_dataset.json', 'r', encoding="utf-8") as f:
    content = json.loads(f.read())

corpus = content['corpus']
texts = [text for node_id, text in content['corpus'].items()]

data_df = pd.read_csv("../data/doc_qa_dataset.csv", encoding="utf-8")
for i, row in data_df.iterrows():
    node_id = f"node_{i + 1}"
    if node_id not in corpus:
        corpus[f"node_{i + 1}"] = row["content"]

content["corpus"] = corpus

with open("../data/doc_qa_test.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(content, ensure_ascii=False, indent=4))
