# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: get_text_id_mapping.py
# @time: 2023/12/25 20:18
import json

with open('../data/doc_qa_test.json', 'r', encoding="utf-8") as f:
    content = json.loads(f.read())

node_id_text_mapping = content['corpus']
text_node_id_mapping = {v: k for k, v in node_id_text_mapping.items()}
