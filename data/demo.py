# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai 
# @contact: lianmingjie@shanda.com
# @file: demo.py
# @time: 2023/12/26 11:15
import json
with open("doc_qa_test.json", "r") as f:
    content = json.loads(f.read())

new_content = {}
n = 5
for k, v in content.items():
    if k in ["queries", "relevant_docs"]:
        new_content[k] = {}
        for key in list(v.keys())[:n]:
            new_content[k][key] = v[key]
    else:
        new_content[k] = v

with open("doc_qa_test_demo.json", "w") as f:
    f.write(json.dumps(new_content, indent=4, ensure_ascii=False))
