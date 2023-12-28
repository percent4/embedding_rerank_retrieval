# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: data_transfer.py
# @time: 2023/12/25 17:51
import pandas as pd
from llama_index.llms import OpenAI
from llama_index.schema import TextNode
from llama_index.evaluation import generate_question_context_pairs
import random
random.seed(42)

llm = OpenAI(model="gpt-4", max_retries=5)

# Prompt to generate questions
qa_generate_prompt_tmpl = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a university professor. Your task is to set {num_questions_per_chunk} questions for the upcoming Chinese quiz.
Questions throughout the test should be diverse. Questions should not contain options or start with Q1/Q2.
Questions must be written in Chinese. The expression must be concise and clear. 
It should not exceed 15 Chinese characters. Words such as "这", "那", "根据", "依据" and other punctuation marks 
should not be used. Abbreviations may be used for titles and professional terms.
"""

nodes = []
data_df = pd.read_csv("../data/doc_qa_dataset.csv", encoding="utf-8")
for i, row in data_df.iterrows():
    if len(row["content"]) > 80 and i > 96:
        node = TextNode(text=row["content"])
        node.id_ = f"node_{i + 1}"
        nodes.append(node)


doc_qa_dataset = generate_question_context_pairs(
    nodes, llm=llm, num_questions_per_chunk=1, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl
)

doc_qa_dataset.save_json("../data/doc_qa_dataset.json")
