# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: query_rewrite.py
# @time: 2023/12/28 12:55
import os
import time
import random
import json
import requests
import numpy as np
from tqdm import tqdm
from retry import retry

from llama_index.llms import OpenAI, ChatMessage
from llama_index import PromptTemplate

llm = OpenAI(model="gpt-3.5-turbo")

query_gen_prompt_str = (
    "为下面的问题提供{num_queries}个查询改写，使之能更好地适应搜索引擎。每行一个改写结果，以-开头，当涉及到缩写时，要提供全称。\n"
    "问题：{query} \n"
    "答案："
)
query_gen_prompt = PromptTemplate(query_gen_prompt_str)


def generate_queries(llm, query_str: str, num_queries: int = 3):
    fmt_prompt = query_gen_prompt.format(
        num_queries=num_queries, query=query_str
    )
    response = llm.complete(fmt_prompt)
    queries = [_.replace("- ", "").strip() for _ in response.text.split("\n") if _.strip()]
    return queries[:num_queries]


@retry(exceptions=Exception, tries=3, max_delay=20)
def get_openai_embedding(req_text: str) -> list[float]:
    time.sleep(random.random() / 2)
    url = "https://api.openai.com/v1/embeddings"
    headers = {'Content-Type': 'application/json', "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
    payload = json.dumps({"model": "text-embedding-ada-002", "input": req_text})
    new_req = requests.request("POST", url, headers=headers, data=payload)
    return new_req.json()['data'][0]['embedding']


if __name__ == '__main__':
    # query = "日美半导体协议要求美国芯片在日本市场份额是多少？"
    # query = "半导体制造设备市场美、日、荷各占多少份额？"
    # query = "尼康和佳能的光刻机在哪个市场占优势？"
    # print(generate_queries(llm, query, num_queries=2))
    num_queries = 2
    with open("../data/doc_qa_test.json", "r", encoding="utf-8") as f:
        content = json.loads(f.read())
    queries = list(content["queries"].values())
    query_num = len(queries)

    rewrite_dict = {}
    embedding_data = np.empty(shape=[query_num * num_queries, 1536])
    for i in tqdm(range(query_num), desc="generate embedding"):
        query = queries[i]
        rewrite_queries = generate_queries(llm, query, num_queries=num_queries)
        rewrite_dict[query] = rewrite_queries
        print(rewrite_queries)
        for j, rewrite_query in enumerate(rewrite_queries):
            embedding_data[2 * i + j] = get_openai_embedding(rewrite_query)

    with open('../data/query_rewrite.json', "w") as f:
        f.write(json.dumps(rewrite_dict, ensure_ascii=False, indent=4))
    np.save("../data/query_rewrite_openai_embedding.npy", embedding_data)

