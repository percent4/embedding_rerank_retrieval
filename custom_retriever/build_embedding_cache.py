# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: build_embedding_cache.py
# @time: 2023/12/26 12:57
import os
import time
import math
import json
import random
import requests
import numpy as np
from retry import retry
from tqdm import tqdm


class EmbeddingCache(object):
    def __init__(self):
        pass

    @staticmethod
    @retry(exceptions=Exception, tries=3, max_delay=20)
    def get_openai_embedding(req_text: str):
        time.sleep(random.random() / 2)
        url = "https://api.openai.com/v1/embeddings"
        headers = {'Content-Type': 'application/json', "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
        payload = json.dumps({"model": "text-embedding-ada-002", "input": req_text})
        new_req = requests.request("POST", url, headers=headers, data=payload)
        return new_req.json()['data'][0]['embedding']

    @staticmethod
    @retry(exceptions=Exception, tries=3, max_delay=20)
    def get_bge_embedding(req_text: str):
        url = "http://localhost:50073/embedding"
        headers = {'Content-Type': 'application/json'}
        payload = json.dumps({"text": req_text})
        new_req = requests.request("POST", url, headers=headers, data=payload)
        return new_req.json()['embedding']

    @staticmethod
    @retry(exceptions=Exception, tries=3, max_delay=20)
    def get_jina_embedding(req_text: str):
        time.sleep(random.random() / 2)
        url = 'https://api.jina.ai/v1/embeddings'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.getenv("JINA_API_KEY")}'
        }
        data = {
            'input': [req_text],
            'model': 'jina-embeddings-v2-base-zh'
        }
        response = requests.post(url, headers=headers, json=data)
        embedding = response.json()["data"][0]["embedding"]
        embedding_norm = math.sqrt(sum([i**2 for i in embedding]))
        return [i/embedding_norm for i in embedding]

    def build_with_context(self, context_type: str):
        with open("../data/doc_qa_test.json", "r", encoding="utf-8") as f:
            content = json.loads(f.read())
        queries = list(content[context_type].values())
        query_num = len(queries)
        embedding_data = np.empty(shape=[query_num, 768])
        for i in tqdm(range(query_num), desc="generate embedding"):
            embedding_data[i] = self.get_bge_embedding(queries[i])
        np.save(f"../data/{context_type}_bce_embedding.npy", embedding_data)

    def build(self):
        self.build_with_context("queries")
        self.build_with_context("corpus")

    @staticmethod
    def load(query_write=False):
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        queries_embedding_data = np.load(os.path.join(current_dir, "data/queries_jina_base_zh_embedding.npy"))
        corpus_embedding_data = np.load(os.path.join(current_dir, "data/corpus_jina_base_zh_late_chunking_embedding.npy"))
        query_embedding_dict = {}
        with open(os.path.join(current_dir, "data/doc_qa_test.json"), "r", encoding="utf-8") as f:
            content = json.loads(f.read())
        queries = list(content["queries"].values())
        corpus = list(content["corpus"].values())
        for i in range(len(queries)):
            query_embedding_dict[queries[i]] = queries_embedding_data[i].tolist()
        if query_write:
            rewrite_queries_embedding_data = np.load(os.path.join(current_dir, "data/query_rewrite_openai_embedding.npy"))
            with open("../data/query_rewrite.json", "r", encoding="utf-8") as f:
                rewrite_content = json.loads(f.read())

            rewrite_queries_list = []
            for original_query, rewrite_queries in rewrite_content.items():
                rewrite_queries_list.extend(rewrite_queries)
            for i in range(len(rewrite_queries_list)):
                query_embedding_dict[rewrite_queries_list[i]] = rewrite_queries_embedding_data[i].tolist()
        return query_embedding_dict, corpus_embedding_data, corpus


if __name__ == '__main__':
    EmbeddingCache().build()
