# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: build_embedding_cache.py
# @time: 2023/12/26 12:57
import os
import time
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
    def __get_openai_embedding(req_text: str):
        time.sleep(random.random() / 2)
        url = "https://api.openai.com/v1/embeddings"
        headers = {'Content-Type': 'application/json', "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
        payload = json.dumps({"model": "text-embedding-ada-002", "input": req_text})
        new_req = requests.request("POST", url, headers=headers, data=payload)
        return new_req.json()['data'][0]['embedding']

    def build_with_context(self, context_type: str):
        with open("../data/doc_qa_test.json", "r", encoding="utf-8") as f:
            content = json.loads(f.read())
        queries = list(content[context_type].values())
        query_num = len(queries)
        embedding_data = np.empty(shape=[query_num, 1536])
        for i in tqdm(range(query_num), desc="generate embedding"):
            embedding_data[i] = self.__get_openai_embedding(queries[i])
        np.save(f"../data/{context_type}_openai_embedding.npy", embedding_data)

    def build(self):
        self.build_with_context("queries")
        self.build_with_context("corpus")

    @staticmethod
    def load():
        queries_embedding_data = np.load("../data/queries_openai_embedding.npy")
        corpus_embedding_data = np.load("../data/corpus_openai_embedding.npy")
        query_embedding_dict = {}
        with open("../data/doc_qa_test.json", "r", encoding="utf-8") as f:
            content = json.loads(f.read())
        queries = list(content["queries"].values())
        corpus = list(content["corpus"].values())
        for i in range(len(queries)):
            query_embedding_dict[queries[i]] = queries_embedding_data[i].tolist()
        return query_embedding_dict, corpus_embedding_data, corpus


if __name__ == '__main__':
    EmbeddingCache().build()
