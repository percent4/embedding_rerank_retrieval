# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: rerank.py
# @time: 2023/12/26 19:21
import os
import time
import requests
import json
from random import randint
import cohere
from typing import List, Tuple
from retry import retry

# cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
#
#
# @retry(exceptions=Exception, tries=5, max_delay=60)
# def cohere_rerank_result(query: str, docs: List[str], top_n) -> List[Tuple]:
#     time.sleep(randint(1, 10))
#     results = cohere_client.rerank(model="rerank-multilingual-v2.0",
#                                    query=query,
#                                    documents=docs,
#                                    top_n=top_n)
#     return [(hit.document['text'], hit.relevance_score) for hit in results]


@retry(exceptions=Exception, tries=5, max_delay=60)
def bge_rerank_result(query: str, docs: List[str], top_n) -> List[Tuple]:
    url = "http://localhost:50072/bge_base_rerank"
    payload = json.dumps({
        "query": query,
        "passages": docs,
        "top_k": top_n
    })
    headers = {'Content-Type': 'application/json'}

    response = requests.request("POST", url, headers=headers, data=payload)
    return [(passage, score) for passage, score in response.json().items()]
