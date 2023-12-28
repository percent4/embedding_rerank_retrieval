# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: cohere_rerank.py
# @time: 2023/12/26 19:21
import os
import time
import random
import cohere
from typing import List, Tuple
from retry import retry

cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))


@retry(exceptions=Exception, tries=5, max_delay=60)
def cohere_rerank_result(query: str, docs: List[str], top_n) -> List[Tuple]:
    time.sleep(1 + 2 * random.random())
    results = cohere_client.rerank(model="rerank-multilingual-v2.0",
                                   query=query,
                                   documents=docs,
                                   top_n=top_n)
    return [(hit.document['text'], hit.relevance_score) for hit in results]
