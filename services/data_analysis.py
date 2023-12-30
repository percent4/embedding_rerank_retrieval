# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: data_analysis.py
# @time: 2023/12/30 11:52
import pandas as pd
from faiss import IndexFlatIP
from llama_index.evaluation.retrieval.metrics import HitRate, MRR

from custom_retriever.bm25_retriever import CustomBM25Retriever
from custom_retriever.vector_store_retriever import VectorSearchRetriever
from custom_retriever.ensemble_retriever import EnsembleRetriever
from custom_retriever.ensemble_rerank_retriever import EnsembleRerankRetriever
from preprocess.get_text_id_mapping import queries, query_relevant_docs, node_id_text_mapping


def get_metric(search_query, search_result):
    hit_rate = HitRate().compute(query=search_query,
                                 expected_ids=query_relevant_docs[search_query],
                                 retrieved_ids=[_.id_ for _ in search_result])
    mrr = MRR().compute(query=search_query,
                        expected_ids=query_relevant_docs[search_query],
                        retrieved_ids=[_.id_ for _ in search_result])
    return [hit_rate.score, mrr.score]


top_k = 3
faiss_index = IndexFlatIP(1536)
data_columns = []
for i, query in enumerate(queries, start=1):
    print(i, query)
    expect_text = node_id_text_mapping[query_relevant_docs[query][0]]
    record = [query]
    # bm25
    bm25_retriever = CustomBM25Retriever(top_k=top_k)
    bm25_search_result = bm25_retriever.retrieve(query)
    bm25_metric = get_metric(query, bm25_search_result)
    record.extend(bm25_metric)
    # embedding search
    vector_search_retriever = VectorSearchRetriever(top_k=top_k, faiss_index=faiss_index)
    embedding_search_result = vector_search_retriever.retrieve(str_or_query_bundle=query)
    embedding_metric = get_metric(query, embedding_search_result)
    faiss_index.reset()
    record.extend(embedding_metric)
    # ensemble search
    ensemble_retriever = EnsembleRetriever(top_k=top_k, faiss_index=faiss_index, weights=[0.5, 0.5])
    ensemble_search_result = ensemble_retriever.retrieve(str_or_query_bundle=query)
    ensemble_metric = get_metric(query, ensemble_search_result)
    faiss_index.reset()
    record.extend(ensemble_metric)
    # ensemble rerank search
    ensemble_retriever = EnsembleRerankRetriever(top_k=top_k, faiss_index=faiss_index)
    ensemble_rerank_search_result = ensemble_retriever.retrieve(str_or_query_bundle=query)
    ensemble_rerank_metric = get_metric(query, ensemble_rerank_search_result)
    faiss_index.reset()
    record.extend(ensemble_rerank_metric)
    record.append(expect_text)
    data_columns.append(record)


df = pd.DataFrame(data_columns,
                  columns=["query", "bm25_hit_rate", "bm25_mrr", "embedding_hit_rate", "embedding_mrr",
                           "ensemble_hit_rate", "ensemble_mrr", "ensemble_rerank_hit_rate",
                           "ensemble_rerank_mrr", "expect text"])
df.to_excel("search_result_analysis.xlsx", index=False)
