# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: evaluation_exp.py
# @time: 2023/12/25 20:01
import asyncio
import time

import pandas as pd
from datetime import datetime
from faiss import IndexFlatIP
from llama_index.evaluation import RetrieverEvaluator
from llama_index.finetuning.embeddings.common import EmbeddingQAFinetuneDataset

from custom_retriever.bm25_retriever import CustomBM25Retriever
from custom_retriever.vector_store_retriever import VectorSearchRetriever
from custom_retriever.ensemble_retriever import EnsembleRetriever
from custom_retriever.ensemble_rerank_retriever import EnsembleRerankRetriever
from custom_retriever.query_rewrite_ensemble_retriever import QueryRewriteEnsembleRetriever


# Display results from evaluate.
def display_results(name_list, eval_results_list):
    pd.set_option('display.precision', 4)
    columns = {"retrievers": [], "hit_rate": [], "mrr": []}
    for name, eval_results in zip(name_list, eval_results_list):
        metric_dicts = []
        for eval_result in eval_results:
            metric_dict = eval_result.metric_vals_dict
            metric_dicts.append(metric_dict)

        full_df = pd.DataFrame(metric_dicts)

        hit_rate = full_df["hit_rate"].mean()
        mrr = full_df["mrr"].mean()

        columns["retrievers"].append(name)
        columns["hit_rate"].append(hit_rate)
        columns["mrr"].append(mrr)

    metric_df = pd.DataFrame(columns)

    return metric_df


doc_qa_dataset = EmbeddingQAFinetuneDataset.from_json("../data/doc_qa_test.json")
metrics = ["mrr", "hit_rate"]
# bm25 retrieve
# evaluation_name_list = []
# evaluation_result_list = []
# cost_time_list = []
# for top_k in [1, 2, 3, 4, 5]:
#     start_time = time.time()
#     bm25_retriever = CustomBM25Retriever(top_k=top_k)
#     bm25_retriever_evaluator = RetrieverEvaluator.from_metric_names(metrics, retriever=bm25_retriever)
#     bm25_eval_results = asyncio.run(bm25_retriever_evaluator.aevaluate_dataset(doc_qa_dataset))
#     evaluation_name_list.append(f"bm25_top_{top_k}_eval")
#     evaluation_result_list.append(bm25_eval_results)
#     cost_time_list.append((time.time() - start_time) * 1000)
#
# print("done for bm25 evaluation!")
# df = display_results(evaluation_name_list, evaluation_result_list)
# df['cost_time'] = cost_time_list
# print(df.head())
# df.to_csv(f"evaluation_bm25_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.csv", encoding="utf-8", index=False)

# embedding retrieve
evaluation_name_list = []
evaluation_result_list = []
cost_time_list = []

for top_k in [1, 2, 3, 4, 5]:
    start_time = time.time()
    faiss_index = IndexFlatIP(768)
    embedding_retriever = VectorSearchRetriever(top_k=top_k, faiss_index=faiss_index)
    embedding_retriever_evaluator = RetrieverEvaluator.from_metric_names(metrics, retriever=embedding_retriever)
    embedding_eval_results = asyncio.run(embedding_retriever_evaluator.aevaluate_dataset(doc_qa_dataset))
    evaluation_name_list.append(f"embedding_top_{top_k}_eval")
    evaluation_result_list.append(embedding_eval_results)
    faiss_index.reset()
    cost_time_list.append((time.time() - start_time) * 1000)

print("done for embedding evaluation!")
df = display_results(evaluation_name_list, evaluation_result_list)
df['cost_time'] = cost_time_list
print(df.head())
df.to_csv(f"evaluation_bge_base_sft_embedding_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.csv", encoding="utf-8", index=False)

# ensemble retrieve
# evaluation_name_list = []
# evaluation_result_list = []
# cost_time_list = []
#
# for top_k in [1, 2, 3, 4, 5]:
#     start_time = time.time()
#     faiss_index = IndexFlatIP(1536)
#     ensemble_retriever = EnsembleRetriever(top_k=top_k, faiss_index=faiss_index, weights=[0.5, 0.5])
#     ensemble_retriever_evaluator = RetrieverEvaluator.from_metric_names(metrics, retriever=ensemble_retriever)
#     ensemble_eval_results = asyncio.run(ensemble_retriever_evaluator.aevaluate_dataset(doc_qa_dataset))
#     evaluation_name_list.append(f"ensemble_top_{top_k}_eval")
#     evaluation_result_list.append(ensemble_eval_results)
#     faiss_index.reset()
#     cost_time_list.append((time.time() - start_time) * 1000)
#
# print("done for ensemble evaluation!")
# df = display_results(evaluation_name_list, evaluation_result_list)
# df['cost_time'] = cost_time_list
# print(df.head())
# df.to_csv(f"evaluation_ensemble_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.csv", encoding="utf-8", index=False)

# ensemble rerank retrieve
# evaluation_name_list = []
# evaluation_result_list = []
# cost_time_list = []
#
# for top_k in [1, 2, 3, 4, 5]:
#     start_time = time.time()
#     faiss_index = IndexFlatIP(1536)
#     ensemble_rerank_retriever = EnsembleRerankRetriever(top_k=top_k, faiss_index=faiss_index)
#     ensemble_rerank_retriever_evaluator = RetrieverEvaluator.from_metric_names(metrics,
#                                                                                retriever=ensemble_rerank_retriever)
#     ensemble_rerank_eval_results = asyncio.run(ensemble_rerank_retriever_evaluator.aevaluate_dataset(doc_qa_dataset,
#                                                                                                      show_progress=True))
#     evaluation_name_list.append(f"ensemble_rerank_top_{top_k}_eval")
#     evaluation_result_list.append(ensemble_rerank_eval_results)
#     faiss_index.reset()
#     cost_time_list.append((time.time() - start_time) * 1000)
#
#     print("done for ensemble_rerank evaluation!")
#     df = display_results(evaluation_name_list, evaluation_result_list)
#     df['cost_time'] = cost_time_list
#     print(df.head())
#     df.to_csv(f"evaluation_ensemble-rerank-bge-base_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.csv", encoding="utf-8", index=False)

# query rewrite ensemble retrieve
# evaluation_name_list = []
# evaluation_result_list = []
# cost_time_list = []
#
# for top_k in [1, 2, 3, 4, 5]:
#     start_time = time.time()
#     faiss_index = IndexFlatIP(1536)
#     query_rewrite_ensemble_retriever = QueryRewriteEnsembleRetriever(top_k=top_k, faiss_index=faiss_index)
#     query_rewrite_ensemble_retriever_evaluator = RetrieverEvaluator.\
#         from_metric_names(metrics, retriever=query_rewrite_ensemble_retriever)
#     query_rewrite_ensemble_eval_results = asyncio.run(query_rewrite_ensemble_retriever_evaluator.aevaluate_dataset(doc_qa_dataset))
#     evaluation_name_list.append(f"query-rewrite-ensemble_top_{top_k}_eval")
#     evaluation_result_list.append(query_rewrite_ensemble_eval_results)
#     faiss_index.reset()
#     cost_time_list.append((time.time() - start_time) * 1000)
#
# print("done for query_rewrite ensemble evaluation!")
# df = display_results(evaluation_name_list, evaluation_result_list)
# df['cost_time'] = cost_time_list
# print(df.head())
# df.to_csv(f"evaluation_query-rewrite-ensemble_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.csv", encoding="utf-8", index=False)

