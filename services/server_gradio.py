# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: server_gradio.py
# @time: 2023/12/29 22:25
from random import shuffle
import gradio as gr
import pandas as pd

from faiss import IndexFlatIP
from llama_index.evaluation.retrieval.metrics import HitRate, MRR

from custom_retriever.bm25_retriever import CustomBM25Retriever
from custom_retriever.vector_store_retriever import VectorSearchRetriever
from custom_retriever.ensemble_retriever import EnsembleRetriever
from custom_retriever.ensemble_rerank_retriever import EnsembleRerankRetriever
from preprocess.get_text_id_mapping import queries, query_relevant_docs
from preprocess.query_rewrite import generate_queries, llm

retrieve_methods = ["bm25", "embedding", "ensemble", "ensemble + bge-rerank-large", "query_rewrite + ensemble"]


def get_metric(search_query, search_result):
    hit_rate = HitRate().compute(query=search_query,
                                 expected_ids=query_relevant_docs[search_query],
                                 retrieved_ids=[_.id_ for _ in search_result])
    mrr = MRR().compute(query=search_query,
                        expected_ids=query_relevant_docs[search_query],
                        retrieved_ids=[_.id_ for _ in search_result])
    return [hit_rate.score, mrr.score]


def get_retrieve_result(retriever_list, retrieve_top_k, retrieve_query):
    columns = {"metric_&_top_k": ["Hit Rate", "MRR"] + [f"top_{k + 1}" for k in range(retrieve_top_k)]}
    if "bm25" in retriever_list:
        bm25_retriever = CustomBM25Retriever(top_k=retrieve_top_k)
        search_result = bm25_retriever.retrieve(retrieve_query)
        columns["bm25"] = []
        columns["bm25"].extend(get_metric(retrieve_query, search_result))
        for i, node in enumerate(search_result, start=1):
            columns["bm25"].append(node.text)
    if "embedding" in retriever_list:
        faiss_index = IndexFlatIP(1536)
        vector_search_retriever = VectorSearchRetriever(top_k=retrieve_top_k, faiss_index=faiss_index)
        search_result = vector_search_retriever.retrieve(str_or_query_bundle=retrieve_query)
        columns["embedding"] = []
        columns["embedding"].extend(get_metric(retrieve_query, search_result))
        for i in range(retrieve_top_k):
            columns["embedding"].append(search_result[i].text)
        faiss_index.reset()
    if "ensemble" in retriever_list:
        faiss_index = IndexFlatIP(1536)
        ensemble_retriever = EnsembleRetriever(top_k=retrieve_top_k, faiss_index=faiss_index, weights=[0.5, 0.5])
        search_result = ensemble_retriever.retrieve(str_or_query_bundle=retrieve_query)
        columns["ensemble"] = []
        columns["ensemble"].extend(get_metric(retrieve_query, search_result))
        for i in range(retrieve_top_k):
            columns["ensemble"].append(search_result[i].text)
        faiss_index.reset()
    if "ensemble + bge-rerank-large" in retriever_list:
        faiss_index = IndexFlatIP(1536)
        ensemble_retriever = EnsembleRerankRetriever(top_k=retrieve_top_k, faiss_index=faiss_index)
        search_result = ensemble_retriever.retrieve(str_or_query_bundle=retrieve_query)
        columns["ensemble + bge-rerank-large"] = []
        columns["ensemble + bge-rerank-large"].extend(get_metric(retrieve_query, search_result))
        for i in range(retrieve_top_k):
            columns["ensemble + bge-rerank-large"].append(search_result[i].text)
        faiss_index.reset()
    if "query_rewrite + ensemble" in retriever_list:
        queries = generate_queries(llm, retrieve_query, num_queries=1)
        print(f"original query: {retrieve_query}\n"
              f"rewrite query: {queries}")
        faiss_index = IndexFlatIP(1536)
        ensemble_retriever = EnsembleRetriever(top_k=retrieve_top_k, faiss_index=faiss_index, weights=[0.5, 0.5])
        search_result = ensemble_retriever.retrieve(str_or_query_bundle=queries[0])
        columns["query_rewrite + ensemble"] = []
        columns["query_rewrite + ensemble"].extend(get_metric(retrieve_query, search_result))
        for i in range(retrieve_top_k):
            columns["query_rewrite + ensemble"].append(search_result[i].text)
        faiss_index.reset()
    retrieve_df = pd.DataFrame(columns)
    return retrieve_df


with gr.Blocks() as demo:
    retrievers = gr.CheckboxGroup(choices=retrieve_methods,
                                  type="value",
                                  label="Retrieve Methods")
    top_k = gr.Dropdown(list(range(1, 6)), label="top_k", value=3)
    shuffle(queries)
    query = gr.Dropdown(queries, label="query", value=queries[0])
    # 设置输出组件
    result_table = gr.DataFrame(label='Result', wrap=True)
    theme = gr.themes.Base()
    # 设置按钮
    submit = gr.Button("Submit")
    submit.click(fn=get_retrieve_result, inputs=[retrievers, top_k, query], outputs=result_table)


demo.launch()
