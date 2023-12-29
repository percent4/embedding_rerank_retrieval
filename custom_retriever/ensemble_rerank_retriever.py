# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: ensemble_rerank_retriever.py
# @time: 2023/12/26 19:18
from typing import List

from llama_index.schema import TextNode
from llama_index.schema import NodeWithScore
from llama_index.retrievers import BaseRetriever
from llama_index.indices.query.schema import QueryBundle, QueryType

from preprocess.get_text_id_mapping import text_node_id_mapping
from custom_retriever.bm25_retriever import CustomBM25Retriever
from custom_retriever.vector_store_retriever import VectorSearchRetriever
from utils.rerank import bge_rerank_result


class EnsembleRerankRetriever(BaseRetriever):
    def __init__(self, top_k, faiss_index):
        super().__init__()
        self.faiss_index = faiss_index
        self.top_k = top_k
        self.embedding_retriever = VectorSearchRetriever(top_k=self.top_k, faiss_index=faiss_index)

    def _retrieve(self, query: QueryType) -> List[NodeWithScore]:
        if isinstance(query, str):
            query = QueryBundle(query)
        else:
            query = query
        # print(query.query_str)
        bm25_search_nodes = CustomBM25Retriever(top_k=self.top_k).retrieve(query)
        embedding_search_nodes = self.embedding_retriever.retrieve(query)
        bm25_docs = [node.text for node in bm25_search_nodes]
        embedding_docs = [node.text for node in embedding_search_nodes]
        # remove duplicate document
        all_documents = set()
        for doc_list in [bm25_docs, embedding_docs]:
            for doc in doc_list:
                all_documents.add(doc)
        doc_lists = list(all_documents)
        rerank_doc_lists = bge_rerank_result(query.query_str, doc_lists, top_n=self.top_k)
        result = []
        for sorted_doc in rerank_doc_lists:
            text, score = sorted_doc
            node_with_score = NodeWithScore(node=TextNode(text=text,
                                                          id_=text_node_id_mapping[text]),
                                            score=score)
            result.append(node_with_score)

        return result


if __name__ == '__main__':
    from faiss import IndexFlatIP

    faiss_index = IndexFlatIP(1536)
    ensemble_retriever = EnsembleRerankRetriever(top_k=2, faiss_index=faiss_index)
    t_result = ensemble_retriever.retrieve(str_or_query_bundle="索尼1953年引入的技术专利是什么？")
    print(t_result)
    faiss_index.reset()
