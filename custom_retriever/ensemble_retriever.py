# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: ensemble_retriever.py
# @time: 2023/12/26 18:50
from typing import List
from operator import itemgetter

from llama_index.schema import TextNode
from llama_index.schema import NodeWithScore
from llama_index.retrievers import BaseRetriever
from llama_index.indices.query.schema import QueryType

from preprocess.get_text_id_mapping import text_node_id_mapping
from custom_retriever.bm25_retriever import CustomBM25Retriever
from custom_retriever.vector_store_retriever import VectorSearchRetriever


class EnsembleRetriever(BaseRetriever):
    def __init__(self, top_k, faiss_index, weights):
        super().__init__()
        self.weights = weights
        self.c: int = 60
        self.faiss_index = faiss_index
        self.top_k = top_k
        self.embedding_retriever = VectorSearchRetriever(top_k=self.top_k, faiss_index=faiss_index)

    def _retrieve(self, query: QueryType) -> List[NodeWithScore]:
        bm25_search_nodes = CustomBM25Retriever(top_k=self.top_k).retrieve(query)
        embedding_search_nodes = self.embedding_retriever.retrieve(query)
        bm25_docs = [node.text for node in bm25_search_nodes]
        embedding_docs = [node.text for node in embedding_search_nodes]
        doc_lists = [bm25_docs, embedding_docs]

        # Create a union of all unique documents in the input doc_lists
        all_documents = set()
        for doc_list in doc_lists:
            for doc in doc_list:
                all_documents.add(doc)

        # Initialize the RRF score dictionary for each document
        rrf_score_dic = {doc: 0.0 for doc in all_documents}

        # Calculate RRF scores for each document
        for doc_list, weight in zip(doc_lists, self.weights):
            for rank, doc in enumerate(doc_list, start=1):
                rrf_score = weight * (1 / (rank + self.c))
                rrf_score_dic[doc] += rrf_score

        # Sort documents by their RRF scores in descending order
        sorted_documents = sorted(rrf_score_dic.items(), key=itemgetter(1), reverse=True)
        result = []
        for sorted_doc in sorted_documents[:self.top_k]:
            text, score = sorted_doc
            node_with_score = NodeWithScore(node=TextNode(text=text,
                                                          id_=text_node_id_mapping[text]),
                                            score=score)
            result.append(node_with_score)

        return result


if __name__ == '__main__':
    from faiss import IndexFlatIP

    faiss_index = IndexFlatIP(1536)
    ensemble_retriever = EnsembleRetriever(top_k=2, faiss_index=faiss_index, weights=[0.5, 0.5])
    t_result = ensemble_retriever.retrieve(str_or_query_bundle="索尼1953年引入的技术专利是什么？")
    print(t_result)
    faiss_index.reset()
