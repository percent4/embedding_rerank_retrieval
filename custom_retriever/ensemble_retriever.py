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
    query = "日本半导体发展史的三个时期是什么？日本半导体发展史可以分为以下三个时期：1. 初期发展（1950年代至1970年代）：在这一时期，日本半导体行业主要依赖于进口技术和设备。日本政府积极推动半导体产业的发展，设立了研究机构和实验室，并提供财政支持。日本企业开始生产晶体管和集成电路，逐渐取得了技术突破和市场份额的增长。2. 高速增长（1980年代至1990年代）：在这一时期，日本半导体行业迅速崛起，成为全球"
    query = "美日半导体协议是由哪两部门签署的？美日半导体协议是由美国商务部和日本经济产业省签署的。"
    query = "日美半导体协议要求美国芯片在日本市场份额是多少？根据日美半导体协议，要求美国芯片在日本市场的份额为20%。"
    query = "尼康和佳能的光刻机在哪个市场占优势？尼康和佳能都是知名的相机制造商，但在光刻机市场上，尼康占据着主导地位。尼康是全球最大的光刻机制造商之一，其光刻机产品广泛应用于半导体行业，尤其在高端光刻机市场上"
    ensemble_retriever = EnsembleRetriever(top_k=3, faiss_index=faiss_index, weights=[0.5, 0.5])
    t_result = ensemble_retriever.retrieve(str_or_query_bundle=query)
    print(t_result)
    faiss_index.reset()
