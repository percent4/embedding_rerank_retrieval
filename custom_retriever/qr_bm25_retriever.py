# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: qr_bm25_retriever.py
# @time: 2025/1/8 15:10
import json
from typing import List
from operator import itemgetter

from elasticsearch import Elasticsearch
from llama_index.schema import TextNode
from llama_index import QueryBundle
from llama_index.schema import NodeWithScore
from llama_index.retrievers import BaseRetriever
from llama_index.indices.query.schema import QueryType

from preprocess.get_text_id_mapping import text_node_id_mapping


class QueryRewriteBM25Retriever(BaseRetriever):
    """Custom retriever for elasticsearch with bm25"""
    def __init__(self, top_k) -> None:
        """Init params."""
        super().__init__()
        self.es_client = Elasticsearch("http://localhost:9200")
        self.top_k = top_k
        self.c: int = 60
        with open('../data/query_rewrite.json', 'r') as f:
            self.query_write_dict = json.loads(f.read())

    def single_retrieve(self, query: str) -> List[NodeWithScore]:
        result = []
        # 查询数据(全文搜索)
        dsl = {
            'query': {
                'match': {
                    'content': query
                }
            },
            "size": self.top_k
        }
        search_result = self.es_client.search(index='docs', body=dsl)
        if search_result['hits']['hits']:
            for record in search_result['hits']['hits']:
                text = record['_source']['content']
                node_with_score = NodeWithScore(node=TextNode(text=text,
                                                id_=text_node_id_mapping[text]),
                                                score=record['_score'])
                result.append(node_with_score)

        return result

    def _retrieve(self, query: QueryBundle) -> List[NodeWithScore]:
        """retrieve after query rewrite."""
        if isinstance(query, str):
            query = QueryBundle(query)
        else:
            query = query

        doc_lists = []
        bm25_search_nodes = self.single_retrieve(query.query_str)
        doc_lists.append([node.text for node in bm25_search_nodes])
        # check: need query rewrite
        for search_query in self.query_write_dict[query.query_str]:
            bm25_search_nodes = self.single_retrieve(search_query)
            doc_lists.append([node.text for node in bm25_search_nodes])
        # print("len of documents: ", len(doc_lists))

        # Create a union of all unique documents in the input doc_lists
        all_documents = set()
        for doc_list in doc_lists:
            for doc in doc_list:
                all_documents.add(doc)
        # print(all_documents)

        # Initialize the RRF score dictionary for each document
        rrf_score_dic = {doc: 0.0 for doc in all_documents}

        # Calculate RRF scores for each document
        for doc_list, weight in zip(doc_lists, [1 / len(doc_lists)] * len(doc_lists)):
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
    from pprint import pprint
    qr_bm25_retriever = QueryRewriteBM25Retriever(top_k=3)
    test_query = "美日半导体协议是由哪两部门签署的？"
    t_result = qr_bm25_retriever.retrieve(str_or_query_bundle=test_query)
    pprint(t_result)
