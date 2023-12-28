# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: bm25_retriever.py
# @time: 2023/12/25 17:42
from typing import List

from elasticsearch import Elasticsearch
from llama_index.schema import TextNode
from llama_index import QueryBundle
from llama_index.schema import NodeWithScore
from llama_index.retrievers import BaseRetriever
from llama_index.indices.query.schema import QueryType

from preprocess.get_text_id_mapping import text_node_id_mapping


class CustomBM25Retriever(BaseRetriever):
    """Custom retriever for elasticsearch with bm25"""
    def __init__(self, top_k) -> None:
        """Init params."""
        super().__init__()
        self.es_client = Elasticsearch([{'host': 'localhost', 'port': 9200}])
        self.top_k = top_k

    def _retrieve(self, query: QueryType) -> List[NodeWithScore]:
        if isinstance(query, str):
            query = QueryBundle(query)
        else:
            query = query

        result = []
        # 查询数据(全文搜索)
        dsl = {
            'query': {
                'match': {
                    'content': query.query_str
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


if __name__ == '__main__':
    custom_bm25_retriever = CustomBM25Retriever(top_k=3)
    t_result = custom_bm25_retriever.retrieve(str_or_query_bundle="请根据灯塔专业版数据，列出2019年至2022年暑期档电影市场的总票房，并分析其变化趋势。")
    print(t_result)
