# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: vector_store_retriever.py
# @time: 2023/12/25 17:43
from typing import List

import numpy as np
from llama_index.schema import TextNode
from llama_index import QueryBundle
from llama_index.schema import NodeWithScore
from llama_index.retrievers import BaseRetriever
from llama_index.indices.query.schema import QueryType

from preprocess.get_text_id_mapping import text_node_id_mapping
from custom_retriever.build_embedding_cache import EmbeddingCache


class VectorSearchRetriever(BaseRetriever):
    def __init__(self, top_k, faiss_index) -> None:
        super().__init__()
        self.top_k = top_k
        self.faiss_index = faiss_index
        self.queries_embedding_dict, self.corpus_embedding, self.corpus = EmbeddingCache().load()
        # add vector
        self.faiss_index.add(self.corpus_embedding)

    def _retrieve(self, query: QueryType) -> List[NodeWithScore]:
        if isinstance(query, str):
            query = QueryBundle(query)
        else:
            query = query

        result = []
        # vector search
        query_embedding = self.queries_embedding_dict[query.query_str]
        distances, doc_indices = self.faiss_index.search(np.array([query_embedding]), self.top_k)

        for i, sent_index in enumerate(doc_indices.tolist()[0]):
            text = self.corpus[sent_index]
            node_with_score = NodeWithScore(node=TextNode(text=text, id_=text_node_id_mapping[text]),
                                            score=distances.tolist()[0][i])
            result.append(node_with_score)

        return result


if __name__ == '__main__':
    from faiss import IndexFlatIP
    faiss_index = IndexFlatIP(1536)
    vector_search_retriever = VectorSearchRetriever(top_k=2, faiss_index=faiss_index)
    t_result = vector_search_retriever.retrieve(str_or_query_bundle="索尼1953年引入的技术专利是什么？")
    print(t_result)
    faiss_index.reset()
