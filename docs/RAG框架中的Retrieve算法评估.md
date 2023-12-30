> 本文将详细介绍RAG框架中的各种Retrieve算法，比如BM25, Embedding Search, Ensemble Search, Rerank等的评估实验过程与结果。本文是目前除了LlamaIndex官方网站例子之外为数不多的介绍Retrieve算法评估实验的文章。

## 什么是RAG中的Retrieve？

`RAG`即Retrieval Augmented Generation的简称，是现阶段增强使用LLM的常见方式之一，其一般步骤为：

1.  文档划分（Document Split）
2.  向量嵌入（Embedding）
3.  文档获取（Retrieve）
4.  Prompt工程（Prompt Engineering）
5.  大模型问答（LLM）

大致的流程图参考如下：

![](https://towhee.io/assets/img/task/retrieval-augmented-generation.png)

通常来说，可将`RAG`划分为召回（**Retrieve**）阶段和答案生成(**Answer Generate**)阶段，而效果优化也从这方面入手。针对召回阶段，文档获取是其中重要的步骤，决定了注入大模型的知识背景，常见的召回算法如下：

- **BM25（又称Keyword Search）**: 使用BM24算法找回相关文档，一般对于特定领域关键词效果较好，比如人名，结构名等；
- **Embedding Search**: 使用Embedding模型将query和corpus进行文本嵌入，使用向量相似度进行文本匹配，可解决BM25算法的相似关键词召回效果差的问题，该过程一般会使用向量数据库（Vector Database）；
- **Ensemble Search**: 融合BM25算法和Embedding Search的结果，使用RFF算法进行重排序，一般会比单独的召回算法效果好；
- **Rerank**: 上述的召回算法一般属于粗召回阶段，更看重性能；Rerank是对粗召回阶段的结果，再与query进行文本匹配，属于Rerank（又称为重排、精排）阶段，更看重效果；

综合上述Retrieve算法的框架示意图如下：

![](https://1673940196-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FRncMhlfeYTrpujwzDIqw%2Fuploads%2FoalmRC4UOlhQNF0hFaBR%2Fspaces_CdDIVDY6AtAz028MFT4d_uploads_ohgmBurknjsKmg53Z00U_image.webp?alt=media&token=33e4c026-8d5e-4e77-98b2-f1dcce42a15b)

上述的Retrieve算法更有优劣，一般会选择合适的场景进行使用或考虑综合几种算法进行使用。那么，它们的效果具体如何呢？


## Retrieve算法评估

那么，如何对Retrieve算法进行具体效果评估呢？

本文将通过构造自有数据集进行测试，分别对上述四种Retrieve算法进行实验，采用`Hit Rate`和`MRR`指标进行评估。

在**LlamaIndex**官方Retrieve Evaluation中，提供了对Retrieve算法的评估示例，具体细节可参考如下：

[https://blog.llamaindex.ai/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83](https://blog.llamaindex.ai/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83)

这是现在网上较为权威的Retrieve Evaluation实验，本文将参考LlamaIndex的做法，给出更为详细的评估实验过程与结果。

Retrieve Evaluation实验的步骤如下：

1. `文档划分`：寻找合适数据集，进行文档划分；
2. `问题生成`：对划分后的文档，使用LLM对文档内容生成问题；
3. `召回文本`：对生成的每个问题，采用不同的Retrieve算法，得到召回结果；
4. `指标评估`：使用`Hit Rate`和`MRR`指标进行评估

步骤是清晰的，那么，我们来看下评估指标：`Hit Rate`和`MRR`。

`Hit Rate`即命中率，一般指的是我们预期的召回文本（真实值）在召回结果的前k个文本中会出现，也就是Recall@k时，能得到预期文本。一般，`Hit Rate`越高，就说明召回算法效果越好。

`MRR`即Mean Reciprocal Rank，是一种常见的评估检索效果的指标。MRR 是衡量系统在一系列查询中返回相关文档或信息的平均排名的逆数的平均值。例如，如果一个系统对第一个查询的正确答案排在第二位，对第二个查询的正确答案排在第一位，则 MRR 为 (1/2 + 1/1) / 2。

在LlamaIndex中，这两个指标的对应类分别为`HitRate`和`MRR`，源代码如下：

```python
class HitRate(BaseRetrievalMetric):
    """Hit rate metric."""

    metric_name: str = "hit_rate"

    def compute(
        self,
        query: Optional[str] = None,
        expected_ids: Optional[List[str]] = None,
        retrieved_ids: Optional[List[str]] = None,
        expected_texts: Optional[List[str]] = None,
        retrieved_texts: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> RetrievalMetricResult:
        """Compute metric."""
        if retrieved_ids is None or expected_ids is None:
            raise ValueError("Retrieved ids and expected ids must be provided")
        is_hit = any(id in expected_ids for id in retrieved_ids)
        return RetrievalMetricResult(
            score=1.0 if is_hit else 0.0,
        )


class MRR(BaseRetrievalMetric):
    """MRR metric."""

    metric_name: str = "mrr"

    def compute(
        self,
        query: Optional[str] = None,
        expected_ids: Optional[List[str]] = None,
        retrieved_ids: Optional[List[str]] = None,
        expected_texts: Optional[List[str]] = None,
        retrieved_texts: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> RetrievalMetricResult:
        """Compute metric."""
        if retrieved_ids is None or expected_ids is None:
            raise ValueError("Retrieved ids and expected ids must be provided")
        for i, id in enumerate(retrieved_ids):
            if id in expected_ids:
                return RetrievalMetricResult(
                    score=1.0 / (i + 1),
                )
        return RetrievalMetricResult(
            score=0.0,
        )
```

## 数据集构造

在文章[NLP（六十一）使用Baichuan-13B-Chat模型构建智能文档](https://mp.weixin.qq.com/s?__biz=MzU2NTYyMDk5MQ==&mid=2247485425&idx=1&sn=bd85ddfce82d77ceec5a66cb96835400&chksm=fcb9be61cbce37773109f9703c2b6c4256d5037c8bf4497dfb9ad0f296ce0ee4065255954c1c&token=1977141018&lang=zh_CN#rd)笔者介绍了如何使用RAG框架来实现智能文档问答。

以这个项目为基础，笔者采集了日本半导体行业相关的网络文章及其他文档，进行文档划分，导入至ElastricSearch，并使用OpenAI Embedding获取文本嵌入向量。语料库一共为433个文档片段（Chunk），其中321个与日本半导体行业相关（不妨称之为`领域文档`）。

还差query数据集。这点是从LlamaIndex官方示例中获取的灵感：**使用大模型生成query**!

针对上述321个领域文档，使用GPT-4模型生成一个与文本内容相关的问题，即query，Python代码如下：

```python
# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: data_transfer.py
# @time: 2023/12/25 17:51
import pandas as pd
from llama_index.llms import OpenAI
from llama_index.schema import TextNode
from llama_index.evaluation import generate_question_context_pairs
import random
random.seed(42)

llm = OpenAI(model="gpt-4", max_retries=5)

# Prompt to generate questions
qa_generate_prompt_tmpl = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a university professor. Your task is to set {num_questions_per_chunk} questions for the upcoming Chinese quiz.
Questions throughout the test should be diverse. Questions should not contain options or start with Q1/Q2.
Questions must be written in Chinese. The expression must be concise and clear. 
It should not exceed 15 Chinese characters. Words such as "这", "那", "根据", "依据" and other punctuation marks 
should not be used. Abbreviations may be used for titles and professional terms.
"""

nodes = []
data_df = pd.read_csv("../data/doc_qa_dataset.csv", encoding="utf-8")
for i, row in data_df.iterrows():
    if len(row["content"]) > 80 and i > 96:
        node = TextNode(text=row["content"])
        node.id_ = f"node_{i + 1}"
        nodes.append(node)


doc_qa_dataset = generate_question_context_pairs(
    nodes, llm=llm, num_questions_per_chunk=1, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl
)

doc_qa_dataset.save_json("../data/doc_qa_dataset.json")
```

原始数据`doc_qa_dataset.csv`是笔者从Kibana中的Discover中导出的，使用llama-index模块和GPT-4模型，以合适的Prompt，对每个领域文档生成一个问题，并保存为doc_qa_dataset.json，这就是我们进行Retrieve Evaluation的数据格式，其中包括queries, corpus, relevant_docs, mode四个字段。

我们来查看第一个文档及生成的答案，如下：

```json
{
    "queries": {
        "7813f025-333d-494f-bc14-a51b2d57721b": "日本半导体产业的现状和影响因素是什么？",
        ...
    },
    "corpus": {
        "node_98": "日本半导体产业在上世纪80年代到达顶峰后就在缓慢退步，但若简单认为日本半导体产业失败了，就是严重误解，今天日本半导体产业仍有非常有竞争力的企业和产品。客观认识日本半导体产业的成败及其背后的原因，对正在大力发展半导体产业的中国，有非常强的参考价值。",
        ...
    },
    "relevant_docs": {
        "7813f025-333d-494f-bc14-a51b2d57721b": [
            "node_98"
        ],
        ...
    },
    "mode": "text"
}
```


## Retrieve算法评估

我们需要评估的Retrieve算法为BM25, Embedding Search, Ensemble Search,  Ensemble + Rerank，下面将分别就Retriever实现方式、指标评估实验对每种Retrieve算法进行详细介绍。

### BM25

BM25的储存采用ElasticSearch，即直接使用ES内置的BM25算法。笔者在llama-index对BaseRetriever进行定制化开发（这也是我们实现自己想法的一种常规方法），对其简单封装，Python代码如下：

```python
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
```

之后，对top_k结果进行指标评估，Python代码如下：

```python
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
evaluation_name_list = []
evaluation_result_list = []
cost_time_list = []
for top_k in [1, 2, 3, 4, 5]:
    start_time = time.time()
    bm25_retriever = CustomBM25Retriever(top_k=top_k)
    bm25_retriever_evaluator = RetrieverEvaluator.from_metric_names(metrics, retriever=bm25_retriever)
    bm25_eval_results = asyncio.run(bm25_retriever_evaluator.aevaluate_dataset(doc_qa_dataset))
    evaluation_name_list.append(f"bm25_top_{top_k}_eval")
    evaluation_result_list.append(bm25_eval_results)
    cost_time_list.append((time.time() - start_time) * 1000)

print("done for bm25 evaluation!")
df = display_results(evaluation_name_list, evaluation_result_list)
df['cost_time'] = cost_time_list
print(df.head())
df.to_csv(f"evaluation_bm25_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.csv", encoding="utf-8", index=False)
```

BM25算法的实验结果如下：

| retrievers      | hit_rate | mrr    | cost_time |
|-----------------|----------|--------|-----------|
| bm25_top_1_eval | 0.7975   | 0.7975 | 461.277   |
| bm25_top_2_eval | 0.8536   | 0.8255 | 510.3021  |
| bm25_top_3_eval | 0.9003   | 0.8411 | 570.6708  |
| bm25_top_4_eval | 0.9159   | 0.845  | 420.7261  |
| bm25_top_5_eval | 0.9408   | 0.85   | 388.5961  |

### Embedding Search

BM25算法的实现是简单的。Embedding Search的较为复杂些，首先需要对queries和corpus进行文本嵌入，这里的Embedding模型使用Openai的text-embedding-ada-002，向量维度为1536，并将结果存入numpy数据结构中，保存为npy文件，方便后续加载和重复使用。

为了避免使用过重的向量数据集，本实验采用内存向量数据集: **faiss**。使用faiss加载向量，index类型选用IndexFlatIP，并进行向量相似度搜索。

Embedding Search也需要定制化开发Retriever及指标评估，这里不再赘述，具体实验可参考文章末尾的Github项目地址。

Embedding Search的实验结果如下：

| retrievers           | hit_rate | mrr    | cost_time |
|----------------------|----------|--------|-----------|
| embedding_top_1_eval | 0.6075   | 0.6075 | 67.6837   |
| embedding_top_2_eval | 0.6978   | 0.6526 | 60.8449   |
| embedding_top_3_eval | 0.7321   | 0.6641 | 59.9051   |
| embedding_top_4_eval | 0.7788   | 0.6758 | 63.5488   |
| embedding_top_5_eval | 0.7944   | 0.6789 | 67.7922   |

> 注意: 这里的召回时间花费比BM25还要少，完全得益于我们已经存储好了文本向量，并使用faiss进行加载、查询。

### Ensemble Search

Ensemble Search融合BM25算法和Embedding Search算法，针对两种算法召回的top_k个文档，使用RRF算法进行重新排序，再获取top_k个文档。RRF算法是经典且优秀的集成排序算法，这里不再展开介绍，后续专门写文章介绍。

Ensemble Search的实验结果如下：

| retrievers          | hit_rate | mrr    | cost_time |
|---------------------|----------|--------|-----------|
| ensemble_top_1_eval | 0.7009   | 0.7009 | 1072.7379 |
| ensemble_top_2_eval | 0.8536   | 0.7741 | 1088.8782 |
| ensemble_top_3_eval | 0.8941   | 0.7928 | 980.7949  |
| ensemble_top_4_eval | 0.919    | 0.8017 | 935.1702  |
| ensemble_top_5_eval | 0.9377   | 0.8079 | 868.299   |

### Ensemble + Rerank

如果还想在Ensemble Search的基础上再进行效果优化，可考虑加入Rerank算法。常见的Rerank模型有Cohere（API调用），BGE-Rerank（开源模型）等。本文使用Cohere Rerank API.

 Ensemble + Rerank的实验结果如下：
 
 | retrievers                 | hit_rate | mrr    | cost_time    |
|----------------------------|----------|--------|--------------|
| ensemble_rerank_top_1_eval | 0.8349   | 0.8349 | 2140632.4041 |
| ensemble_rerank_top_2_eval | 0.9034   | 0.8785 | 2157657.2871 |
| ensemble_rerank_top_3_eval | 0.9346   | 0.9008 | 2200800.936  |
| ensemble_rerank_top_4_eval | 0.947    | 0.9078 | 2150398.7348 |
| ensemble_rerank_top_5_eval | 0.9657   | 0.9099 | 2149122.9382 |

## 指标可视化及分析

### 指标可视化

上述的评估结果不够直观，我们使用Plotly模块绘制指标的条形图，结果如下：

![Hit Rate](https://s2.loli.net/2023/12/28/5VjRy7rCeXOtAZq.png)

![MRR](https://s2.loli.net/2023/12/28/s9SvU4kL7Zc1MK5.png)

### 指标分析

我们对上述统计图进行指标分析，可得到结论如下：

- 对于每种Retrieve算法，随着k的增加，top_k的Hit Rate指标和MRR指标都有所增加，即检索效果变好，这是显而易见的结论；
- 就总体检索效果而言，Ensemble + Rerank > Ensemble > 单独的Retrieve
- 本项目中就单独的Retrieve算法而言，BM25的检索效果比Embedding Search好（可能与生成的问答来源于文档有关），但这不是普遍结论，两种算法更有合适的场景
- 加入Rerank后，检索效果可获得一定的提升，以top_3评估结果来说，ensemble的Hit Rate为0.8941，加入Rerank后为0.9346，提升约4%

## 总结

本文详细介绍了RAG框架，并结合自有数据集对各种Retrieve算法进行评估。笔者通过亲身实验和编写Retriever代码，深入了解了RAG框架中的经典之作LlamaIndex，同时，本文也是难得的介绍RAG框架Retrieve阶段评估实验的文章。

本文的所有过程及指标结果已开源至Github，网址为：[https://github.com/percent4/embedding_rerank_retrieval](https://github.com/percent4/embedding_rerank_retrieval) .

后续，笔者将在此项目基础上，验证各种优化RAG框架Retrieve效果的手段，比如Query Rewrite, Query Transform, HyDE等，这将是一个获益无穷的项目啊！

## 参考文献

1. Retrieve Evaluation官网文章：https://blog.llamaindex.ai/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83
2. Retrieve Evaluation Colab上的代码：https://colab.research.google.com/drive/1TxDVA__uimVPOJiMEQgP5fwHiqgKqm4-?usp=sharing
3. LlamaIndex官网：https://docs.llamaindex.ai/en/stable/index.html
4. RetrieverEvaluator in LlamaIndex: https://docs.llamaindex.ai/en/stable/module_guides/evaluating/usage_pattern_retrieval.html
5. NLP（六十一）使用Baichuan-13B-Chat模型构建智能文档: https://mp.weixin.qq.com/s?__biz=MzU2NTYyMDk5MQ==&mid=2247485425&idx=1&sn=bd85ddfce82d77ceec5a66cb96835400&chksm=fcb9be61cbce37773109f9703c2b6c4256d5037c8bf4497dfb9ad0f296ce0ee4065255954c1c&token=1977141018&lang=zh_CN#rd
6. NLP（六十九）智能文档助手升级: https://mp.weixin.qq.com/s?__biz=MzU2NTYyMDk5MQ==&mid=2247485609&idx=1&sn=f8337b4822b1cdf95a586af6097ef288&chksm=fcb9b139cbce382f735e4c119ade8084067cde0482910c72767f36a29e7291385cbe6dfbd6a9&payreadticket=HBB91zkl4I6dXpw0Q4OcOF8ECZz0pS9kOGHJqycwrN7jFWHyUOCBe7sWFWytD7_9wo_NzcM#rd
7. NLP（八十一）智能文档问答助手项目改进: https://mp.weixin.qq.com/s?__biz=MzU2NTYyMDk5MQ==&mid=2247486103&idx=1&sn=caa204eda0760bab69b7e40abff8e696&chksm=fcb9b307cbce3a1108d305ec44281e3446241e90e9c17d62dd0b6eaa48cba5e20d31f0129584&token=1977141018&lang=zh_CN#rd
