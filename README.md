本项目是针对RAG中的Retrieve阶段的召回技术及算法效果所做评估实验。使用主体框架为`LlamaIndex`，版本为0.9.21.

Retrieve Method：

- BM25 Retriever
- Embedding Retriever
- Ensemble Retriever
- Ensemble Retriever + Cohere Rerank
- Ensemble Retriever + BGE-BASE Rerank
- Ensemble Retriever + BGE-LARGE Rerank

参考文章：[NLP（八十二）RAG框架中的Retrieve算法评估](https://mp.weixin.qq.com/s?__biz=MzU2NTYyMDk5MQ==&mid=2247486199&idx=1&sn=f24175b05bdf5bc6dd42efed4d5acae8&chksm=fcb9b367cbce3a711fabd1a56bb5b9d803aba2f42964b4e1f9a4dc6e2174f0952ddb9e1d4c55&token=1977141018&lang=zh_CN#rd)

## 数据

## 评估结果

BM25 Retriever Evaluation:

| retrievers      | hit_rate           | mrr                | cost_time          |
|-----------------|--------------------|--------------------|--------------------|
| bm25_top_1_eval | 0.7975077881619937 | 0.7975077881619937 | 461.2770080566406  |
| bm25_top_2_eval | 0.8535825545171339 | 0.8255451713395638 | 510.3020668029785  |
| bm25_top_3_eval | 0.9003115264797508 | 0.8411214953271028 | 570.6708431243896  |
| bm25_top_4_eval | 0.9158878504672897 | 0.8450155763239875 | 420.72606086730957 |
| bm25_top_5_eval | 0.940809968847352  | 0.8500000000000001 | 388.5960578918457  |

Embedding Retriever Evaluation:

| retrievers           | hit_rate           | mrr                | cost_time          |
|----------------------|--------------------|--------------------|--------------------|
| embedding_top_1_eval | 0.6074766355140186 | 0.6074766355140186 | 67.68369674682617  |
| embedding_top_2_eval | 0.6978193146417445 | 0.6526479750778816 | 60.84489822387695  |
| embedding_top_3_eval | 0.7320872274143302 | 0.6640706126687436 | 59.905052185058594 |
| embedding_top_4_eval | 0.778816199376947  | 0.6757528556593978 | 63.54880332946777  |
| embedding_top_5_eval | 0.794392523364486  | 0.6788681204569056 | 67.79217720031738  |

Ensemble Retriever Evaluation:

| retrievers          | hit_rate           | mrr                | cost_time          |
|---------------------|--------------------|--------------------|--------------------|
| ensemble_top_1_eval | 0.7009345794392523 | 0.7009345794392523 | 1072.7379322052002 |
| ensemble_top_2_eval | 0.8535825545171339 | 0.7741433021806854 | 1088.8781547546387 |
| ensemble_top_3_eval | 0.8940809968847352 | 0.7928348909657321 | 980.7949066162109  |
| ensemble_top_4_eval | 0.9190031152647975 | 0.8016614745586708 | 935.1701736450195  |
| ensemble_top_5_eval | 0.9376947040498442 | 0.8078920041536861 | 868.2990074157715  |

Ensemble Retriever + Rerank Evaluation:

| retrievers                 | hit_rate           | mrr                | cost_time         |
|----------------------------|--------------------|--------------------|-------------------|
| ensemble_rerank_top_1_eval | 0.8348909657320872 | 0.8348909657320872 | 2140632.404088974 |
| ensemble_rerank_top_2_eval | 0.9034267912772586 | 0.8785046728971962 | 2157657.287120819 |
| ensemble_rerank_top_3_eval | 0.9345794392523364 | 0.9008307372793353 | 2200800.935983658 |
| ensemble_rerank_top_4_eval | 0.9470404984423676 | 0.9078400830737278 | 2150398.734807968 |
| ensemble_rerank_top_5_eval | 0.9657320872274143 | 0.9098650051921081 | 2149122.938156128 |

![Hit Rate](https://s2.loli.net/2023/12/28/5VjRy7rCeXOtAZq.png)

![MRR](https://s2.loli.net/2023/12/28/s9SvU4kL7Zc1MK5.png)

## 不同Rerank算法之间的比较

bge-rerank-base:

| retrievers                          | hit_rate | mrr    |
|-------------------------------------|----------|--------|
| ensemble_bge_base_rerank_top_1_eval | 0.8255   | 0.8255 |
| ensemble_bge_base_rerank_top_2_eval | 0.8785   | 0.8489 |
| ensemble_bge_base_rerank_top_3_eval | 0.9346   | 0.8686 |
| ensemble_bge_base_rerank_top_4_eval | 0.947    | 0.872  |
| ensemble_bge_base_rerank_top_5_eval | 0.9564   | 0.8693 |

bge-rerank-large:

| retrievers                           | hit_rate | mrr    |
|--------------------------------------|----------|--------|
| ensemble_bge_large_rerank_top_1_eval | 0.8224   | 0.8224 |
| ensemble_bge_large_rerank_top_2_eval | 0.8847   | 0.8364 |
| ensemble_bge_large_rerank_top_3_eval | 0.9377   | 0.8572 |
| ensemble_bge_large_rerank_top_4_eval | 0.9502   | 0.8564 |
| ensemble_bge_large_rerank_top_5_eval | 0.9626   | 0.8537 |

![不同Rerank模型的Hit Rate](https://s2.loli.net/2023/12/29/vsuXBtbLdaVDS39.png)

## 可视化分析

![retrieval_website.png](https://s2.loli.net/2023/12/30/mZkJ37KRHTFSsyO.png)

## 改进方案

1. Query Transform
2. HyDE