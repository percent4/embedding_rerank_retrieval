本项目是针对RAG中的Retrieve阶段的召回技术及算法效果所做评估实验。使用主体框架为`LlamaIndex`，版本为0.9.21.

Retrieve Method：

- BM25 Retriever
- Embedding Retriever(OpenAI, BGE, BGE-Finetune)
- Ensemble Retriever
- Ensemble Retriever + Cohere Rerank
- Ensemble Retriever + BGE-BASE Rerank
- Ensemble Retriever + BGE-LARGE Rerank

参考文章（也可查看`docs`文件夹）：

1. [NLP（八十二）RAG框架中的Retrieve算法评估](https://mp.weixin.qq.com/s?__biz=MzU2NTYyMDk5MQ==&mid=2247486199&idx=1&sn=f24175b05bdf5bc6dd42efed4d5acae8&chksm=fcb9b367cbce3a711fabd1a56bb5b9d803aba2f42964b4e1f9a4dc6e2174f0952ddb9e1d4c55&token=1977141018&lang=zh_CN#rd)
2. [NLP（八十三）RAG框架中的Rerank算法评估](https://mp.weixin.qq.com/s?__biz=MzU2NTYyMDk5MQ==&mid=2247486225&idx=1&sn=235eb787e2034f24554d8e997dbb4718&chksm=fcb9b281cbce3b9761342ebadbe001747ce2e74d84340f78b0e12c4d4c6aed7a7817f246c845&token=1977141018&lang=zh_CN#rd)
3. [NLP（八十四）RAG框架中的召回算法可视化分析及提升方法](https://mp.weixin.qq.com/s?__biz=MzU2NTYyMDk5MQ==&mid=2247486264&idx=1&sn=afa31ecc8b23724154a08090ccfab213&chksm=fcb9b2a8cbce3bbeb6daaee6308c10f097c32d304f076c3061718e669fd366c8aec9e6cf379d&token=823710334&lang=zh_CN#rd)
4. [NLP（八十六）RAG框架Retrieve阶段的Embedding模型微调](https://mp.weixin.qq.com/s?__biz=MzU2NTYyMDk5MQ==&mid=2247486333&idx=1&sn=29d00d472647bc5d6e336bec22c88139&chksm=fcb9b2edcbce3bfb42ea149d96fb1296b10a79a60db7ad2da01b85ab223394191205426bc025&token=1376257911&lang=zh_CN#rd)
5. [NLP（一百零一）Embedding模型微调实践](https://mp.weixin.qq.com/s/lJ3Mycjw1G99T08r8c7dSQ)
6. [NLP（一百零二）ReRank模型微调实践](https://mp.weixin.qq.com/s/RiPYANTyEgFtIIFHaKq3Rg)

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

ft-bge-rerank-base:

| retrievers                             | hit_rate | mrr      | 
|----------------------------------------|----------|----------|
| ensemble_ft_bge_base_rerank_top_1_eval | 0.8474   | 0.8474   |
| ensemble_ft_bge_base_rerank_top_2_eval | 0.9003   | 0.8816   |
| ensemble_ft_bge_base_rerank_top_3_eval | 0.9408   | 0.9102   | 
| ensemble_ft_bge_base_rerank_top_4_eval | 0.9533   | 0.9180   | 
| ensemble_ft_bge_base_rerank_top_5_eval | 0.9657   | 0.9240   | 


ft-bge-rerank-large:

| retrievers                              | hit_rate | mrr     |
|-----------------------------------------|----------|---------|
| ensemble_ft_bge_large_rerank_top_1_eval | 0.8474   | 0.8474  |
| ensemble_ft_bge_large_rerank_top_2_eval | 0.9003   | 0.8769  |
| ensemble_ft_bge_large_rerank_top_3_eval | 0.9439   | 0.9024  |
| ensemble_ft_bge_large_rerank_top_4_eval | 0.9564   | 0.9029  |
| ensemble_ft_bge_large_rerank_top_5_eval | 0.9688   | 0.9028  |


![不同Rerank模型的Hit Rate](https://s2.loli.net/2024/06/19/MrNFwl4IeKJGPxa.png)

## 不同Embedding模型之间的比较

jina-base-zh-embedding:

| retrievers           | hit_rate           | mrr                | cost_time          |
|----------------------|--------------------|--------------------|--------------------|
| embedding_top_1_eval | 0.5389408099688473 | 0.5389408099688473 | 34.9421501159668   |
| embedding_top_2_eval | 0.6448598130841121 | 0.5919003115264797 | 35.04490852355957  |
| embedding_top_3_eval | 0.7165109034267912 | 0.6157840083073729 | 40.548086166381836 |
| embedding_top_4_eval | 0.7476635514018691 | 0.6235721703011423 | 41.40806198120117  |
| embedding_top_5_eval | 0.7694704049844237 | 0.6279335410176532 | 43.450117111206055 |

bge-base-embedding:

| retrievers           | hit_rate           | mrr                | cost_time          |
|----------------------|--------------------|--------------------|--------------------|
| embedding_top_1_eval | 0.6043613707165109 | 0.6043613707165109 | 40.014028549194336 |
| embedding_top_2_eval | 0.7071651090342679 | 0.6557632398753894 | 38.26403617858887  |
| embedding_top_3_eval | 0.7538940809968847 | 0.6713395638629284 | 39.404869079589844 |
| embedding_top_4_eval | 0.7912772585669782 | 0.6806853582554517 | 43.24913024902344  |
| embedding_top_5_eval | 0.8099688473520249 | 0.684423676012461  | 53.58481407165527  |

bge-large-embedding:

| retrievers           | hit_rate           | mrr                | cost_time          |
|----------------------|--------------------|--------------------|--------------------|
| embedding_top_1_eval | 0.5919003115264797 | 0.5919003115264797 | 50.39501190185547  |
| embedding_top_2_eval | 0.7133956386292835 | 0.6526479750778816 | 52.02889442443848  |
| embedding_top_3_eval | 0.7725856697819314 | 0.6723779854620976 | 51.7120361328125   |
| embedding_top_4_eval | 0.794392523364486  | 0.6778296988577361 | 51.872968673706055 |
| embedding_top_5_eval | 0.822429906542056  | 0.6834371754932502 | 56.67304992675781  |

bge-m3-embedding:

| retrievers           | hit_rate           | mrr                | cost_time          |
|----------------------|--------------------|--------------------|--------------------|
| embedding_top_1_eval | 0.6822429906542056 | 0.6822429906542056 | 43.41626167297363  |
| embedding_top_2_eval | 0.778816199376947  | 0.7305295950155763 | 44.278860092163086 |
| embedding_top_3_eval | 0.8193146417445483 | 0.7440290758047767 | 45.64094543457031  |
| embedding_top_4_eval | 0.8504672897196262 | 0.7518172377985461 | 46.158790588378906 |
| embedding_top_5_eval | 0.8722741433021807 | 0.7561786085150571 | 50.23527145385742  |

bce-embedding:

| retrievers           | hit_rate           | mrr                | cost_time          |
|----------------------|--------------------|--------------------|--------------------|
| embedding_top_1_eval | 0.5794392523364486 | 0.5794392523364486 | 42.510032653808594 |
| embedding_top_2_eval | 0.6853582554517134 | 0.632398753894081  | 42.72007942199707  |
| embedding_top_3_eval | 0.7227414330218068 | 0.6448598130841121 | 41.066884994506836 |
| embedding_top_4_eval | 0.7507788161993769 | 0.6518691588785047 | 43.18714141845703  |
| embedding_top_5_eval | 0.7663551401869159 | 0.6549844236760125 | 44.08693313598633  |

bge-base-embedding-finetune:

| retrievers           | hit_rate           | mrr                | cost_time          |
|----------------------|--------------------|--------------------|--------------------|
| embedding_top_1_eval | 0.7289719626168224 | 0.7289719626168224 | 48.82097244262695  |
| embedding_top_2_eval | 0.8598130841121495 | 0.794392523364486  | 42.237043380737305 |
| embedding_top_3_eval | 0.9003115264797508 | 0.8078920041536863 | 42.33193397521973  |
| embedding_top_4_eval | 0.9065420560747663 | 0.8094496365524404 | 45.35722732543945  |
| embedding_top_5_eval | 0.9158878504672897 | 0.811318795430945  | 50.804853439331055 |

bge-large-embedding-finetune:

| retrievers           | hit_rate           | mrr                | cost_time          |
|----------------------|--------------------|--------------------|--------------------|
| embedding_top_1_eval | 0.7570093457943925 | 0.7570093457943925 | 47.14798927307129  |
| embedding_top_2_eval | 0.881619937694704  | 0.8193146417445483 | 44.70491409301758  |
| embedding_top_3_eval | 0.9190031152647975 | 0.8317757009345794 | 46.12398147583008  |
| embedding_top_4_eval | 0.9376947040498442 | 0.8364485981308412 | 49.448251724243164 |
| embedding_top_5_eval | 0.9376947040498442 | 0.8364485981308412 | 57.805776596069336 |

![不同Embedding模型之间的Hit Rate比较](https://s2.loli.net/2024/02/04/9ZHclTtyBN6CM8n.png)

![不同Embedding模型之间的MRR比较](https://s2.loli.net/2024/02/04/6UGQpCdlLoDAKiP.png)

## 可视化分析

![retrieval_website.png](https://s2.loli.net/2023/12/30/mZkJ37KRHTFSsyO.png)

| 检索类型 | 优点 | 缺点 |
|----------|------|------|
| 向量检索 (Embedding) | 1. 语义理解更强。<br>2. 能有效处理模糊或间接的查询。<br>3. 对自然语言的多样性适应性强。<br>4. 能识别不同词汇的相同意义。 | 1. 计算和存储成本高。<br>2. 索引时间较长。<br>3. 高度依赖训练数据的质量和数量。<br>4. 结果解释性较差。 |
| 关键词检索 (BM25) | 1. 检索速度快。<br>2. 实现简单，资源需求低。<br>3. 结果易于理解，可解释性强。<br>4. 对精确查询表现良好。 | 1. 对复杂语义理解有限。<br>2. 对查询变化敏感，灵活性差。<br>3. 难以处理同义词和多义词。<br>4. 需要用户准确使用关键词。 |

- `query`: "NEDO"的全称是什么？

![Embedding召回优于BM25](https://s2.loli.net/2024/01/20/Uh1FGYJT26ONd3t.png)

在这个例子中，Embedding召回结果优于BM25，BM25召回结果虽然在top_3结果中存在，但排名第三，排在首位的是不相关的文本，而Embedding由于文本相似度的优势，将正确结果放在了首位。

- `query`: 日本半导体产品的主要应用领域是什么？

![BM25召回优于Embedding](https://s2.loli.net/2024/01/20/BSO19sKko8gclem.png)

在这个例子中，BM25召回结果优于Embedding。

- `query`: 《美日半导体协议》对日本半导体市场有何影响？

![ensemble算法提升了排名](https://s2.loli.net/2024/01/20/wHU4LP7iRXfQ5CW.png)

在这个例子中，正确文本在BM25算法召回结果中排名第二，在Embedding算法中排第三，混合搜索排名第一，这里体现了混合搜索的优越性。

- `query`: 80年代日本电子产业的辉煌表现在哪些方面？

![Rerank的优越性](https://s2.loli.net/2024/01/20/6S1wBXv7caZDCkd.png)

在这个例子中，不管是BM25, Embedding,还是Ensemble，都没能将正确文本排在第一位，而经过Rerank以后，正确文本排在第一位，这里体现了Rerank算法的优势。

## 改进方案

1. Query Rewrite

- 原始query: 半导体制造设备市场美、日、荷各占多少份额？
- 改写后query：美国、日本和荷兰在半导体制造设备市场的份额分别是多少？

改写后的query在BM25和Embedding的top 3召回结果中都能找到。该query对应的正确文本为：

> 全球半导体设备制造领域，美国、日本和荷兰控制着全球370亿美元半导体制造设备市场的90％以上。其中，美国的半导体制造设备（SME）产业占全球产量的近50%，日本约占30%，荷兰约占17％%。更具体地，以光刻机为例，EUV光刻工序其实有众多日本厂商的参与，如东京电子生产的EUV涂覆显影设备，占据100%的市场份额，Lasertec Corp.也是全球唯一的测试机制造商。另外还有EUV光刻胶，据南大光电在3月发布的相关报告中披露，全球仅有日本厂商研发出了EUV光刻胶。

从中我们可以看到，在改写后的query中，美国、日本、荷兰这三个词发挥了重要作用，因此，**query改写对于含有缩写的query有一定的召回效果改善**。

2. HyDE

HyDE（全称Hypothetical Document Embeddings）是RAG中的一种技术，它基于一个假设：相较于直接查询，通过大语言模型 (LLM) 生成的答案在嵌入空间中可能更为接近。HyDE 首先响应查询生成一个假设性文档（答案），然后将其嵌入，从而提高搜索的效果。

比如：

- 原始query: 美日半导体协议是由哪两部门签署的？
- 加上回答后的query: 美日半导体协议是由哪两部门签署的？美日半导体协议是由美国商务部和日本经济产业省签署的。

加上回答后的query使用BM25算法可以找回正确文本，且排名第一位，而Embedding算法仍无法召回。

正确文本为：

> 1985年6月，美国半导体产业贸易保护的调子开始升高。美国半导体工业协会向国会递交一份正式的“301条款”文本，要求美国政府制止日本公司的倾销行为。民意调查显示，68%的美国人认为日本是美国最大的威胁。在舆论的引导和半导体工业协会的推动下，美国政府将信息产业定为可以动用国家安全借口进行保护的新兴战略产业，半导体产业成为美日贸易战的焦点。1985年10月，美国商务部出面指控日本公司倾销256K和1M内存。一年后，日本通产省被迫与美国商务部签署第一次《美日半导体协议》。

从中可以看出，大模型的回答是正确的，美国商务部这个关键词发挥了重要作用，因此，HyDE对于特定的query有召回效果提升。