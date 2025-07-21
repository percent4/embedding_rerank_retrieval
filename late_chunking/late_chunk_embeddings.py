# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: late_chunk_embeddings.py
# @time: 2024/12/20 15:31
import json
from transformers import AutoModel
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

import warnings
warnings.filterwarnings('ignore')

file_path = '../data/doc_qa_test.json'
with open(file_path, 'r') as f:
    data = json.load(f)

corpus = []

for i in range(len(data['corpus'])):
    corpus.append(data['corpus'][f'node_{i+1}'])


for i, _ in enumerate(corpus):
    print(f'node_{i+1}: {_}')

node_id_dict = {}
_id = 0
for node_id, node_text in data['corpus'].items():
    node_id_dict[_id] = int(node_id.split('_')[-1]) - 1
    _id += 1
id_node_dict = {v: k for k, v in node_id_dict.items()}

print(node_id_dict)
print(id_node_dict)


total_text = ''.join(corpus)    # 全部文本
print(f"测试数据的全量字符数: {len(total_text)}")

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-zh', trust_remote_code=True)
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-zh', trust_remote_code=True)

# 获取每个text的token数量
corpus_token_num_list = [tokenizer(text, return_tensors='pt')['input_ids'].shape[1] - 2 for text in corpus]
print(corpus_token_num_list)
CLUSTER_MAX_TOKEN_NUM = 4000


# 对corpus_token_num_list按token_num进行聚合，每组的text长度不超过CLUSTER_MAX_TOKEN_NUM，但接可能接近CLUSTER_MAX_TOKEN_NUM
def merge_closest_to_n(arr, n):
    """
    合并连续的整数项，使得总和小于n且尽可能接近n。

    :param arr: List[int]，整数数组
    :param n: int，固定值
    :return: List[List[int]]，合并后的连续项目分组
    """
    result = []
    i = 0

    while i < len(arr):
        current_sum = 0
        temp_group = []

        for j in range(i, len(arr)):
            if current_sum + arr[j] < n:
                current_sum += arr[j]
                temp_group.append(arr[j])
            else:
                break

        if temp_group:
            result.append(temp_group)
            i += len(temp_group)  # 跳过合并的部分
        else:
            result.append([arr[i]])  # 无法合并的单独项
            i += 1

    return result


known_merge_list = [[1, 20], [21, 25], [26, 27], [28, 36], [37, 41], [42, 44], [45, 48], [49, 53], [54, 64], [65, 72], [73, 86], [87, 97]]
cluster_token_num_list = []
for merge_list in known_merge_list:
    cluster_token_num_list.append([corpus_token_num_list[i] for i in range(merge_list[0] - 1, merge_list[1])])

cluster_token_num_list.extend(merge_closest_to_n(corpus_token_num_list[97: ], CLUSTER_MAX_TOKEN_NUM))
print(cluster_token_num_list)
print(len(cluster_token_num_list))


# 对聚合后的cluster_token_num_list进行分组，获取拼接后的text和span_list
cluster_text_list = []
span_list_list = []
cnt = 0
for token_num_list in cluster_token_num_list:
    cluster_text = ''
    span_list = []
    start, end = 0, 0
    for i, token_num in enumerate(token_num_list):
        cluster_text += corpus[cnt]
        start = end if i else end + 1
        end = start + token_num if i else start + token_num - 1
        span_list.append((start, end))
        cnt += 1
    cluster_text_list.append(cluster_text)
    span_list_list.append(span_list)

print(cluster_text_list)
print(span_list_list)
# 统计span_list_list中的span数量
span_num = sum([len(_) for _ in span_list_list])
print(span_num)


# late chunking
def late_chunking(
    model_output: 'BatchEncoding', span_annotation: list, max_length=None
):
    token_embeddings = model_output[0]
    outputs = []
    for embeddings, annotations in zip(token_embeddings, span_annotation):
        if (
            max_length is not None
        ):  # remove annotations which go bejond the max-length of the model
            annotations = [
                (start, min(end, max_length - 1))
                for (start, end) in annotations
                if start < (max_length - 1)
            ]
        pooled_embeddings = [
            embeddings[start:end].sum(dim=0) / (end - start)
            for start, end in annotations
            if (end - start) >= 1
        ]
        pooled_embeddings = [
            embedding.detach().cpu().numpy() for embedding in pooled_embeddings
        ]
        outputs.append(pooled_embeddings)

    return outputs


embedding_sum = 0
embedding_data = np.empty(shape=[span_num, 768])
cnt = 0
for input_text, spans in tqdm(zip(cluster_text_list, span_list_list), desc="generate embedding"):
    print(input_text, spans)
    inputs = tokenizer(input_text, return_tensors='pt', max_length=CLUSTER_MAX_TOKEN_NUM, truncation=True)
    model_output = model(**inputs)
    embeddings = late_chunking(model_output, [spans])[0]
    # print(embeddings)
    print(len(embeddings))
    embedding_sum += len(embeddings)
    for embedding in embeddings:
        # 对embedding进行归一化, 使其范数为1
        embedding_norm = embedding / np.linalg.norm(embedding)
        print(f"_id: {id_node_dict[cnt]}")
        embedding_data[id_node_dict[cnt]] = embedding_norm
        cnt += 1
        # print(f"cnt: {cnt}")


np.save(f"../data/corpus_jina_base_zh_late_chunking_embedding.npy", embedding_data)

print("总共的embedding数量: ", embedding_sum)
