import gradio as gr
import numpy as np
from transformers import AutoModel, AutoTokenizer

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-zh', trust_remote_code=True)
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-zh', trust_remote_code=True)


def chunk_by_sentences(input_text: str, tokenizer: callable, separator: str):
    inputs = tokenizer(input_text, return_tensors='pt', return_offsets_mapping=True)
    punctuation_mark_id = tokenizer.convert_tokens_to_ids(separator)
    print(f"separator: {separator}, punctuation_mark_id: {punctuation_mark_id}")
    sep_id = tokenizer.eos_token_id
    token_offsets = inputs['offset_mapping'][0]
    token_ids = inputs['input_ids'][0]
    chunk_positions = [
        (i, int(start + 1))
        for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets))
        if token_id == punctuation_mark_id
           and (
                   token_offsets[i + 1][0] - token_offsets[i][1] >= 0
                   or token_ids[i + 1] == sep_id
           )
    ]
    chunks = [
        input_text[x[1]: y[1]]
        for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    span_annotations = [
        (x[0], y[0]) for (x, y) in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    return chunks, span_annotations


def late_chunking(model_output, span_annotation, max_length=None):
    token_embeddings = model_output[0]
    outputs = []
    for embeddings, annotations in zip(token_embeddings, span_annotation):
        if max_length is not None:
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


def embedding_retriever(query_input, text_input, separator):
    chunks, span_annotations = chunk_by_sentences(text_input, tokenizer, separator)
    print(f"chunks: ", chunks)
    inputs = tokenizer(text_input, return_tensors='pt', max_length=4096, truncation=True)
    model_output = model(**inputs)
    late_chunking_embeddings = late_chunking(model_output, [span_annotations])[0]

    query_inputs = tokenizer(query_input, return_tensors='pt')
    query_embedding = model(**query_inputs)[0].detach().cpu().numpy().mean(axis=1)

    traditional_chunking_embeddings = model.encode(chunks)

    cos_sim = lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    naive_embedding_score_dict = {}
    late_chunking_embedding_score_dict = {}
    for chunk, trad_embed, new_embed in zip(chunks, traditional_chunking_embeddings, late_chunking_embeddings):
        # 计算query和每个chunk的embedding的cosine similarity，相似度分数转化为float类型
        naive_embedding_score_dict[chunk] = round(cos_sim(query_embedding, trad_embed).tolist()[0], 4)
        late_chunking_embedding_score_dict[chunk] = round(cos_sim(query_embedding, new_embed).tolist()[0], 4)

    naive_embedding_order = sorted(
        naive_embedding_score_dict.items(), key=lambda x: x[1], reverse=True
    )
    late_chunking_order = sorted(
        late_chunking_embedding_score_dict.items(), key=lambda x: x[1], reverse=True
    )

    df_data = []
    for i in range(len(naive_embedding_order)):
        df_data.append([i+1, naive_embedding_order[i][0], naive_embedding_order[i][1],
                        late_chunking_order[i][0], late_chunking_order[i][1]])
    return df_data


if __name__ == '__main__':
    with gr.Blocks() as demo:
        query = gr.TextArea(lines=1, placeholder="your query", label="Query")
        text = gr.TextArea(lines=3, placeholder="your text", label="Text")
        sep = gr.TextArea(lines=1, placeholder="your separator", label="Separator")
        submit = gr.Button("Submit")
        result = gr.DataFrame(headers=["order", "naive_embedding_text", "naive_embedding_score",
                                       "late_chunking_text", "late_chunking_score"],
                              label="Retrieve Result",
                              wrap=True)
        examples = gr.Examples(
            examples=[
                ["王安石是哪里人？",
                 "王安石（1021年12月19日－1086年5月21日），字介甫，号半山。抚州临川县（今属江西省抚州市）人。中国北宋时期政治家、文学家、思想家、改革家。庆历二年（1042年），王安石中进士，历任扬州签判、鄞县知县、舒州通判等职，政绩显著。宋仁宗末年，曾作《上仁宗皇帝言事书》，要求对宋初以来的法度进行全盘改革，但未被采纳。",
                 "。"]
            ],
            inputs=[query, text, sep]
        )

        submit.click(fn=embedding_retriever,
                     inputs=[query, text, sep],
                     outputs=[result])
    demo.launch()
