# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: embedding_server.py
# @time: 2024/1/5 11:03
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')


class Sentence(BaseModel):
    text: str


@app.get('/')
def home():
    return 'hello world'


@app.post('/embedding')
def get_embedding(sentence: Sentence):
    embedding = model.encode(sentence.text, normalize_embeddings=True).tolist()
    return {"text": sentence.text, "embedding": embedding}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=50072)
