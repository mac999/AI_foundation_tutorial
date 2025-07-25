# 필요한 라이브러리 설치
# !pip install gensim

import gensim
from gensim.models import Word2Vec
import numpy as np

# Example text data
sentences = [
    "I feel good today",
    "The weather is clear and warm today",
    "I ate kimchi for lunch",
    "Exercising makes me feel better"
]

# Preprocess text data (split words by spaces)
sentences = [sentence.split() for sentence in sentences]

# Train Word2Vec model
model = Word2Vec(sentences, vector_size=10, window=3, min_count=1, sg=0)

# Check the vector for a specific word
vector = model.wv['feel']
print("Vector for the word 'feel':", vector)

# Find similar words
similar_words = model.wv.most_similar('feel', topn=3)
print("Words similar to 'feel':", similar_words)