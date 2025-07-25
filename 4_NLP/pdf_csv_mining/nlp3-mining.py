# Import other libraries
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbs

# Options for plots
plt.rcParams['figure.figsize'] = (10, 6)
sbs.set('paper')

# Import litstudy
path = os.path.abspath(os.path.join('..'))
if path not in sys.path:
    sys.path.append(path)

import litstudy

# Load the CSV files. 데이터셋은 여기 링크에서 다운로드. 
docs1 = litstudy.load_ieee_csv('data/ieee_1.csv')
docs2 = litstudy.load_ieee_csv('data/ieee_2.csv')
docs3 = litstudy.load_ieee_csv('data/ieee_3.csv')
docs4 = litstudy.load_ieee_csv('data/ieee_4.csv')
docs5 = litstudy.load_ieee_csv('data/ieee_5.csv')
docs_ieee = docs1 | docs2 | docs3 | docs4 | docs5
print(len(docs_ieee), 'papers loaded from IEEE')

docs_springer = litstudy.load_springer_csv('data/springer.csv')
print(len(docs_springer), 'papers loaded from Springer')

# IEEE, springer 문헌 데이터 병합
docs_csv = docs_ieee | docs_springer
print(len(docs_csv), 'papers loaded from CSV')

docs_exclude = litstudy.load_ris_file('data/exclude.ris')
docs_remaining = docs_csv - docs_exclude   # 특정 문헌 제외

print(len(docs_exclude), 'papers were excluded')
print(len(docs_remaining), 'paper remaining')

# 로그 라이브러리 임포트. scopus 논문 검색. 미리, open api key를 https://dev.elsevier.com/apikey/manage 획득해야 함. 
import logging
try:
    import pybliometrics
    pybliometrics.init()    
    logging.getLogger().setLevel(logging.CRITICAL)
    docs_scopus, docs_notfound = litstudy.refine_scopus(docs_remaining)
except Exception as e: # 에러 발생 가능. https://pybliometrics.readthedocs.io/en/stable/access/errors.html
   print(e)
   exit()

print(len(docs_scopus), 'papers found on Scopus')
print(len(docs_notfound), 'papers were not found and were discarded')

# 특정 년도 논문 필터링
docs = docs_scopus.filter_docs(lambda d: d.publication_year >= 2000)
print(len(docs), 'papers remaining')

# 년도, 저자, 언어 등 히스토그램 차트 출력
litstudy.plot_year_histogram(docs, vertical=True);
litstudy.plot_affiliation_histogram(docs, limit=15);
litstudy.plot_author_histogram(docs);
litstudy.plot_language_histogram(docs);
litstudy.plot_number_authors_histogram(docs);

# 약어 논문 맵핑 처리. 
mapping = {
    "IEEE International parallel and distributed processing symposium IPDPS": "IEEE IPDPS",
    "IEEE International parallel and distributed processing symposium workshops IPDPSW": "IEEE IPDPS Workshops",
}
litstudy.plot_source_histogram(docs, mapper=mapping, limit=15);  # 출처 차트
litstudy.plot_cocitation_network(docs, max_edges=500)  # 인용 네트워크 차트

# 말뭉치(corpus) 분석. 
corpus = litstudy.build_corpus(docs, ngram_threshold=0.8)  # 말뭉치(corpus) 생성
litstudy.compute_word_distribution(corpus).filter(like='_', axis=0).sort_index()  # 단어 빈도수 계산
plt.figure(figsize=(20, 3))
litstudy.plot_word_distribution(corpus, limit=50, title="Top words", vertical=True, label_rotation=45);  # 단어 빈도수 출력

# 토픽 간 유사도 거리 클러스터링을 위한 NMF(Non-negative Matrix Factorization) 분석 (여기 참고)
num_topics = 15
topic_model = litstudy.train_nmf_model(corpus, num_topics, max_iter=250)
for i in range(num_topics):
    print(f'Topic {i+1}:', topic_model.best_tokens_for_topic(i))

plt.figure(figsize=(15, 5))
litstudy.plot_topic_clouds(topic_model, ncols=5);

plt.figure(figsize=(20, 20))
litstudy.plot_embedding(corpus, topic_model);
