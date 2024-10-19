import nltk
nltk.download('brown')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

import pandas as pd
import numpy as np
import re
import string
from tqdm import tqdm
import os
from IPython.display import FileLink
import matplotlib.pyplot as plt

from nltk.corpus import stopwords, brown
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize
from natasha import MorphVocab, Doc, Segmenter, NewsMorphTagger, NewsEmbedding
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict, Counter
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from wordcloud import WordCloud
import time

brown_freq = Counter(brown.words())

segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()

# Инициализация модели и токенизатора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("ai-forever/ru-en-RoSBERTa")
model = AutoModel.from_pretrained("ai-forever/ru-en-RoSBERTa").to(device)

def remove_punct(word):
    """Удаление пунктуации из текста."""
    word = "".join([char for char in word if char.isalpha() or char == ' '])
    return "".join([char for char in word if char not in string.punctuation])

def remove_stopwords(sentence: str) -> str:
    """Удаление стоп-слов и токенизация предложения."""
    stop_words = get_stop_words('ru')
    stop_words.extend(['сожаление', 'чтото'])
    words = word_tokenize(sentence)
    words = [remove_punct(word) for word in words]
    filtered_sentence = [word for word in words if word not in stop_words and len(word) >= 5]
    return ' '.join(filtered_sentence) if len(filtered_sentence) >= 2 else ''

def lemmatize(sentence: str) -> str:
    """Лемматизация текста."""
    lemmatized = ''
    doc = Doc(sentence)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        lemmatized += token.lemma + ' '
    return lemmatized.strip()

def count_words_phrases(lemmatized_texts):
    """Подсчет количества слов и фраз в лемматизированных текстах."""
    word_count = Counter(lemmatized_texts)
    return dict(word_count)

def get_word_embedding(word: str):
    """Получение эмбеддинга для слова."""
    inputs = tokenizer("clustering: " + word, return_tensors="pt", max_length=512, padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0]

def normalize_embeddings(embedding):
    """Нормализация эмбеддингов."""
    return F.normalize(embedding, p=2, dim=1)

def get_sentence_embedding(sentence: str):
    """Получение эмбеддинга для предложения."""
    sentence_vector = normalize_embeddings(get_word_embedding(sentence)).cpu().numpy().reshape(-1)
    return sentence_vector

def get_leaving_reasons(text):
    """Получение причин ухода, с лемматизацией и предобработкой."""
    text = text.lower()
    text = re.split(r'[.,\n]+', text)
    text = [remove_punct(word) for word in text]
    pattern = r' ([абвиоу]) |(?:\sили\s)'
    split_by_chars = [re.split(pattern, text0) for text0 in text]
    split_by_chars = [part.strip() for sentence in split_by_chars for part in sentence if part is not None]
    lemmatized_texts = [remove_stopwords(lemmatize(sentence)) for sentence in split_by_chars if remove_stopwords(lemmatize(sentence)) != '']
    
    return lemmatized_texts

def select_most_frequent_word(cluster_words, word_freq_dict):
    """Выбор самого частого слова в кластере."""
    sorted_words = sorted(cluster_words, key=lambda word: (-word_freq_dict[word], -brown_freq.get(word, 0)))
    return sorted_words[0]

def get_embeddings(data):
    """Получение эмбеддингов для текстов."""
    lemmatized_texts = [get_leaving_reasons(text) for text in data]
    lemmatized_texts = [sentence for text in lemmatized_texts for sentence in text]
        
    word_freq_dict = count_words_phrases(lemmatized_texts)
    
    embeddings = []
    for sentence in tqdm(word_freq_dict.keys()):
        embeddings.append(get_sentence_embedding(sentence))
    
    embeddings = np.array(embeddings)
    return embeddings, word_freq_dict

def clustering(data):
    """Кластеризация слов по эмбеддингам."""
    agglomerative = AgglomerativeClustering(n_clusters=None, distance_threshold=2)
    embeddings, word_freq_dict = get_embeddings(data)
    labels = agglomerative.fit_predict(embeddings)

    word_clusters = defaultdict(list)
    for word, label in zip(word_freq_dict.keys(), labels):
        word_clusters[label].append(word)
    return word_clusters, word_freq_dict

def get_lemmatized_texts_with_mapping(texts):
    """
    Лемматизация текстов и сохранение исходных предложений в словарь для последующего восстановления.
    """
    
    lemmatized_texts = []
    lemmatize_dict = {}

    for sentence in texts:
        sentence = sentence.lower()
        sentence = re.split(r'[.,\n]+', sentence)
        pattern = r' ([абвиоу]) |(?:\sили\s)'
        split_by_chars = [re.split(pattern, text0) for text0 in sentence]
        split_by_chars = [part.strip() for sentence_ in split_by_chars for part in sentence_ if part is not None and len(part) >= 2] 
        for sentence in split_by_chars:
            lemmatized_sentence = remove_stopwords(lemmatize(sentence))
            if len(lemmatized_sentence) > 0:
                lemmatize_dict[lemmatized_sentence] = sentence  # Сохраняем оригинальное предложение
                lemmatized_texts.append(lemmatized_sentence)  # Сохраняем лемматизированное предложение
    return lemmatized_texts, lemmatize_dict

def form_final_word_dict_with_restore(data):
    """
    Формирование итогового словаря и восстановление исходных фраз.
    """
    final_word_dict = {}
    lemmatized_texts, lemmatize_dict = get_lemmatized_texts_with_mapping(data)

    # Кластеризация
    word_clusters, word_freq_dict = clustering(lemmatized_texts)
    
    for label, cluster_words in word_clusters.items():
        # Выбираем самое частое слово в кластере
        most_frequent_word = select_most_frequent_word(cluster_words, word_freq_dict)
        
        # Суммируем частоты встречаемости всех слов в кластере
        cluster_word_count_sum = sum(word_freq_dict[word] for word in cluster_words)
        
        # Восстанавливаем оригинальные фразы для наиболее частого слова
        original_sentence = lemmatize_dict.get(most_frequent_word, most_frequent_word)
        
        # Добавляем в итоговый словарь: ключ — исходная фраза, значение — сумма частот всех слов в кластере
        final_word_dict[original_sentence] = cluster_word_count_sum
    
    return final_word_dict, lemmatize_dict

def lemma_replacement(lemma, lemmatize_dict):
    """Функция для замены лемматизированных фраз обратно."""
    if lemma in lemmatize_dict.keys():
        return lemmatize_dict[lemma]
    return lemma

def get_multi_leaving_reasons(data):
    """Применение функции к ключам словаря."""
    final_word_dict, lemmatize_dict = form_final_word_dict_with_restore(data)
    d, func = final_word_dict, lemma_replacement
    return {func(k, lemmatize_dict): v for k, v in d.items()}

def generate_word_cloud(word_dict):
    """Генерация облака слов."""
    wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate_from_frequencies(word_dict)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def plot_word_histogram(word_dict: dict) -> None:
    """Построение гистограммы частот слов."""
    sorted_word_dict = dict(sorted(word_dict.items(), key=lambda item: item[1], reverse=True))
    
    words = list(sorted_word_dict.keys())
    frequencies = list(sorted_word_dict.values())

    plt.figure(figsize=(10, 10))
    plt.bar(words, frequencies, color='skyblue')
    plt.ylabel('Частота')
    plt.xlabel('Слова / фразы')
    plt.title('Распределение частот слов и фраз')
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    plt.show()