"""Предобработка текста для инференса.

Полностью повторяет логику очистки из обучающего ноутбука
(Project_Фамилия.ipynb, ячейки с определением стоп-слов и функцией clean),
чтобы вход модели в API совпадал с тем, на чём обучался TF-IDF-векторизатор.
"""

import re
import string

import nltk
from nltk.corpus import stopwords


def _load_stopwords() -> set:
    """Загружает стоп-слова NLTK, при необходимости скачивая их."""
    try:
        base = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        base = set(stopwords.words("english"))
    return base


# Кастомные стоп-слова из ноутбука (часто встречаются в обоих классах и не несут тональности)
CUSTOM_STOPWORDS = {
    "film", "movie", "one", "would", "could",
    "really", "even", "get", "make", "see",
    "also", "first", "people", "much", "like",
}

STOP_WORDS = _load_stopwords().union(CUSTOM_STOPWORDS)

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean(text: str) -> str:
    """Очищает текст так же, как при обучении модели.

    Шаги: нижний регистр, удаление HTML-тегов, удаление ссылок,
    удаление пунктуации, удаление стоп-слов.
    """
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = text.translate(_PUNCT_TABLE)
    text = " ".join(w for w in text.split() if w not in STOP_WORDS)
    return text
