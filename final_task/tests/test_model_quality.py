# tests/test_model_quality.py

import pytest
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# 1. Параметры порогов качества
# -------------------------------
# Подкорректируйте эти значения под ваши требования.
CLEAN_F1_THRESHOLD = 0.90
NOISY_F1_THRESHOLD = 0.70

# -------------------------------
# 2. Фикстура: загрузка модели
# -------------------------------
@pytest.fixture(scope="session")
def bert_classifier():
    """
    Загружает предобученную модель BERT и соответствующий токенизатор.
    Возвращает tuple (classifier, tokenizer, device).
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = BertForSequenceClassification.from_pretrained("my_bert_model").to(device)
    tokenizer = BertTokenizer.from_pretrained("my_bert_model")

    return classifier, tokenizer, device

# -------------------------------
# 3. Функция-помощник: предсказание списка
# -------------------------------
def predict_batch(classifier, tokenizer, device, texts):
    """
    Принимает:
      - classifier: экземпляр BertForSequenceClassification
      - tokenizer: соответствующий BertTokenizer
      - device: CUDA или CPU
      - texts: список строк (промпты)

    Возвращает:
      - preds: список целых 0 или 1 (0=regular, 1=jailbreak)
    """
    preds = []
    classifier.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(device)

            outputs = classifier(**inputs)
            pred_label = torch.argmax(outputs.logits, dim=-1).item()
            preds.append(pred_label)
    return preds

# -------------------------------
# 4. Фикстуры: чтение датасетов
# -------------------------------
@pytest.fixture(scope="session")
def clean_data():
    """
    Читает CSV с "чистым" датасетом.
    Ожидает файл combined_prompts_balanced.csv в корне проекта.
    Возвращает tuple (texts, labels).
    """
    df = pd.read_csv("tests/combined_prompts_balanced.csv", encoding="utf-8")
    texts = df["prompt"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels

@pytest.fixture(scope="session")
def noisy_data():
    """
    Читает CSV с "зашумлённым" датасетом.
    Ожидает файл combined_prompts_balanced_noisy.csv в корне проекта.
    Возвращает tuple (texts, labels).
    """
    df = pd.read_csv("tests/combined_prompts_balanced_noisy.csv", encoding="utf-8")
    texts = df["prompt"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels

# -------------------------------
# 5. Тесты качества модели
# -------------------------------
def test_model_on_clean_data(bert_classifier, clean_data):
    """
    Проверяет F1-score модели на чистом датасете.
    Убеждается, что F1 >= CLEAN_F1_THRESHOLD.
    """
    classifier, tokenizer, device = bert_classifier
    texts, labels = clean_data

    preds = predict_batch(classifier, tokenizer, device, texts)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    # Выводим метрики в консоль, чтобы видеть во время запуска pytest
    print(f"\n=== CLEAN DATA METRICS ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    assert f1 >= CLEAN_F1_THRESHOLD, (
        f"F1-score на чистом датасете слишком низкий: {f1:.4f} < {CLEAN_F1_THRESHOLD}"
    )

def test_model_on_noisy_data(bert_classifier, noisy_data):
    """
    Проверяет F1-score модели на зашумлённом датасете.
    Убеждается, что F1 >= NOISY_F1_THRESHOLD.
    """
    classifier, tokenizer, device = bert_classifier
    texts, labels = noisy_data

    preds = predict_batch(classifier, tokenizer, device, texts)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    print(f"\n=== NOISY DATA METRICS ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    assert f1 >= NOISY_F1_THRESHOLD, (
        f"F1-score на зашумлённом датасете слишком низкий: {f1:.4f} < {NOISY_F1_THRESHOLD}"
    )
