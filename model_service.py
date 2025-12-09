# model_service.py 
import pandas as pd
import numpy as np
import joblib 
from catboost import CatBoostClassifier 
from loguru import logger
from typing import List, Tuple

# --- КОНСТАНТЫ ИЗ ОБУЧЕННОЙ МОДЕЛИ ---

MODEL_PATH: str = 'catboost_best_model.pkl'

# 13 признаков, на которых обучалась финальная модель
SELECTED_FEATURES: List[str] = [
    'Age', 'Cholesterol', 'Heart rate', 'Obesity', 'Exercise Hours Per Week',
    'Previous Heart Problems', 'Medication Use', 'Sedentary Hours Per Day',
    'Income', 'BMI', 'Triglycerides', 'Systolic blood pressure',
    'Diastolic blood pressure'
]

# Категориальные признаки, которые CatBoost ожидает увидеть строками
CATEGORICAL_FEATURES: List[str] = [
    'Obesity', 'Previous Heart Problems', 'Medication Use'
]

# Оптимальный порог для бинаризации предсказаний 
OPTIMAL_THRESHOLD: float = 0.51 


class HeartRiskPredictor:
    """
    Класс для загрузки модели CatBoost и выполнения предсказаний.
    """
    def __init__(self):
        self.model: CatBoostClassifier = None
        self._load_model()

    def _load_model(self):
        """
        Загружает обученную модель из .pkl файла с помощью joblib.
        """
        try:
            # Используем joblib.load() для загрузки .pkl файла
            self.model = joblib.load(MODEL_PATH)
            logger.info(f"Модель успешно загружена с помощью joblib: {MODEL_PATH}")
        except Exception as e:
            # Критическая ошибка: без модели API не работает
            logger.error(f"❌ Ошибка при загрузке модели из {MODEL_PATH}. Убедитесь, что файл существует.", exc_info=True)
            raise RuntimeError(f"Не удалось загрузить модель: {MODEL_PATH}")

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Выполняет предобработку входных данных: проверку признаков и преобразование типов.
        """
        # Проверка наличия всех необходимых признаков
        missing_cols = [col for col in SELECTED_FEATURES if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют необходимые столбцы: {', '.join(missing_cols)}. Пожалуйста, проверьте файл.")

        # Выбор признаков
        X_proc = df[SELECTED_FEATURES].copy()

        # Преобразование категориальных признаков в строку
        for col in CATEGORICAL_FEATURES:
            if col in X_proc.columns:
                X_proc[col] = X_proc[col].astype(str)

        logger.debug(f"Данные предобработаны. Форма: {X_proc.shape}")
        return X_proc

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Выполняет предсказание риска и бинаризацию результата.
        """
        X_proc = self._preprocess(data)

        # Получаем вероятности класса 1
        probas = self.model.predict_proba(X_proc)[:, 1]
        logger.debug("Вероятности предсказаны.")

        # Применяем оптимальный порог 0.51
        predictions = (probas >= OPTIMAL_THRESHOLD).astype(int)

        # Формируем DataFrame с результатами
        result_df = pd.DataFrame({
            'Heart Attack Risk (Probability)': probas,
            'Prediction': predictions
        })

        # Если в исходных данных был столбец 'id', делаем его индексом
        if 'id' in data.columns:
            result_df.index = data['id']
            result_df.index.name = 'id'
            
        return result_df