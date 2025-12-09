# test_client.py

import requests
import json
import os
import pandas as pd
from loguru import logger
import io

# --- КОНСТАНТЫ ДЛЯ ТЕСТИРОВАНИЯ ---
API_URL = "http://127.0.0.1:8000/predict"
SAMPLE_CSV_FILE = "heart_test.csv" 
OUTPUT_SUBMISSION_FILE = "student_submission.csv" # Файл для сдачи

def test_api():
    """
    Отправляет POST-запрос с CSV файлом к API, получает ответ и сохраняет его в требуемом формате.
    """
    # Проверяем, что файл существует
    if not os.path.exists(SAMPLE_CSV_FILE):
        logger.error(f"❌ Критическая ошибка: Тестовый файл '{SAMPLE_CSV_FILE}' не найден в директории.")
        logger.info("Убедитесь, что вы загрузили heart_test.csv в папку проекта.")
        return

    logger.info(f"Отправка POST-запроса на {API_URL} с файлом {SAMPLE_CSV_FILE}")
    
    try:
        with open(SAMPLE_CSV_FILE, 'rb') as f:
            # Отправляем файл на сервер
            files = {'file': (SAMPLE_CSV_FILE, f, 'text/csv')}
            response = requests.post(API_URL, files=files)
            
        logger.info(f"Получен ответ. Статус: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            
            if data and 'predictions' in data:
                results_list = data['predictions']
                
                # 1. Создаем DataFrame из JSON-ответа
                result_df = pd.DataFrame(results_list)

                # 2. Форматируем для сдачи: только 'id' и 'prediction'
                submission_df = result_df[['id', 'Prediction']].rename(columns={'Prediction': 'prediction'})
                
                # 3. Сохраняем в CSV, готовый для test.py
                submission_df.to_csv(OUTPUT_SUBMISSION_FILE, index=False)
                
                logger.success(f"✅ Результаты предсказания сохранены в файл: {OUTPUT_SUBMISSION_FILE}")
                logger.info(f"Количество предсказаний: {len(submission_df)}")
                
                # Вывод первых 5 результатов в консоль
                print("\n" + "="*50)
                print(" ПЕРВЫЕ 5 РЕЗУЛЬТАТОВ ПРЕДСКАЗАНИЯ ")
                print("="*50)
                print(submission_df.head(5).set_index('id').to_string())
                print("="*50)
            else:
                logger.warning("Ответ не содержит поля 'predictions' или пуст.")
                
        else:
            # Вывод деталей ошибки, если статус не 200
            error_details = response.json().get('error', response.text)
            logger.error(f"❌ Ошибка API. Сервер вернул ошибку: {error_details}")

    except requests.exceptions.ConnectionError:
        logger.error(f"❌ Ошибка подключения: Убедитесь, что сервер FastAPI запущен по адресу {API_URL}")
    except Exception as e:
        logger.error(f"❌ Произошла непредвиденная ошибка: {e}", exc_info=True)


if __name__ == "__main__":
    test_api()