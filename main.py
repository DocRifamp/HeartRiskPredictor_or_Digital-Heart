# main.py

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
import pandas as pd
from io import StringIO
import sys
from loguru import logger
from model_service import HeartRiskPredictor, OPTIMAL_THRESHOLD, SELECTED_FEATURES, MODEL_PATH, CATEGORICAL_FEATURES

# --- 1. СИСТЕМА ЛОГИРОВАНИЯ ---
# Настройка loguru: вывод в консоль и в файл с ротацией
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("api_log.log", rotation="10 MB", level="DEBUG", enqueue=True)
logger.info("Запуск приложения FastAPI...")

# --- Инициализация модели ---
try:
    predictor = HeartRiskPredictor()
except RuntimeError:
    # Если модель не загрузилась, приложение завершает работу
    logger.critical("❌ Приложение остановлено из-за критической ошибки загрузки модели.")
    sys.exit(1)

# --- Инициализация FastAPI ---
app = FastAPI(
    title="Heart Attack Risk Prediction API",
    description="REST API для предсказания риска сердечного приступа (CatBoost, 13 признаков, порог 0.51).",
    version="1.0.0",
    docs_url=None, # Отключаем стандартный docs_url
    redoc_url=None # Отключаем стандартный redoc_url
)

# --- 2. ОБРАБОТКА ОШИБОК ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Обработчик кастомных HTTP исключений (4xx)."""
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Обработчик всех остальных необработанных исключений (500)."""
    logger.error(f"Непредвиденная ошибка: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Внутренняя ошибка сервера. Подробности записаны в файл лога 'api_log.log'."},
    )

# --- 3. ЭНДПОИНТ ДЛЯ ПРЕДСКАЗАНИЯ ---
@app.post("/predict",
          summary="Выполнить предсказание риска по CSV файлу",
          description="Загрузите CSV-файл. API вернет 'id' (если есть), вероятность и бинарное предсказание (0/1).")
async def predict_csv(
    file: UploadFile = File(..., description="CSV файл с тестовой выборкой (данные должны быть нормализованы).")
):
    """
    Принимает CSV файл, выполняет предсказание.
    """
    if not file.filename.endswith('.csv'):
        # 400 Bad Request
        raise HTTPException(status_code=400, detail="Неверный формат файла. Ожидается CSV.")

    logger.info(f"Получен файл: {file.filename}. Размер: {file.size} байт.")
    
    try:
        content = await file.read()
        
        # Попытка чтения CSV, включая обработку распространенных кодировок
        try:
            s = str(content, 'utf-8')
            data = pd.read_csv(StringIO(s))
        except UnicodeDecodeError:
            s = str(content, 'cp1251')
            data = pd.read_csv(StringIO(s))
        except Exception as e:
            # 400 Bad Request
            raise HTTPException(status_code=400, detail="Не удалось прочитать CSV. Проверьте кодировку или формат.") from e

    except Exception as e:
        # 500 Internal Server Error
        raise HTTPException(status_code=500, detail=f"Ошибка сервера при обработке файла: {e}")

    logger.info(f"CSV прочитан. Количество строк для предсказания: {len(data)}")

    try:
        # Выполнение предсказания
        prediction_df = predictor.predict(data)
        
    except ValueError as e:
        # Ошибка, связанная с данными (например, отсутствуют нужные столбцы) - 400 Bad Request
        raise HTTPException(status_code=400, detail=f"Ошибка в данных: {e}")
    
    # Формирование ответа в формате JSON (DataFrame -> список словарей)
    results = prediction_df.reset_index().to_dict('records')
        
    logger.info(f"Предсказание выполнено успешно. Возврат {len(results)} записей.")
    
    return {"status": "success", "predictions": results}

# --- 4. ДОКУМЕНТАЦИЯ API ---
# Включаем кастомные URL для документации (docs и redoc)
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Интерфейс Swagger UI для интерактивной документации."""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI"
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """Интерфейс ReDoc для красивой документации."""
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc"
    )

@app.get("/model_info",
          summary="Информация о модели",
          description="Возвращает параметры, использованные для предсказания.")
async def model_info():
    """
    Возвращает ключевые параметры используемой модели.
    """
    return {
        "model_file": MODEL_PATH,
        "prediction_threshold": OPTIMAL_THRESHOLD,
        "required_features": SELECTED_FEATURES,
        "categorical_features_for_catboost": CATEGORICAL_FEATURES,
        "note": "Данные для предсказания должны быть нормализованы, как в обучающей выборке."
    }