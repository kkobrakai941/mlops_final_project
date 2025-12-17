from fastapi import FastAPI, Response
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import os
import psycopg2
import pickle
from prometheus_client import Counter, Histogram, generate_latest

app = FastAPI()

class Features(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

MODEL_PATH = os.getenv("MODEL_PATH", "/models/model.pkl")

iris = load_iris()
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = LogisticRegression(max_iter=200)
    model.fit(iris.data, iris.target)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_NAME = os.getenv("DB_NAME", "mlops")

def get_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME
    )
    return conn

with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS predictions (id SERIAL PRIMARY KEY, feature1 FLOAT, feature2 FLOAT, feature3 FLOAT, feature4 FLOAT, prediction INT)"
        )
        conn.commit()

prediction_counter = Counter("prediction_count", "\u041a\u043e\u043b\u0438\u0447\u0435\u0441\u0442\u0432\u043e \u043f\u0440\u0435\u0434\u0441\u043a\u0430\u0437\u0430\u043d\u0438\u0439")
prediction_latency = Histogram("prediction_latency_seconds", "\u0412\u0440\u0435\u043c\u044f \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0438 \u0437\u0430\u043f\u0440\u043e\u0441\u0430")

@app.post("/predict")
def predict(features: Features):
    with prediction_latency.time():
        data = [[features.feature1, features.feature2, features.feature3, features.feature4]]
        pred = int(model.predict(data)[0])
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO predictions (feature1, feature2, feature3, feature4, prediction) VALUES (%s, %s, %s, %s, %s)",
                    (features.feature1, features.feature2, features.feature3, features.feature4, pred)
                )
                conn.commit()
        prediction_counter.inc()
        return {"prediction": pred}

@app.get("/reload_model")
def reload_model():
    global model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return {"status": "model reloaded"}
    else:
        return {"status": "model file not found"}

@app.get("/metrics")
def metrics():
    content = generate_latest()
    return Response(content=content, media_type="text/plain; version=0.0.4")
