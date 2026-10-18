from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from triage import config


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load(config.MODEL_PATH)
    app.state.pipeline = bundle["pipeline"]
    app.state.features = bundle["features"]
    app.state.model_version = bundle["model_version"]
    yield


app = FastAPI(title="Diabetes triage service", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": app.state.model_version}


@app.post("/predict")
def predict(patient: PatientFeatures):
    row = pd.DataFrame([patient.model_dump()])[app.state.features]
    prediction = float(app.state.pipeline.predict(row)[0])
    return {"prediction": prediction}
import os
