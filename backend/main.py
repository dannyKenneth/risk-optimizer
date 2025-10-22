from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Data model for portfolio input
class PortfolioData(BaseModel):
    weights: list[float]
    returns: list[float]
    covariance: list[list[float]]

# Endpoint for portfolio optimization
@app.post("/optimize")
def optimize_portfolio(data: PortfolioData):
    weights = np.array(data.weights)
    returns = np.array(data.returns)
    covariance = np.array(data.covariance)

    expected_return = np.dot(weights, returns)
    risk = np.sqrt(np.dot(weights, np.dot(covariance, weights)))

    return {"expected_return": expected_return, "risk": risk}
