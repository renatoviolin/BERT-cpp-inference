# %% ---------------------------------------------
import torch
from fastapi import FastAPI, Request
import logging
import numpy as np
logger = logging.getLogger('hypercorn.error')


# %% ---------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load('../model/large_lm.pt')
model.eval()
model.to(device)
app = FastAPI()


# %% ---------------------------------------------
def inference(tokens, idx_to_predict):
    tokens = torch.tensor(tokens).unsqueeze(0)
    with torch.no_grad():
        output = model(tokens.to(device))[0].detach().cpu()
    output = torch.softmax(output[0][idx_to_predict], 0)
    prob, idx = output.topk(k=5, dim=0)
    result = {}
    result["Prediction"] = [int(i.numpy()) for i in idx]
    result["Confidence"] = [float(p.numpy()) for p in prob]
    return result


@app.post('/predict')
async def predict(request: Request):
    result = (await request.json())
    request_id = result.get('request_id')
    tokens = result.get('tokens')
    idx_to_predict = result.get('idx_to_predict')

    logger.info(f'Request id: {request_id}')
    out = inference(tokens, int(idx_to_predict))
    return out