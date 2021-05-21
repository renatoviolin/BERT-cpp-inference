# %% ---------------------------------------------
from fastapi import FastAPI, Request
import onnxruntime
import numpy as np
import logging
logger = logging.getLogger('hypercorn.error')


# %% ---------------------------------------------
app = FastAPI()
onnx_session = onnxruntime.InferenceSession("../model/bert_onnx.pt")


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# %% ---------------------------------------------
def inference(tokens, idx_to_predict):

    tokens = tokens + [0] * (512 - len(tokens))
    tokens = np.array(tokens).reshape(1, -1)

    ort_inputs = {onnx_session.get_inputs()[0].name: tokens}
    ort_outs = onnx_session.run(None, ort_inputs)
    output = np.array(ort_outs)[0][0]
    output = softmax(output[idx_to_predict])

    idx = np.argpartition(output, -5)[-5:]
    prob = output[idx]
    result = {}
    result["Prediction"] = [int(i) for i in idx]
    result["Confidence"] = [float(p) for p in prob]

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
