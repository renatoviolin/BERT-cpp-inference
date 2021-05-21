# %% ---------------------------------------------
import requests
import json
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiohttp
import time
import numpy as np
from transformers import AutoTokenizer
import numpy as np
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# %% ---------------------------------------------
n_request = 10
URL = "http://localhost:8000/predict"
text = "[CLS] In deep [MASK], each level learns to transform its input data into a slightly more abstract and composite representation. In an image recognition application, the raw input may be a matrix of pixels; the first representational layer may abstract the pixels and encode edges; the second layer may compose and encode arrangements of edges; the third layer may encode a nose and eyes; and the fourth layer may recognize that the image contains a face. Importantly, a deep learning process can learn which features to optimally place in which level on its own. (Of course, this does not completely eliminate the need for hand-tuning; for example, varying numbers of layers and layer sizes can provide different degrees of abstraction.) Simpler models that use task-specific handcrafted features such as Gabor filters and support vector machines (SVMs) were a popular choice in the 1990s and 2000s, because of artificial neural network's (ANN) computational cost and a lack of understanding of how the brain wires its biological networks. Generally speaking, deep learning is a machine learning method that takes in an input X, and uses it to predict an output of Y. As an example, given the stock prices of the past week as input, my deep learning algorithm will try to predict the stock price of the next day. Given a large dataset of input and output pairs, a deep learning algorithm will try to minimize the difference between its prediction and expected output. By doing this, it tries to learn the association/pattern between given inputs and outputs — this in turn allows a deep learning model to generalize to inputs that it hasn’t seen before. As another example, let’s say that inputs are images of dogs and cats, and outputs are labels for those images (i.e. is the input picture a dog or a cat). If an input has a label of a dog, but the deep learning algorithm predicts a cat, then my deep learning algorithm will learn that the features of my given image (e.g. sharp teeth, facial features) are going to be associated with a dog. I won’t go too in depth into the math, but information is passed between network layers through the function shown above. The major points to keep note of here are the tunable weight and bias parameters — represented by w and b respectively in the function above. These are essential to the actual “learning” process of a deep learning algorithm. After the neural network passes its inputs all the way to its outputs.[SEP]"


# %% ---------------------------------------------
async def _async_fetch(session, data):
    async with session.post(URL, json=data) as response:
        r = await response.text()
        return r


async def run():
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    index_str = ','.join([str(s) for s in indexed_tokens])
    async with aiohttp.ClientSession() as session:
        tasks = [_async_fetch(session, data={"tokens": index_str,
                                             "idx_to_predict": "3",
                                             "request_id": str(i)}) for i in range(n_request)]
        # tasks = [_async_fetch(session, data={"tokens": indexed_tokens,
        #                                      "idx_to_predict": "3",
        #                                      "request_id": str(i)}) for i in range(n_request)]
        results = await asyncio.gather(*tasks)
    return results


# %% ---------------------------------------------
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    times = []
    print(f'Starting {n_request} request asynchronous')
    for i in range(1, 10, 1):
        start = time.perf_counter()
        loop.run_until_complete(run())
        end = time.perf_counter()
        elapsed = end - start
        times.append(elapsed)
        print(f'Run {i}: {elapsed:.5f} seconds')

    loop.close()
    print(f'Mean: {np.mean(times):.5f} seconds')
