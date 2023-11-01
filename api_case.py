"""Main class for PCR Service"""

import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()
MODEL_PATH = "/model/word2vec-model.pkl"


class ScrapperItem(BaseModel):
    task: str
    product_id: str
    product_count: int


@app.post('/product_recommendation')
def product_recommendation(resp_json: ScrapperItem):
    """
    Planet scrapping function
    :return: json
        output body
    """
    failure_message = {
        "status": "Failure",
        "failure_reason": "Unknown customer or url"
    }

    try:
        if resp_json.product_id != "" and resp_json.product_count > 0:
            product_id = resp_json.product_id
            product_count = resp_json.product_count
            output = model.wv.similar_by_vector(model.wv[product_id], topn=product_count + 1)[1:]

        else:
            output = failure_message
        return JSONResponse(content=jsonable_encoder(output))
    except KeyError as key_err:
        failure_message["failure_reason"] = str(key_err)

    return JSONResponse(content=jsonable_encoder(failure_message))


if __name__ == '__main__':
    with open(MODEL_PATH, 'rb') as mf:
        # load model
        model = pickle.load(mf)

    uvicorn.run(app, host='0.0.0.0', port=5000)
