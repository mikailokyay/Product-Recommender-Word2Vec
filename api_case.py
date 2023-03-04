"""Main class for PCR Service"""

import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)
MODEL_PATH = "model/word2vec-model.pkl"


@app.route('/product_recommendation', methods=['POST'])
def product_recommendation():
    """
    Product recommendation function
    :return: json
        output body
    """
    product = request.get_json(force=True)
    failure_message = {
        "status": "Failure",
        "failure_reason": "wrong product id or product count"
    }

    try:
        if product["product_id"] != "" and product["product_count"] > 0:
            product_id = product["product_id"]
            product_count = product["product_count"]
            output = model.wv.similar_by_vector(model.wv[product_id], topn=product_count + 1)[1:]
        else:
            output = failure_message
        return str(output)
    except KeyError as key_err:
        failure_message["failure_reason"] = str(key_err)

    return jsonify(failure_message)


if __name__ == '__main__':
    with open(MODEL_PATH, 'rb') as mf:
        # load model
        model = pickle.load(mf)

    app.run(debug=True, host='0.0.0.0', port=5000)
