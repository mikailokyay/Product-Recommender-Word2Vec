"""
Product Recommendation System with Word2Vec
"""
import os
import pickle
import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm


class ProductRecommender:
    """
    Product Recommender class
    """

    def __init__(self, data_file, model_path, model_name):

        self.dataset = pd.read_csv(data_file, encoding="utf8").sort_values(by='timestamp', ascending=False)
        self.cleaned_data = self.dataset
        if len(self.find_all(model_path, model_name)) < 1:
            self.cleaned_data = self.get_cleaned_dataset(self.dataset)
            self.products_train = self.get_train_data()

    @staticmethod
    def find_all(path, name):
        """
        Check file in directory
        :param path: str
            path of file
        :param name:str
            name of file
        :return:list
            It is returns file in path as list
        """
        result = []
        for root, _, files in os.walk(path):
            if name in files:
                result.append(os.path.join(root, name))
        return result

    @staticmethod
    def get_cleaned_dataset(data):
        """
        This function is used for data cleaning and user based and product based ratings columns adding
        :param data: DataFrame
            Whole data which is read from csv file
        :return: DataFrame
            Cleaned and repetitive_data_count column added
        """
        # Delete rows with no client (if there is such a case)
        data = data.dropna(axis=0, subset=['client'])

        # Drop duplicate data which has same client, timestamp and product_id
        data = data[['client', 'timestamp', 'product_id']].drop_duplicates()

        new_dataset = data.copy()
        ex_client = -1
        ex_product_id = -1
        for idx, row in tqdm(data.iterrows()):
            if row["client"] == ex_client and \
                    row["product_id"] == ex_product_id:
                new_dataset.loc[idx, "repetitive_data_count"] = new_dataset["repetitive_data_count"][idx - 1] + 1
                new_dataset.drop(idx - 1, inplace=True)
            else:
                new_dataset.loc[idx, "repetitive_data_count"] = 1
            ex_client = row["client"]
            ex_product_id = row["product_id"]
        return new_dataset

    def get_train_data(self):
        """
        Word2Vec train data collection function
        :return: list of list
            client based products
        """
        clients = self.cleaned_data["client"].unique().tolist()
        train_data = self.cleaned_data[self.cleaned_data['client'].isin(clients)]

        # list to capture purchase history of the customers
        products_train = []
        # populate the list with the product codes
        for i in tqdm(clients):
            temp = train_data[train_data["client"] == i]["product_id"].tolist()
            products_train.append(temp)
        return products_train

    def train(self):
        """
        Model training function
        :return: word2vec model
            Returns trained word2vec model
        """
        # train word2vec model
        model = Word2Vec(window=10, sg=1, hs=0,
                         negative=5,  # for negative sampling
                         alpha=0.04, min_alpha=0.0007,
                         seed=14)

        model.build_vocab(self.products_train, progress_per=100)

        model.train(self.products_train, total_examples=model.corpus_count,
                    epochs=10, report_delay=1)

        return model

    @staticmethod
    def word2vec_to_pickle(model, model_file):
        """
        This function is used for save trained word2vec model as pickle
        :param model:string
            file name which will be saved as pkl
        :param model_file: string
            model file name with path and name
        """
        with open(model_file, 'wb') as file:
            pickle.dump(model, file)

    @staticmethod
    def load_model(model_file_name):
        """
        Load pickle model
        :param model_file_name:string
            Saved model file name
        :return:loaded model
        """
        with open(model_file_name, 'rb') as file:
            # load model
            pickle_model = pickle.load(file)
        return pickle_model

    @staticmethod
    def similar_products(model, product, rec_count=15):
        """
        This function is used for product recommendation for a product with its id
        :param model: word2vec model
        :param rec_count:int
            similar product count
        :param product: string
            Product id
        :return:list of tuple
            Product recommendations and scores in list of tuples
        """
        # extract most similar products for the input vector
        model_output = model.wv.similar_by_vector(model.wv[product], topn=rec_count + 1)[1:]
        return model_output


if __name__ == '__main__':
    MODEL_PATH = "../model/"
    MODEL_NAME = "word2vec-model.pkl"
    PCR = ProductRecommender("../data/product_data.csv", MODEL_PATH, MODEL_NAME)
    if len(PCR.find_all(MODEL_PATH, MODEL_NAME)) < 1:
        word2vec_model = PCR.train()
        PCR.word2vec_to_pickle(word2vec_model, MODEL_PATH+MODEL_NAME)
    else:
        word2vec_model = PCR.load_model(MODEL_PATH+MODEL_NAME)

    # test with "32322582" product_id
    output = PCR.similar_products(word2vec_model, "32322582", 15)
    print(output)
    print(word2vec_model.corpus_total_words)
