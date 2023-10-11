import typing
from pathlib import Path

class model_load(object):
    
    def __init__(self):
        self.model = self.__load_sklearn_model()
        self.vectorizer = self.__load_tfidf_vectorizer()

    def __load_sklearn_model(self):
        
        """""
        load sklearn model from path
        
        """

        path = 'models/sklearn/modelweb.pkl'
        import pickle
        with open(path, "rb") as file:
            return pickle.load(file)
        
    def __load_tfidf_vectorizer(self):
        
        path = 'models/sklearn/tfidf_vectorizer.pkl'
        import pickle

        with open(path, "rb") as file:
            return pickle.load(file)
        
    def predict(self,data):
        tfidf_vectors = self.vectorizer.transform(data)
        prediction = self.model.predict(tfidf_vectors)
        modelo = model_load()
        return(modelo.clases(prediction[0]))
    
    def clases(self,data):
        clases = ['Adult', 'Business/Corporate', 'Computers and Technology',
       'E-Commerce', 'Education', 'Food', 'Forums', 'Games',
       'Health and Fitness', 'Law and Government', 'News', 'Photography',
       'Social Networking and Messaging', 'Sports', 'Streaming Services',
       'Travel']
        return clases[data]
    