import numpy as np
from catboost import Pool
from flask import Flask
from flask_restful import Api, Resource

from main import Model

app = Flask(__name__)
api = Api()
class Main(Resource):
    model = None
    def __init__(self):
        model = Model()
        train, test = model.loadData()
        train_dataset, eval_dataset, train_y = Model.prepare_data(train, test)
        model.train(train_dataset, eval_dataset)
        y_pred = model.predict(eval_dataset)

        from sklearn.metrics import confusion_matrix
        CM = confusion_matrix(y_pred, train_y)
        print(CM)

    def get(self):
        return {"result":"seccesful"}
api.add_resource(Main,"/api/main")
api.init_app(app)





if __name__ == "__main__":
    app.run(host="127.0.0.1",port = 3000, debug=True)