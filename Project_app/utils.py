import config
import pickle
import json

class Diabetes:

    def __init__(self,input_data):
        self.input_data = input_data


    def load_data(self):
        with open(config.model_path,"rb") as f:
            self.model = pickle.load(f)

        with open(config.scaling_obj,"rb") as f:
            self.scaling_obj = pickle.load(f)

        with open(config.project_data_path,"r") as f:
            self.project_data = json.load(f)

    def getPredict(self):
        self.load_data()
        test_array = []
        for feature in self.project_data["features"]:
            test_array.append(float(self.input_data[feature]))

        scaled_array = self.scaling_obj.transform([test_array])

        return self.model.predict(scaled_array)[0]

        



        