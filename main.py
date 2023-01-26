from flask import Flask,request,render_template
from  Project_app.utils import Diabetes

app = Flask(__name__)


@app.route("/")
def base():
    return render_template("home.html")

@app.route("/predict", methods=["post"])
def home():
    data = dict(request.form)
    diabetes = Diabetes(data)
    prediction = diabetes.getPredict()
    return render_template("result.html",res = prediction)



if __name__ == "__main__":
    app.run(debug=True)
