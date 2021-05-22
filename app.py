from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

classifier = pickle.load(open('classifier.pkl','rb'))

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict',methods=['POST'])
def predict():
    f_acidity = request.form['f_acidity']
    v_acidity = request.form['v_acidity']
    citric = request.form['citric']
    residual_sugar = request.form['residual_sugar']
    chlorides = request.form['chlorides']
    total_sulfur = request.form['total_sulfur']
    density = request.form['density']
    ph = request.form['ph']
    sulphates = request.form['sulphates']
    alcohol = request.form['alcohol']

    inpu = [[f_acidity,v_acidity,citric,residual_sugar,chlorides,total_sulfur,density,ph,sulphates,alcohol]]
    x = pd.DataFrame(inpu, columns=["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides",
                                    "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"])
    result = classifier.predict(x)

    return render_template('index.html', x=('Wine Quality is {}'.format(result)))

if __name__ == "__main__":
    app.run(debug=True)