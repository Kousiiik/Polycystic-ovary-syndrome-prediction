from flask import Flask, request, render_template,redirect,url_for
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/<name>")
def rande(name):
    return redirect(url_for('about'))

@app.route('/',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    if prediction == 1:
        return render_template('index.html', prediction_text='The Person has PCOS with the accuracy of 90.79%')
    else :
        return render_template('index.html',prediction_text = 'The Person does not have PCOS with the accuracy of 90.79%')
        


if __name__ == "__main__":
    app.run(debug=True)
