import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# load the model from disk

# REg
model = pickle.load(open('model.pkl', 'rb'))

# Classification
clf = pickle.load(open('nlp_model.pkl', 'rb'))
cv = pickle.load(open('tranform.pkl', 'rb'))
svm = pickle.load(open('svm.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/email')
def email():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    print(request.form['test_score'])
    format = request.args.get('format')
    print("here", format)
    print("new line")

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    if(format == 'json'):
        return jsonify({'salary': output})

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

@app.route('/svm_model' , methods=['POST'])
def svm_model():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = svm.predict(final_features)
    output = prediction[0]
    if(format == 'json'):
        return jsonify({'y': output})

    return render_template('svm.html', prediction_text='y= {}'.format(output))    


@app.route('/predict_email', methods=['POST'])
def predict_email():

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return my_prediction


if __name__ == "__main__":
    app.run(debug=True)  # auto-reload on code change
