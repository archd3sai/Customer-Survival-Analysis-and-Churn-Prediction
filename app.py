import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    SeniorCitizen = 0
    if 'SeniorCitizen' in request.form:
        SeniorCitizen = 1
    Partner = 0
    if 'Partner' in request.form:
        Partner = 1
    Dependents = 0
    if 'Dependents' in request.form:
        Dependents = 1
    PaperlessBilling = 0
    if 'PaperlessBilling' in request.form:
        PaperlessBilling = 1

    MonthlyCharges = int(request.form["MonthlyCharges"])
    tenure_group = int(request.form["Tenure"])

    InternetService_Fiberoptic = 0
    InternetService_No = 0
    if request.form["InternetService"] == 0:
        InternetService_No = 1
    elif request.form["InternetService"] == 2:
        InternetService_Fiberoptic = 1

    OnlineSecurity_Nointernetservice = 0
    OnlineSecurity_Yes = 0

    if request.form["OnlineSecurity"] == 2:
        OnlineSecurity_Nointernetservice = 1
    elif request.form["OnlineSecurity"] == 1:
        OnlineSecurity_Yes = 1

    OnlineBackup_Nointernetservice = 0
    OnlineBackup_Yes = 0
    if request.form["OnlineBackup"] == 2:
        OnlineBackup_Nointernetservice = 1
    elif request.form["OnlineBackup"] == 1:
        OnlineBackup_Yes = 1

    DeviceProtection_Nointernetservice = 0
    DeviceProtection_Yes = 0
    if request.form["DeviceProtection"] == 2:
        DeviceProtection_Nointernetservice = 1
    elif request.form["DeviceProtection"] == 1:
        DeviceProtection_Yes = 1

    TechSupport_Nointernetservice = 0
    TechSupport_Yes = 0
    if request.form["TechSupport"] == 2:
        TechSupport_Nointernetservice = 1
    elif request.form["TechSupport"] == 1:
        TechSupport_Yes = 1

    StreamingTV_Nointernetservice = 0
    StreamingTV_Yes = 0
    if request.form["StreamingTV"] == 2:
        StreamingTV_Nointernetservice = 1
    elif request.form["StreamingTV"] == 1:
        StreamingTV_Yes = 1

    StreamingMovies_Nointernetservice = 0
    StreamingMovies_Yes = 0
    if request.form["StreamingMovies"] == 2:
        StreamingMovies_Nointernetservice = 1
    elif request.form["StreamingMovies"] == 1:
        StreamingMovies_Yes = 1

    Contract_Oneyear = 0
    Contract_Twoyear = 0
    if request.form["Contract"] == 1:
        Contract_Oneyear = 1
    elif request.form["Contract"] == 2:
        Contract_Twoyear = 1

    PaymentMethod_CreditCard = 0
    PaymentMethod_ElectronicCheck = 0
    PaymentMethod_MailedCheck = 0
    if request.form["PaymentMethod"] == 1:
        PaymentMethod_CreditCard = 1
    elif request.form["PaymentMethod"] == 2:
        PaymentMethod_ElectronicCheck = 1
    elif request.form["PaymentMethod"] == 3:
        PaymentMethod_MailedCheck = 1

    features = [SeniorCitizen, Partner, Dependents, PaperlessBilling, MonthlyCharges, tenure_group, InternetService_Fiberoptic, InternetService_No, OnlineSecurity_Nointernetservice, OnlineSecurity_Yes, OnlineBackup_Nointernetservice, OnlineBackup_Yes, DeviceProtection_Nointernetservice,DeviceProtection_Yes, TechSupport_Nointernetservice,TechSupport_Yes, StreamingTV_Nointernetservice, StreamingTV_Yes,StreamingMovies_Nointernetservice, StreamingMovies_Yes,Contract_Oneyear,Contract_Twoyear,PaymentMethod_CreditCard, PaymentMethod_ElectronicCheck, PaymentMethod_MailedCheck]

    final_features = [np.array(features)]
    prediction = model.predict_proba(final_features)

    output = prediction[0,1]

    return render_template('index.html', prediction_text='The Probability of customer churn: {}'.format(round(output, 2)))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(port = 8000, debug=True)
