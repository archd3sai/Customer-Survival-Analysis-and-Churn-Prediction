import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
import shap
shap.initjs()
import time
from flask import Flask, request, jsonify, render_template

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

    def tenure(t):
        if t<=12:
            return 1
        elif t>12 and t<=24:
            return 2
        elif t>24 and t<=36:
            return 3
        elif t>36 and t<=48:
            return 4
        elif t>48 and t<=60:
            return 5
        else:
            return 6

    tenure_group = tenure(int(request.form["Tenure"]))

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

    columns = ['SeniorCitizen', 'Partner', 'Dependents', 'PaperlessBilling', 'MonthlyCharges', 'tenure_group', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

    final_features = [np.array(features)]
    prediction = model.predict_proba(final_features)

    output = prediction[0,1]

    explainer = joblib.load(filename="explainer.bz2")
    shap_values = explainer.shap_values(np.array(final_features))
    shap.force_plot(explainer.expected_value[1], shap_values[1], columns, matplotlib = True, show = False).savefig('static/images/shap.png', bbox_inches="tight")

    def degree_range(n):
        start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
        end = np.linspace(0,180,n+1, endpoint=True)[1::]
        mid_points = start + ((end-start)/2.)
        return np.c_[start, end], mid_points

    def rot_text(ang):
        rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
        return rotation

    def gauge(labels=['LOW','MEDIUM','HIGH','EXTREME'], \
              colors=['#007A00','#0063BF','#FFCC00','#ED1C24'], Probability=1, fname=False):

        N = len(labels)
        colors = colors[::-1]


        """
        begins the plotting
        """

        fig, ax = plt.subplots()

        ang_range, mid_points = degree_range(4)

        labels = labels[::-1]

        """
        plots the sectors and the arcs
        """
        patches = []
        for ang, c in zip(ang_range, colors):
            # sectors
            patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
            # arcs
            patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))

        [ax.add_patch(p) for p in patches]


        """
        set the labels (e.g. 'LOW','MEDIUM',...)
        """

        for mid, lab in zip(mid_points, labels):

            ax.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, \
                horizontalalignment='center', verticalalignment='center', fontsize=14, \
                fontweight='bold', rotation = rot_text(mid))

        """
        set the bottom banner and the title
        """
        r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
        ax.add_patch(r)

        ax.text(0, -0.05, 'Churn Probability ' + np.round(Probability,2).astype(str), horizontalalignment='center', \
             verticalalignment='center', fontsize=22, fontweight='bold')

        """
        plots the arrow now
        """

        pos = (1-Probability)*180
        ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
                     width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')

        ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
        ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

        """
        removes frame and ticks, and makes axis equal and tight
        """

        ax.set_frame_on(False)
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axis('equal')
        plt.tight_layout()

        plt.savefig('static/images/new_plot.png')

    gauge(Probability = output)

    t = time.time()
    return render_template('index.html', prediction_text='The Probability of customer churn: {}'.format(round(output, 2)), url_1 ='static/images/new_plot.png?{}'.format(t), url_2 ='static/images/shap.png?{}'.format(t))


if __name__ == "__main__":
    app.run(port = 8000, debug=True)
