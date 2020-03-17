# Customer Survival Analysis and Churn Prediction

Customer attrition, also known as customer churn, customer turnover, or customer defection, is the loss of clients or customers.

Telephone service companies, Internet service providers, pay TV companies, insurance firms, and alarm monitoring services, often use customer attrition analysis and customer attrition rates as one of their key business metrics because the cost of retaining an existing customer is far less than acquiring a new one. Companies from these sectors often have customer service branches which attempt to win back defecting clients, because recovered long-term customers can be worth much more to a company than newly recruited clients.

predictive analytics use churn prediction models that predict customer churn by assessing their propensity of risk to churn. Since these models generate a small prioritized list of potential defectors, they are effective at focusing customer retention marketing programs on the subset of the customer base who are most vulnerable to churn.

In this project I aim to perform customer survival analysis and build a model which can predict customer churn.

## Project Organization
```
.
├── Images/                             : contains images for report
├── static/                             : contains gauge chart to show churn probability in Flask App 
│   └── images/
│       └── new_plot.png
├── templates/                          : contains html template for flask app
│   └── index.html
├── Customer Survival Analysis.ipynb    : Survival Analysis kaplan-Meier curve and log-rank test
├── main.ipynb                          : Data Analysis and Random Forest model building
├── app.py                              : Flask App
├── app.png                             : Final App image  
├── explainer.bz2                       : Shap Explainer
├── model.pkl                           : Random Forest model
├── requirements.txt                    : requirements to run this model
├── LICENSE.md                          : MIT License
└── README.md                           : Report
```

## Final Customer Churn Prediction App
<img src=https://github.com/archd3sai/Customer-Survival-Analysis-and-Churn-Prediction/blob/master/app.png>

## Customer Survival Analysis

**Survival Analysis:** 
Survival analysis is generally defined as a set of methods for analyzing data where the outcome variable is the time until the occurrence of an event of interest. The event can be death, occurrence of a disease, marriage, divorce, etc. The time to event or survival time can be measured in days, weeks, years, etc.

For example, if the event of interest is heart attack, then the survival time can be the time in years until a person develops a heart attack.

**Objective:**
The objective of this analysis is to utilize non-parametric methods of survival analysis to answer the following questions.
- How the likelihood of the customer churn changes over time?
- How we can model the relationship between customer churn, time, and other customer characteristics?
- What are the significant factors that drive customer churn?

**Kaplan-Meier Survival Curve:**

![survivalcurve](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/SurvivalCurve.png)

From above graph, we can say that
- AS expected, for telcom, churn is relatively low. The company was able to retain more than 60% of its customers even after 72 months.
- There is a constant decrease in churning probability between 3-60 months.
- After 60 months or 5 years, churing probability decreases with a higher rate and we can say that the customers have become loyal to the service.

**Log-Rank Test:** 

Log-rank test is carried out to analyze churning probabilities group wise and the plots below show same.

<img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/gender.png" width="425"/> <img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/Senior%20Citizen.png" width="425"/> 

<img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/partner_1.png" width="425"/> <img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/dependents.png" width="425"/> 

<img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/phoneservice.png" width="425"/> <img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/MultipleLines.png" width="425"/> 

<img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/InternetService.png" width="425"/> <img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/OnlineSecurity.png" width="425"/> 

<img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/OnlineBackup.png" width="425"/> <img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/DeviceProtection.png" width="425"/> 

<img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/TechSupport.png" width="425"/> <img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/StreamingTv.png" width="425"/> 

<img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/Contract.png" width="425"/> <img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/StreamingMovies.png" width="425"/> 

<img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/paymentmethod.png" width="425"/> <img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/PaperlessBilling.png" width="425"/> 

From above graphs we can conclude following:
- Customer's Gender and the phone service type are not indictive features.
- If customer is senior setizen and/or single and/or does not have dependents, he or she is less likely to churn. The reason might be low monthly payment for a single person. 
- If customer is not enrolled in services like online backup, online security, device protection, tech support, streaming Tv and streaming movies, customer's churning probability is significantly less.
- The company should traget customers who do not opt for internet service as their churning probability remains high throughout the tenure.
- More offers should be given to customers who opt for two year contract as the end of their contract to retain them. 
- If customer's paying method is automatic, he or she is more likely to churn.

## Customer Churn Prediction
I aim to implement machine learning model to accurately predict if the customer will churn or not.

### Analysis

**Churn and Tenure Relationship:**
![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/tenure-churn.png)

**Monthly Charges:**

![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/monthlycharges.png)

**Total Charges:**

![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/totalcharges.png)

From above plots, we can say that there is a less probability of churning of customers who spend less on services and whose monthly charges are less.

### Modelling

Logistic Regression Model:

![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/l1.png)
![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/f1.png)

In the data, we have less number of customers who stopped their service so model fit more on data with churn feature as a No. This is a problem of class imbalance and to deal with that, I carried out oversampling using synthetic minority oversampling technique (SMOTE). Then applying Logistic Regression model we get better results which can be seen below.

![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/l2.png)
![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/f2.png)

From the feature importance analysis, we can see the coefficients of variables and how they affect the churning of customer.




