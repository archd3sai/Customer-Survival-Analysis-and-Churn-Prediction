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
- There is a constant decrease in survival probability probability between 3-60 months.
- After 60 months or 5 years, survival probability decreases with a higher rate. 

**Log-Rank Test:** 

Log-rank test is carried out to analyze churning probabilities group wise and to find if there is statistical significance between groups. The plots show survival curve group wise.

<img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/gender.png" width="425"/> <img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/Senior%20Citizen.png" width="425"/> 

<img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/partner_1.png" width="425"/> <img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/dependents.png" width="425"/> 

<img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/phoneservice.png" width="425"/> <img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/MultipleLines.png" width="425"/> 

<img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/InternetService.png" width="425"/> <img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/OnlineSecurity.png" width="425"/> 

<img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/OnlineBackup.png" width="425"/> <img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/DeviceProtection.png" width="425"/> 

<img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/TechSupport.png" width="425"/> <img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/StreamingTv.png" width="425"/> 

<img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/Contract.png" width="425"/> <img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/StreamingMovies.png" width="425"/> 

<img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/paymentmethod.png" width="425"/> <img src="https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/PaperlessBilling.png" width="425"/> 

From above graphs we can conclude following:
- Customer's Gender and the phone service type are not indictive features and their p value of log rank test is above threshold value 0.05.
- If customer is young and has a family, he or she is less likely to churn. The reason might be the busy life, more money or another factors.
- If customer is not enrolled in services like online backup, online security, device protection, tech support, streaming Tv and streaming movies even though having active internet service, the survival probability is less.
- The company should traget customers who opt for internet service as their survival probability constantly descreases. Also, Fiber Optilc type of Internet Service is costly and fast compared to DSL and this might be the reason of higher customer churning. 
- More offers should be given to customers who opt for month-to-month contract and company should target customers to subscribe for long-term service. 
- If customer's paying method is automatic, he or she is less likely to churn. The reason is in the case of electronic check and mailed check, a customer has to make an effort to pay and it takes time.

## Customer Churn Prediction
I aim to implement a machine learning model to accurately predict if the customer will churn or not.

### Analysis

**Churn and Tenure Relationship:**
![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/tenure-churn.png)

- As we can see the higher the tenure, the lesser the churn rate. This tells us that the customer becomes loyal with the tenure.

**Tenure Distrbution by Various Services:**
![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/tenure-dist.png)

- When the customers are new they do not opt for various services and their churning rate is very high. This can be seen in above plot for Streaming Movies and this holds true for all various services.

**Internet Service By Contract Type:**
![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/internetservice-contract.png)

- Many of the people of who opt for month-to-month Contract choose Fiber optic as Internet service and this is the reason for higher churn rate for fiber optic Internet service type.

**Payment method By Contract Type:**
![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/payment-contract.png)

- People having month-to-month contract prefer paying by Electronic Check mostly or mailed check. The reason might be short subscription cancellation process compared to automatic payment.

**Monthly Charges:**

![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/monthlycharges.png)

- As we can see the customers paying high monthly fees churn more.

### Modelling

For the modelling, I will use tress based Ensemble method as we do not have linear 

![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/l1.png)
![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/f1.png)

In the data, we have less number of customers who stopped their service so model fit more on data with churn feature as a No. This is a problem of class imbalance and to deal with that, I carried out oversampling using synthetic minority oversampling technique (SMOTE). Then applying Logistic Regression model we get better results which can be seen below.

![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/l2.png)
![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/f2.png)

From the feature importance analysis, we can see the coefficients of variables and how they affect the churning of customer.




