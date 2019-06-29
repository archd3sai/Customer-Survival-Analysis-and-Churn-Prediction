# Customer Churn Analysis and Prediction

Customer attrition, also known as customer churn, customer turnover, or customer defection, is the loss of clients or customers.

Telephone service companies, Internet service providers, pay TV companies, insurance firms, and alarm monitoring services, often use customer attrition analysis and customer attrition rates as one of their key business metrics because the cost of retaining an existing customer is far less than acquiring a new one. Companies from these sectors often have customer service branches which attempt to win back defecting clients, because recovered long-term customers can be worth much more to a company than newly recruited clients.

predictive analytics use churn prediction models that predict customer churn by assessing their propensity of risk to churn. Since these models generate a small prioritized list of potential defectors, they are effective at focusing customer retention marketing programs on the subset of the customer base who are most vulnerable to churn.

In this project I aim to perform customer survival analysis and build a model which can predict customer churn.

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
- If customer does not churn after 1-2 months, his or her probability of churning decreases rapidly.
- There is a constant decrease in churning probability between 3-60 months.
- After 60 months or 5 years, churing probability decreases with a higher rate and we can say that the customers have become loyal to the service.

**Log-Rank Test:** 

Log-rank test is carried out to analyze churning probabilities group wise and the plots below show same.

![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/gender.png) | ![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/Senior%20Citizen.png)

![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/partner.png) | ![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/dependents.png)

![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/phoneservice.png) | ![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/MultipleLines.png)

![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/InternetService.png) ![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/OnlineSecurity.png)

![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/OnlineBackup.png) ![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/DeviceProtection.png)

![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/TechSupport.png) ![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/StreamingTv.png)

![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/Contract.png) ![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/StreamingMovies.png)

![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/paymentmethod.png) ![](https://github.com/archd3sai/Customer-Churn-Analysis-and-Prediction/blob/master/Images/PaperlessBilling.png)






## Customer Churn Prediction
logistic regression with synthetic minority oversampling technique (SMOTE). 
