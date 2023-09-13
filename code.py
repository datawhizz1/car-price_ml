# car-price_ml
#This ML model predicts the price of a car considering its engine and body specs and model.

#Importing Relevant Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression

#Data Cleaning
data=pd.read_csv('1.04.+Real-life+example.csv')
data.isnull()
data.describe(include='all')
data.isnull().sum()
data.info()
no_mv_data= data.dropna(axis=0)
no_mv_data.describe(include='all')
#analysing the PDFs
sns.displot(no_mv_data['Price'])
#dealing with outliers
upper=no_mv_data['Price'].quantile(0.99)
sns.displot(new_data_1['Mileage'])
upper=new_data_1['Mileage'].quantile(0.99)
new_data_1=new_data_1[no_mv_data['Mileage']<upper]
sns.distplot(new_data_1['EngineV'])
new_data_1=new_data_1[new_data_1['EngineV']<6.5]
sns.displot(new_data_1['Year'])
lower=new_data_1['Year'].quantile(0.01)
new_data_1=new_data_1[new_data_1['Year']>lower]
#resetting the index
data_cleaned=new_data_1.reset_index(drop=True)

#Verifying OLS Assumptions
#Checking for linearity
f, (ax1,ax2,ax3)=plt.subplots(3,1,sharey=True,figsize=(5,10))
ax1.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax1.set_title("Mileage vs Price")
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax2.set_title("Engine Volume vs Price")
ax3.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax3.set_title("Year vs Price")
plt.show()
new_data_1=no_mv_data[no_mv_data['Price']<upper]
#log transformation
log_price=np.log(data_cleaned['Price'])
data_cleaned['LogPrice']=log_price
f, (ax1,ax2,ax3)=plt.subplots(3,1,sharey=True,figsize=(5,10))
ax1.scatter(data_cleaned['Mileage'],data_cleaned['LogPrice'])
ax1.set_title("Mileage vs Price")
ax2.scatter(data_cleaned['EngineV'],data_cleaned['LogPrice'])
ax2.set_title("Engine Volume vs Price")
ax3.scatter(data_cleaned['Year'],data_cleaned['LogPrice'])
ax3.set_title("Year vs Price")
plt.show()
#checking for multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor as vf
variables=data_cleaned[['Mileage','Year','EngineV']]
vif=pd.DataFrame()
vif['VIF']=[vf(variables.values,i) for i in range(variables.shape[1])]
vif['features']=variables.columns
data_cleaned=data_cleaned.drop(['Year'],axis=1) #high multicollinearity

#Creating Dummy Variables
data_with_dummies=pd.get_dummies(data_cleaned,drop_first=True)
data_preprocessed=data_with_dummies
data_preprocessed.describe(include='all')

#Building the Model

target=data_preprocessed['LogPrice']
input=data_preprocessed.drop(['LogPrice'],axis=1)

#Standardizing the inputs
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(input)
inputs_scaled=scaler.transform(input)

#Train-test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inputs_scaled, target,test_size=0.15,random_state=42)

reg=LinearRegression()
reg.fit(x_train,y_train)

y_hat=reg.predict(x_train)
plt.scatter(y_train,y_hat)
plt.xlabel("Targets")
plt.ylabel("Predictions")
plt.xlim(7,12)
plt.ylim(7,12)
plt.show()
reg.intercept_

sns.displot(y_hat-y_train)

reg.score(x_train,y_train)#r-sqaured

reg.intercept_

#Creating Regression Summary Table
reg_summary=pd.DataFrame(input.columns.values,columns=['Features'])
reg_summary['Weights']=reg.coef_
pd.set_option('display.float_format',lambda x:'%.2f' %x)

#Testing
pred_y=reg.predict(x_test)
plt.scatter(np.exp(y_test),pred_y)
plt.show()



