import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
time=pd.read_csv("C:/Users/MAHESHWARI/datasciencecourse/1DATA SCIENCE COURSE DATA SETS/Assignments/Simple linear regression/Salary_Data.csv")
print(time.head())
print(time.shape)
input()

slr_model2=smf.ols("Salary~YearsExperience",data=time).fit()
print("Training Done")
print(slr_model2.summary())

pred=slr_model2.predict(time.iloc[:,0])
print("predicrion Done")
print(pred)

#visulization
plt.scatter(x=time['YearsExperience'],y=time['Salary'],color='red');plt.plot(time['YearsExperience'],pred,color='blue')
plt.xlabel('YearsExperience');plt.ylabel('Salary')
plt.show()
input()
pred.corr(time.Salary)
