import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
time=pd.read_csv("C:/Users/MAHESHWARI/datasciencecourse/1DATA SCIENCE COURSE DATA SETS/Assignments/Simple linear regression/delivery_time.csv")
print(time.head())
print(time.shape)
input()

slr_model=smf.ols("DeliveryTime~SortingTime",data=time).fit()
print("Training Done")
print(slr_model.summary())

pred=slr_model.predict(time.iloc[:,1])
print(pred)

#visulization
plt.scatter(x=time['SortingTime'],y=time['DeliveryTime'],color='red');plt.plot(time['SortingTime'],pred,color='blue')
plt.xlabel('SortingTime');plt.ylabel('DeliveryTime')
plt.show()
input()
#pred.corr(time.Delivery Time)
