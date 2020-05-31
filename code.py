#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the Dataset
dataset=pd.read_csv("student-math.csv",sep=";")
#Creating final_grade
dataset['final_grade']=dataset['G1']+dataset['G2']+dataset['G3']
#Splitting into input and output features
x=dataset.iloc[:,:-2].values
y=dataset.iloc[:,-1]

#Encoding Categorical features
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
label_encoder_1=LabelEncoder()
x[:,0]=label_encoder_1.fit_transform(x[:,0])
x[:,1]=label_encoder_1.fit_transform(x[:,1])
x[:,3]=label_encoder_1.fit_transform(x[:,3])
x[:,4]=label_encoder_1.fit_transform(x[:,4])
x[:,5]=label_encoder_1.fit_transform(x[:,5])
x[:,8]=label_encoder_1.fit_transform(x[:,8])
x[:,9]=label_encoder_1.fit_transform(x[:,9])
x[:,10]=label_encoder_1.fit_transform(x[:,10])
x[:,11]=label_encoder_1.fit_transform(x[:,11])
x[:,15]=label_encoder_1.fit_transform(x[:,15])
x[:,16]=label_encoder_1.fit_transform(x[:,16])
x[:,17]=label_encoder_1.fit_transform(x[:,17])
x[:,18]=label_encoder_1.fit_transform(x[:,18])
x[:,19]=label_encoder_1.fit_transform(x[:,19])
x[:,20]=label_encoder_1.fit_transform(x[:,20])
x[:,21]=label_encoder_1.fit_transform(x[:,21])
x[:,22]=label_encoder_1.fit_transform(x[:,22])
ct=ColumnTransformer([('encoder',OneHotEncoder(),[8,9,10,11])],remainder='passthrough')
x=np.array(ct.fit_transform(x),dtype=np.float)
#Splitting the data into train set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Building the Linear Regression model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
#Fitting the Regressor
regressor.fit(x_train,y_train)

#Predicting the results
y_pred=(regressor.predict(x_test))
score=regressor.score(x_test,y_test)
print("Test score:",score)
score_2=regressor.score(x_test,y_pred)
print("Prediction score",score_2)
score_3=regressor.score(x_train, y_train)
print('Train Score: ', score_3)


import statsmodels.api as sm
def backwardElimination(x, sl):
	numVars = len(x[0])
	for i in range(0, numVars):
		regressor_OLS = sm.OLS(y, x).fit()
		maxVar = max(regressor_OLS.pvalues)
		if maxVar > sl:
			for j in range(0, numVars - i):
				if (regressor_OLS.pvalues[j] == maxVar):
					x = np.delete(x, j, 1)
	print(regressor_OLS.summary())
	return x
SL = 0.05
x_opt = x[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]]
x_Modeled = backwardElimination(x_opt, SL)

#Visualise
plt.scatter(y_test, y_pred, color='blue', marker="*")
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.title('y_pred vs y_test')
plt.show()


# Ploting the dependency of final_grade on  some important attributes
#age
dataset.boxplot(by='age',column=['final_grade'],grid=False)
#famrel
dataset.boxplot(by='famrel',column=['final_grade'],grid=False)
#G1
dataset.boxplot(by='G1',column=['final_grade'],grid=False)
#G2
dataset.boxplot(by='G2',column=['final_grade'],grid=False)

#Using Random Forest Regressor  on opt_x(Important Features)
Opt_x=pd.DataFrame(data=x_Modeled,columns=['age','famrel','absences','G1','G2'])
X=Opt_x.iloc[:,:]
Y=dataset.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestRegressor
regressor_1=RandomForestRegressor(n_estimators=10,random_state=0)
regressor_1.fit(X_train,Y_train)
Y_pred=regressor_1.predict(X_test)
Score=regressor_1.score(X_test,Y_test)
print("Test score:",Score)
Score_2=regressor_1.score(X_test,Y_pred)
print("Preiction score",Score_2)
Score_3=regressor_1.score(X_train, Y_train)
print("Train ",Score_3)











