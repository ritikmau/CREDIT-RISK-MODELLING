import pandas as pd
import numpy as np



#for labe encoding
from sklearn.preprocessing import OneHotEncoder


#test train split
from sklearn.model_selection import train_test_split


#importing data
data = pd.read_csv("C:/Users/VIVEK/OneDrive/Documents/projects/CREDIT-RISK-MODELLING/credit_risk/credit_risk_dataset.csv")

x1 = data.iloc[:, np.r_[0:3,6:11]] #numeric
char_x2 = data.iloc[:, np.r_[3:6]] #characters
y = data.iloc[:,11]

#lable encoding
encoded_x2= OneHotEncoder(sparse_output=False)
x2=encoded_x2.fit_transform(char_x2)

# 'sparse_output=False' ensures a dense NumPy array output,
# otherwise, it returns a sparse matrix by default.

x = np.concatenate((x1, x2), axis=1)

#deletine unnecessary dataframes 
#del x1,x2

#test train split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)


