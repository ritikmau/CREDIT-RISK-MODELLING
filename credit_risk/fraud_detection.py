import pandas as pd
import numpy as np



#for labe encoding
from sklearn.preprocessing import OneHotEncoder


#test train split
from sklearn.model_selection import train_test_split


#Ml algorithms
from sklearn.tree import DecisionTreeClassifier

#confusion metrics
from sklearn.metrics import confusion_matrix

#for accuracy 
from sklearn.metrics import accuracy_score


#visualization of decision tree
from sklearn.tree import export_graphviz  


#importing data
data = pd.read_csv("credit_risk_dataset.csv")

#data preprocessing
print(data.isnull().sum())

data = data.dropna()              # drop rows with NaN

data = data.drop_duplicates()       #drop duplicate rows




x1 = data.iloc[:, 0:8] #numeric
char_x2 = data.iloc[:, 8:11] #characters


y = data.iloc[:,11].values
data["cb_person_default_on_file"]= data["cb_person_default_on_file"].map({"Y":1, "N":0}) 


#lable encoding
encoded_x2= OneHotEncoder(sparse_output=False)
x2=encoded_x2.fit_transform(char_x2)
# 'sparse_output=False' ensures a dense NumPy array output,
# otherwise, it returns a sparse matrix by default.


# convert x2 to DataFrame with proper column names
x2 = pd.DataFrame(x2, columns=encoded_x2.get_feature_names_out(char_x2.columns))

# merge numeric + categorical
x = pd.concat([x1.reset_index(drop=True), x2.reset_index(drop=True)], axis=1)

#deletine unnecessary dataframes 
#del x1,x2



#test train split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)


classifier = DecisionTreeClassifier(criterion= 'entropy', random_state=0)
classifier.fit(x_train, y_train)


y_pred = classifier.predict(x_test)


cm = confusion_matrix(y_test, y_pred)


acc=accuracy_score(y_test, y_pred)

print("confusion metrix:\n",cm)

print("accuracy=", acc)

export_graphviz(
    classifier,
    out_file='DT_Viz.dot',
    feature_names=x.columns,      # works now because x is DataFrame
    class_names=['No Default', 'Default'],
    filled=True
)