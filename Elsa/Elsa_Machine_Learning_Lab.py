import numpy as np
import pandas as pd
import re
from pandas import Series, DataFrame
import random
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pylab import rcParams
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Load our data file.
orders= pd.read_csv('~/Desktop/Machine_Learning_Lab_class/data/Orders.csv')
returns= pd.read_csv('~/Desktop/Machine_Learning_Lab_class/data/Returns.csv')
orders.isnull()
np.sum(orders.isnull())

orders.columns


# Problem 1:Convert Profit and Sales to numeric
orders['Profit'] = orders.Profit.str.replace('[,$]','').astype('float')
orders['Sales'] = orders.Sales.str.replace('[,$]','').astype('float')

# Problem 2:
    #1. Is there any seasonal trend of inventory in the company?
    #2. Is the seasonal trend the same for different categories?

#Is there any seasonal trend of inventory in the company?
orders['Ship.Date'] = orders['Ship.Date'].astype('datetime64[ns]')
orders['Order.Date'] = orders['Order.Date'].astype('datetime64[ns]')
orders['month']=orders['Order.Date'].dt.month
orders['year']=orders['Order.Date'].dt.year

plt.figure(figsize=(12, 6))
orders.groupby('month')['Quantity'].sum().plot.bar(color='b')

#Is the seasonal trend the same for different categories?
orders.groupby(['month','Category'])['Quantity'].sum().unstack().plot.bar()

#### Problem 3: Why did customers make returns?
complete=pd.merge(orders, returns, how='inner', right_on='Order ID', left_on ='Order.ID')

#How much profit did we lose due to returns each year?
complete.columns
orders.columns
returns.columns
# How much profit did we lose due to returns each year?
complete.groupby('year')['Profit'].sum().plot.bar()

complete.groupby('year')['Profit'].sum()
# How many customer returned more than once? more than 5 times?
s=complete.groupby('Customer.ID').size()
s[s>1].count()
s[s>1].count()/s.count()

# More than 5 times?
s[s>5].count()
s[s>5].count()/s.count()

# Which regions are more likely to return orders?
complete.columns

#Which regions are more likely to return orders?
complete.groupby('Region_x')['Returned'].count().sort_values(ascending=False).head(10)

# Which categories (sub-categories) of products are more likely to be returned?
complete.groupby('Category')['Returned'].count().sort_values(ascending=False).plot.bar()
complete.groupby('Sub.Category')['Returned'].count().sort_values(ascending=False).plot.bar()

### Part II: Machine Learning and Business Use Case
### Problem 4: Feature Engineering
# Generate a categorical variable which indicates whether an order has been returned or not.
orders_with_return=pd.merge(orders, returns, how='outer', right_on='Order ID', left_on ='Order.ID')

orders_with_return.shape
orders.shape
returns.shape
orders_with_return.columns

orders_with_return['Returned'] = orders_with_return['Returned'].fillna('No')
orders_with_return['Returned']=pd.get_dummies(orders_with_return['Returned'], drop_first=True, dummy_na=True)
orders_with_return.drop("Order ID", axis = 1, inplace = True)
orders_with_return['Returned'].sample(300)

##### Step 2:
#generate a feature which can measure how long it takes the company to process each order.
#- ***Hint:*** Process.Time = Ship.Date - Order.Date
orders.columns
orders_with_return.columns

orders_with_return['Process.Time'] = orders_with_return['Ship.Date'].dt.dayofyear - orders_with_return['Order.Date'].dt.dayofyear


##### Step 3:
#- Let us generate a feature indictes how many times the product has been returned before.
#- If it never got returned, we just impute using 0.
#- ***Hint:*** Group by different Product.ID
count_returned=orders_with_return.groupby('Product.ID').size().reset_index(name ='Return.Times')
orders_with_return = pd.merge(orders_with_return, count_returned, left_on = 'Product.ID', right_on ='Product.ID', how = 'outer')

orders_with_return.columns
orders_with_return = orders_with_return.rename(columns={'Returned_x': 'Returned', 'Returned_Y': 'count_returned'})

###### Problem 5: Fitting Models
# Save the 'Id' column
#Split the data into test and training sets
#train_mask = np.random.rand(len(orders_with_return)) < 0.8
#train = orders_with_return[train_mask]
#test = orders_with_return[~train_mask]

#useless_features=['Row.ID','Ship.Date','Order.Date','Order ID','Region_y','Product.Name','Customer.Name']
#train.drop(useless_features,axis=1).columns

useless_features=['Row.ID','Ship.Date','Order.Date','Region_y','Product.Name','Customer.Name']
orders_with_return
## Drop non useful columns
orders_with_return.drop(useless_features, axis = 1, inplace = True)
orders_with_return.columns

#Ship.Mode
orders_with_return['Ship.Mode'].unique()
ship_mode_cols = ['Standard Class','Second Class', 'First Class', 'Same Day']
ship_mode_dict = {'Standard Class':4,'Second Class': 3, 'First Class':2, 'Same Day':1}
orders_with_return['Ship.Mode'] = orders_with_return['Ship.Mode'].map(lambda x: ship_mode_dict.get(x, 0))

#Order.Priority
orders_with_return['Order.Priority'].unique()
ship_mode_cols = ['Low','Medium', 'High', 'Critical']
ship_mode_dict = {'Low':4,'Medium': 3, 'High':2, 'Critical':1}
orders_with_return['Order.Priority'] = orders_with_return['Order.Priority'].map(lambda x: ship_mode_dict.get(x, 0))

#Fit model:
use_columns = ['Sales', 'Quantity', 'Return.Times', 'Discount', 'Process.Time','Shipping.Cost', 'Segment',\
               'Ship.Mode', 'Region_x', 'Category', 'month', 'Order.Priority', 'Profit']


X = pd.get_dummies(orders_with_return[use_columns], drop_first=True, dummy_na=True)
y = orders_with_return['Returned']

# Train.Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit logistic Regression Model

LR = LogisticRegression(class_weight='balanced')
LR.fit(X_train, y_train)
y_predict = LR.predict(X_test)

# Evaluate the model
confusion_matrix(y_test, y_predict)
#Fit a random forest model
rf = RandomForestClassifier(n_estimators=1000, max_depth=9, class_weight='balanced', n_jobs=-1)
rf.fit(X_train, y_train)
y_predict = rf.predict(X_test)

#Evaluate the model
confusion_matrix(y_test, y_predict)
roc_auc_score(y_test, y_predict)
