import numpy as np
import pandas as pd
import re
from pandas import Series, DataFrame
import random
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pylab import rcParams

# Load our data file.
orders= pd.read_csv('~/Desktop/Machine_Learning_Lab/data/Orders.csv')
returns= pd.read_csv('~/Desktop/Machine_Learning_Lab/data/Returns.csv')
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

orders['Order.Date'] = orders['Order.Date'].astype('datetime64[ns]')
orders = orders.set_index('Order.Date')
orders['month']=orders.index.month
orders['year']=orders.index.year

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
