
# Instructions

This simulates retail sales for cigarettes, Juul and other e-cigarettes. We are interested in what conclusions you draw from this data about Juul's performance and market impact and how you visualize them. 

Feel free to use whatever means you prefer for your analysis and communication of it and its results. What you send back should be able to stand on its own, i.e., we should be able to understand it without requiring additional narration.

Please only use RMA and ignore CRMA and SRMA geographies for this analysis.

# Data exploration


```python
import pandas as pd
import numpy as np
import os 
import sys
import re
import datetime
os.listdir()
```




    ['.DS_Store',
     '.git',
     '.ipynb_checkpoints',
     'data_exploration.ipynb',
     'draft.ipynb',
     'final.ipynb',
     'market_share.csv',
     'notebook.tex',
     'output_11_0.png',
     'output_13_0.png',
     'output_15_0.png',
     'output_17_0.png',
     'output_19_0.png',
     'output_19_1.png',
     'output_21_0.png',
     'output_21_1.png',
     'output_28_0.png',
     'output_31_0.png',
     'output_33_0.png',
     'output_9_0.png',
     'sales.csv',
     'sales_new.csv',
     'sku.csv',
     'sku_new.csv',
     'Untitled.ipynb']




```python
sales = pd.read_csv('sales.csv'); sales.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Geography</th>
      <th>Product</th>
      <th>Time</th>
      <th>Dollar Sales</th>
      <th>Unit Sales</th>
      <th>SKU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Circle K Florida-RMA - Conv</td>
      <td>CIGARETTES</td>
      <td>4 Weeks Ending 01-31-16</td>
      <td>28921840.49</td>
      <td>4968512.070</td>
      <td>Cigarettes Total</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Circle K Florida-RMA - Conv</td>
      <td>CIGARETTES</td>
      <td>4 Weeks Ending 02-28-16</td>
      <td>30276220.80</td>
      <td>5139634.753</td>
      <td>Cigarettes Total</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Circle K Florida-RMA - Conv</td>
      <td>CIGARETTES</td>
      <td>4 Weeks Ending 03-27-16</td>
      <td>31535167.82</td>
      <td>5366848.000</td>
      <td>Cigarettes Total</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Circle K Florida-RMA - Conv</td>
      <td>CIGARETTES</td>
      <td>4 Weeks Ending 04-24-16</td>
      <td>31693487.95</td>
      <td>5420033.091</td>
      <td>Cigarettes Total</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Circle K Florida-RMA - Conv</td>
      <td>CIGARETTES</td>
      <td>4 Weeks Ending 05-22-16</td>
      <td>31390945.73</td>
      <td>5380230.139</td>
      <td>Cigarettes Total</td>
    </tr>
  </tbody>
</table>
</div>




```python
sku = pd.read_csv('sku.csv'); sku.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SKU Legend</th>
      <th>Unit Sales</th>
      <th>SKU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JUUL-ELECTRONIC SMOKING DEVICES</td>
      <td>–</td>
      <td>JUUL Total</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CIGARETTES</td>
      <td>75,083,502,411</td>
      <td>Cigarettes Total</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ELECTRONIC SMOKING DEVICES</td>
      <td>939,305,632</td>
      <td>E-Cigs Total</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JUUL ELCTRNC SMKNG ACSRY MIINT DISPOSABLE 4 CT...</td>
      <td>27,211,643</td>
      <td>JUUL Refill Kits</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JUUL ELCTRNC SMKNG ACSRY BRUULE DISPOSABLE 4 C...</td>
      <td>7,979,019</td>
      <td>JUUL Refill Kits</td>
    </tr>
  </tbody>
</table>
</div>



## Removoing non RMA Geos

As per instructions


```python
# filtering sales to only RMA geos
sales = sales[np.array(sales.Geography.apply(lambda x: re.search("-RMA", x))) != None]
```

## Fixing Date Strings

Removing the "4 weeks ending" string and creating a start and end date column for each sale


```python
# is it always ending in 4 weeks?
print(sales.shape == 
      sales[np.array(sales.Time.apply(lambda x: re.search("4 Weeks Ending", x))) != None].shape) #yes

sales['end_time'] = sales.Time.apply(lambda t: re.split(" ", t)[-1])
sales.end_time = sales.end_time.apply(lambda x: datetime.datetime.strptime(x, "%m-%d-%y"))
sales['start_time'] = sales.end_time.apply(lambda x: x - datetime.timedelta(weeks = 4))
```

    True


## Simplifying SKU Legend Names using Tf-Idf

Using Tf-Idf to extract the important words from each legend name


```python
from sklearn.feature_extraction.text import TfidfVectorizer
```


```python
sku['sku_id'] = sku.index
sku['unit_sales'] = sku.loc[:,['Unit Sales']]
sku['sku_legend'] = sku.loc[:,['SKU Legend']]
sku = sku.drop(labels = ['SKU Legend', 'Unit Sales'], axis = 1)
sku['sku_legend_text'] = sku.sku_legend.apply(lambda x: ''.join(re.findall("[A-Za-z|\s]", x)).lower())
# Simplifying SKU Names using TF-IDF
vectorizer = TfidfVectorizer()
response = vectorizer.fit_transform(list(sku.sku_legend_text))
sku_dtm = pd.DataFrame(response.toarray().transpose(), index = vectorizer.get_feature_names())
sku_simple = []
for sku_id in list(sku.sku_id):
    tf = sku_dtm.iloc[:,sku_id]
    tf = sku_dtm.iloc[:,sku_id]
    tf = tf[tf != 0]
    sku_simple.append([sku_id, ' '.join(list(tf.sort_values()[-3:].index))])
sku_simple = pd.DataFrame(sku_simple); sku_simple.columns = ['sku_id', 'sku_tf_idf']
sku = sku.merge(sku_simple,how = 'inner', on = 'sku_id')
sku
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SKU</th>
      <th>sku_id</th>
      <th>unit_sales</th>
      <th>sku_legend</th>
      <th>sku_legend_text</th>
      <th>sku_tf_idf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JUUL Total</td>
      <td>0</td>
      <td>–</td>
      <td>JUUL-ELECTRONIC SMOKING DEVICES</td>
      <td>juulelectronic smoking devices</td>
      <td>devices smoking juulelectronic</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cigarettes Total</td>
      <td>1</td>
      <td>75,083,502,411</td>
      <td>CIGARETTES</td>
      <td>cigarettes</td>
      <td>cigarettes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E-Cigs Total</td>
      <td>2</td>
      <td>939,305,632</td>
      <td>ELECTRONIC SMOKING DEVICES</td>
      <td>electronic smoking devices</td>
      <td>electronic devices smoking</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JUUL Refill Kits</td>
      <td>3</td>
      <td>27,211,643</td>
      <td>JUUL ELCTRNC SMKNG ACSRY MIINT DISPOSABLE 4 CT...</td>
      <td>juul elctrnc smkng acsry miint disposable  ct</td>
      <td>elctrnc disposable miint</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JUUL Refill Kits</td>
      <td>4</td>
      <td>7,979,019</td>
      <td>JUUL ELCTRNC SMKNG ACSRY BRUULE DISPOSABLE 4 C...</td>
      <td>juul elctrnc smkng acsry bruule disposable  ct</td>
      <td>elctrnc disposable bruule</td>
    </tr>
    <tr>
      <th>5</th>
      <td>JUUL Refill Kits</td>
      <td>5</td>
      <td>7,395,437</td>
      <td>JUUL ELCTRNC SMKNG ACSRY FRUUT DISPOSABLE 4 CT...</td>
      <td>juul elctrnc smkng acsry fruut disposable  ct</td>
      <td>elctrnc disposable fruut</td>
    </tr>
    <tr>
      <th>6</th>
      <td>JUUL Refill Kits</td>
      <td>6</td>
      <td>6,871,591</td>
      <td>JUUL ELCTRNC SMKNG ACSRY MANGO DISPOSABLE 4 CT...</td>
      <td>juul elctrnc smkng acsry mango disposable  ct</td>
      <td>elctrnc disposable mango</td>
    </tr>
    <tr>
      <th>7</th>
      <td>JUUL Refill Kits</td>
      <td>7</td>
      <td>9,532,052</td>
      <td>JUUL ELCTRNC SMKNG ACSRY TABAAC DISPOSABLE 4 C...</td>
      <td>juul elctrnc smkng acsry tabaac disposable  ct</td>
      <td>elctrnc disposable tabaac</td>
    </tr>
    <tr>
      <th>8</th>
      <td>JUUL Devices</td>
      <td>8</td>
      <td>5,596,081</td>
      <td>JUUL ELECTRONIC SMKNG DVC ELECTRONIC CIGRTT KT...</td>
      <td>juul electronic smkng dvc electronic cigrtt kt...</td>
      <td>kt rechargeable electronic</td>
    </tr>
    <tr>
      <th>9</th>
      <td>JUUL Devices</td>
      <td>9</td>
      <td>1,533,344</td>
      <td>JUUL ELECTRONIC SMKNG DVC ELECTRONIC CIGRTT KT...</td>
      <td>juul electronic smkng dvc electronic cigrtt kt...</td>
      <td>kt rechargeable electronic</td>
    </tr>
    <tr>
      <th>10</th>
      <td>JUUL Refill Kits</td>
      <td>10</td>
      <td>131</td>
      <td>JUUL ELCTRNC SMKNG ACSRY ASSORTED DISPOSABLE 4...</td>
      <td>juul elctrnc smkng acsry assorted disposable  ...</td>
      <td>elctrnc disposable assorted</td>
    </tr>
    <tr>
      <th>11</th>
      <td>JUUL Accessories</td>
      <td>11</td>
      <td>35,395</td>
      <td>JUUL ELCTRNC SMKNG ACSRY 1 CT - 0819913011561</td>
      <td>juul elctrnc smkng acsry  ct</td>
      <td>smkng acsry elctrnc</td>
    </tr>
    <tr>
      <th>12</th>
      <td>JUUL Refill Kits</td>
      <td>12</td>
      <td>215,610</td>
      <td>JUUL ELCTRNC SMKNG ACSRY COOL CUCUMBER DISPOSA...</td>
      <td>juul elctrnc smkng acsry cool cucumber disposa...</td>
      <td>disposable cool cucumber</td>
    </tr>
    <tr>
      <th>13</th>
      <td>JUUL Refill Kits</td>
      <td>13</td>
      <td>77,402</td>
      <td>JUUL ELCTRNC SMKNG ACSRY CLASSIC MENTHOL DISPO...</td>
      <td>juul elctrnc smkng acsry classic menthol dispo...</td>
      <td>disposable classic menthol</td>
    </tr>
    <tr>
      <th>14</th>
      <td>JUUL Refill Kits</td>
      <td>14</td>
      <td>33,324</td>
      <td>JUUL ELCTRNC SMKNG ACSRY CLASSIC TOBACCO DISPO...</td>
      <td>juul elctrnc smkng acsry classic tobacco dispo...</td>
      <td>disposable classic tobacco</td>
    </tr>
  </tbody>
</table>
</div>




```python
sales.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Geography</th>
      <th>Product</th>
      <th>Time</th>
      <th>Dollar Sales</th>
      <th>Unit Sales</th>
      <th>SKU</th>
      <th>end_time</th>
      <th>start_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Circle K Florida-RMA - Conv</td>
      <td>CIGARETTES</td>
      <td>4 Weeks Ending 01-31-16</td>
      <td>28921840.49</td>
      <td>4968512.070</td>
      <td>Cigarettes Total</td>
      <td>2016-01-31</td>
      <td>2016-01-03</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Circle K Florida-RMA - Conv</td>
      <td>CIGARETTES</td>
      <td>4 Weeks Ending 02-28-16</td>
      <td>30276220.80</td>
      <td>5139634.753</td>
      <td>Cigarettes Total</td>
      <td>2016-02-28</td>
      <td>2016-01-31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Circle K Florida-RMA - Conv</td>
      <td>CIGARETTES</td>
      <td>4 Weeks Ending 03-27-16</td>
      <td>31535167.82</td>
      <td>5366848.000</td>
      <td>Cigarettes Total</td>
      <td>2016-03-27</td>
      <td>2016-02-28</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Circle K Florida-RMA - Conv</td>
      <td>CIGARETTES</td>
      <td>4 Weeks Ending 04-24-16</td>
      <td>31693487.95</td>
      <td>5420033.091</td>
      <td>Cigarettes Total</td>
      <td>2016-04-24</td>
      <td>2016-03-27</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Circle K Florida-RMA - Conv</td>
      <td>CIGARETTES</td>
      <td>4 Weeks Ending 05-22-16</td>
      <td>31390945.73</td>
      <td>5380230.139</td>
      <td>Cigarettes Total</td>
      <td>2016-05-22</td>
      <td>2016-04-24</td>
    </tr>
  </tbody>
</table>
</div>




```python
sku.to_csv('sku_new.csv', index = False)
sales.to_csv('sales_new.csv', index = False)
```
