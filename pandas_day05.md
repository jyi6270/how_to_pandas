```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import json

import warnings
warnings.filterwarnings("ignore")

print('numpy version - ', np.__version__)
print('pandas version - ', pd.__version__)
```

    numpy version -  1.21.5
    pandas version -  1.4.2
    


```python
def seriesInfo(s):
    print('index - ', s.index, type(s.index))
    print('value - ', s.values, type(s.values))
    print()
    print('data - ')
    print(s)
```


```python
def frmInfo(df):
    print('shape - ', df.shape)
    print('size - ', df.size)
    print('ndim - ', df.ndim)
    print('row index - ', df.index, type(df.index))
    print('col index - ', df.columns, type(df.columns))
    print('values - \n', df.values, type(df.values))
    print()
    print('data - ')
    display(df)def frmInfo(df):
    print('shape - ', df.shape)
    print('size - ', df.size)
    print('ndim - ', df.ndim)
    print('row index - ', df.index, type(df.index))
    print('col index - ', df.columns, type(df.columns))
    print('values - \n', df.values, type(df.values))
    print()
    print('data - ')
    display(df)
```

#### 피봇


```python
import seaborn as sns
titanic_frm = sns.load_dataset('titanic')
iris_frm = sns.load_dataset('iris')
tips_frm = sns.load_dataset('tips')
```


```python
class_sex_subset = titanic_frm.loc[: , ['pclass', 'sex']]
class_sex_subset.head()

class_sex_subset = pd.DataFrame(titanic_frm,
                               columns = ['pclass', 'sex'])
class_sex_subset.head()
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
      <th>pclass</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>female</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>male</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('문) 성별과 객실등급에 따른 승객 수 집계하고 싶다면 - ')
print('groupby - ')
display(titanic_frm.groupby(['sex', 'pclass']).size())
print()
class_sex_pivot = titanic_frm.groupby(['sex','pclass']).size().reset_index(name='cnt')
display(class_sex_pivot)
print()
print('pivot - ')
display(class_sex_pivot.pivot(index='sex', columns = 'pclass', values='cnt'))
```

    문) 성별과 객실등급에 따른 승객 수 집계하고 싶다면 - 
    groupby - 
    


    sex     pclass
    female  1          94
            2          76
            3         144
    male    1         122
            2         108
            3         347
    dtype: int64



```python
print('문) 성별과 생존여부에 따른 승객 수 집계하고 싶다면 - ')
print('groupby - ')
display(titanic_frm.groupby(['sex', 'survived']).size())
print()
class_sex_pivot = titanic_frm.groupby(['sex','survived']).size().reset_index(name='cnt')
display(class_sex_pivot)
print()
print('pivot - ')
display(class_sex_pivot.pivot(index='sex', columns = 'survived', values='cnt'))
```

    문) 성별과 생존여부에 따른 승객 수 집계하고 싶다면 - 
    groupby - 
    


    sex     survived
    female  0            81
            1           233
    male    0           468
            1           109
    dtype: int64


    
    


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
      <th>sex</th>
      <th>survived</th>
      <th>cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>0</td>
      <td>81</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>1</td>
      <td>233</td>
    </tr>
    <tr>
      <th>2</th>
      <td>male</td>
      <td>0</td>
      <td>468</td>
    </tr>
    <tr>
      <th>3</th>
      <td>male</td>
      <td>1</td>
      <td>109</td>
    </tr>
  </tbody>
</table>
</div>


    
    pivot - 
    


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
      <th>survived</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>sex</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>81</td>
      <td>233</td>
    </tr>
    <tr>
      <th>male</th>
      <td>468</td>
      <td>109</td>
    </tr>
  </tbody>
</table>
</div>


- pivot_table(index, columns, values, aggfunc, margins, margins_name)


```python
data = {
    'person' : ['A','A','A','B','B','A','A','C','B','B','B'] , 
    'day' : ['monday','tuesday','wednesday','monday','tuesday','monday','thursday','friday', 'tuesday', 'wednesday','thursday'] , 
    'sport' : ['baseball','basketball','soccer','golf','golf','basketball','soccer','tennis','baseball', 'basketball','baseball'] , 
    'time' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
}
tmp_frm = pd.DataFrame(data)

```


```python
tmp_frm.pivot_table(index= 'person', columns = 'day', values= 'time', margins= True)
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
      <th>day</th>
      <th>friday</th>
      <th>monday</th>
      <th>thursday</th>
      <th>tuesday</th>
      <th>wednesday</th>
      <th>All</th>
    </tr>
    <tr>
      <th>person</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>NaN</td>
      <td>3.500000</td>
      <td>7.0</td>
      <td>2.000000</td>
      <td>3.0</td>
      <td>3.8</td>
    </tr>
    <tr>
      <th>B</th>
      <td>NaN</td>
      <td>4.000000</td>
      <td>11.0</td>
      <td>7.000000</td>
      <td>10.0</td>
      <td>7.8</td>
    </tr>
    <tr>
      <th>C</th>
      <td>8.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>All</th>
      <td>8.0</td>
      <td>3.666667</td>
      <td>9.0</td>
      <td>5.333333</td>
      <td>6.5</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('문) 성별에 따른 생존여부의 분석결과를 pivot_table - ')
titanic_frm['cnt']=1
titanic_frm
display(titanic_frm.pivot_table(index= 'sex', columns = 'survived', values = 'cnt', 
                               aggfunc = np.sum,
                               margins = True,
                               margins_name= '분석결과'))
```

    문) 성별에 따른 생존여부의 분석결과를 pivot_table - 
    


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
      <th>survived</th>
      <th>0</th>
      <th>1</th>
      <th>분석결과</th>
    </tr>
    <tr>
      <th>sex</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>81</td>
      <td>233</td>
      <td>314</td>
    </tr>
    <tr>
      <th>male</th>
      <td>468</td>
      <td>109</td>
      <td>577</td>
    </tr>
    <tr>
      <th>분석결과</th>
      <td>549</td>
      <td>342</td>
      <td>891</td>
    </tr>
  </tbody>
</table>
</div>



```python
tips_frm.head()
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
      <th>total_bill</th>
      <th>tip</th>
      <th>sex</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.99</td>
      <td>1.01</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.34</td>
      <td>1.66</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.01</td>
      <td>3.50</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23.68</td>
      <td>3.31</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24.59</td>
      <td>3.61</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('식사대금 대비 팁의 비율이 가장 높은 데이터를 추출하고 싶다면 - ')
print('step01 - tip_ratio 컬럼 추가')
tips_frm['tip_ratio'] = tips_frm['tip'] / tips_frm['total_bill']
tips_frm.sort_values(by = 'tip_ratio', ascending=False).head(1)
print()

print('idxmax(), idxmin() - ')
print('index - ', tips_frm['tip_ratio'].idxmax())
display(tips_frm.loc[tips_frm['tip_ratio'].idxmax(), :])
```

    식사대금 대비 팁의 비율이 가장 높은 데이터를 추출하고 싶다면 - 
    step01 - tip_ratio 컬럼 추가
    
    idxmax(), idxmin() - 
    index -  172
    


    total_bill        7.25
    tip               5.15
    sex               Male
    smoker             Yes
    day                Sun
    time            Dinner
    size                 2
    tip_ratio     0.710345
    Name: 172, dtype: object



```python
print('성별에 따른 집계 - ')
tips_frm.groupby('sex').count()
```

    성별에 따른 집계 - 
    




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
      <th>total_bill</th>
      <th>tip</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
      <th>tip_ratio</th>
    </tr>
    <tr>
      <th>sex</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Male</th>
      <td>157</td>
      <td>157</td>
      <td>157</td>
      <td>157</td>
      <td>157</td>
      <td>157</td>
      <td>157</td>
    </tr>
    <tr>
      <th>Female</th>
      <td>87</td>
      <td>87</td>
      <td>87</td>
      <td>87</td>
      <td>87</td>
      <td>87</td>
      <td>87</td>
    </tr>
  </tbody>
</table>
</div>




```python
tips_frm.groupby('sex').size()
```




    sex
    Male      157
    Female     87
    dtype: int64




```python
tips_frm.groupby(['sex', 'smoker']).size()
```




    sex     smoker
    Male    Yes       60
            No        97
    Female  Yes       33
            No        54
    dtype: int64




```python
print('문) 성별과 흡연유무에 따른 집계 및 누적집계 pivot table - ')
display(tips_frm.pivot_table(index = 'sex', columns = 'smoker', values = 'tip_ratio',
                            aggfunc = pd.count,
                            margins = True,
                            margins_name = '분석결과'))
```

    문) 성별과 흡연유무에 따른 집계 및 누적집계 pivot table - 
    


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Input In [34], in <cell line: 2>()
          1 print('문) 성별과 흡연유무에 따른 집계 및 누적집계 pivot table - ')
          2 display(tips_frm.pivot_table(index = 'sex', columns = 'smoker', values = 'tip_ratio',
    ----> 3                             aggfunc = pd.count,
          4                             margins = True,
          5                             margins_name = '분석결과'))
    

    File ~\anaconda3\lib\site-packages\pandas\__init__.py:261, in __getattr__(name)
        257     from pandas.core.arrays.sparse import SparseArray as _SparseArray
        259     return _SparseArray
    --> 261 raise AttributeError(f"module 'pandas' has no attribute '{name}'")
    

    AttributeError: module 'pandas' has no attribute 'count'



```python
print('[문제01] - ')
print('qcut 명령으로 세 개의 나이(label = 청소년, 성인, 노인) 그룹(age_category)을 만든다 - ')

titanic_frm['age_category'] = pd.qcut(titanic_frm['age'], 3, labels=['청소년', '성인', '노인'])
titanic_frm.head()
```

    [문제01] - 
    qcut 명령으로 세 개의 나이(label = 청소년, 성인, 노인) 그룹(age_category)을 만든다 - 
    




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
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>adult_male</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
      <th>age_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
      <td>청소년</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>False</td>
      <td>노인</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
      <td>성인</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
      <td>노인</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
      <td>노인</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>2</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
      <td>S</td>
      <td>Second</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
      <td>성인</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>B</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
      <td>청소년</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>23.4500</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>C</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>True</td>
      <td>성인</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
      <td>Q</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Queenstown</td>
      <td>no</td>
      <td>True</td>
      <td>성인</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 16 columns</p>
</div>




```python
print('[문제02] - ')
# 성별, 선실, 나이 그룹의 생존율 서브셋을 만들고 행에는 성별 및 나이그룹에 대한 다중인덱스를 사용하고
# 열에는 선실 인덱스 사용, 생존율은 해당 그룹의 생존인원수를 전체 인원수로 나눔
# groupby - pivot

print('pivot_table - ')
sex_class_category_subset = titanic_frm.loc[:, ['sex', 'pclass', 'age_category', 'survived']]
sex_class_category_subset

```

    [문제02] - 
    pivot_table - 
    




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
      <th>sex</th>
      <th>pclass</th>
      <th>age_category</th>
      <th>survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>male</td>
      <td>3</td>
      <td>청소년</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>1</td>
      <td>노인</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>3</td>
      <td>성인</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>female</td>
      <td>1</td>
      <td>노인</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>3</td>
      <td>노인</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>male</td>
      <td>2</td>
      <td>성인</td>
      <td>0</td>
    </tr>
    <tr>
      <th>887</th>
      <td>female</td>
      <td>1</td>
      <td>청소년</td>
      <td>1</td>
    </tr>
    <tr>
      <th>888</th>
      <td>female</td>
      <td>3</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>889</th>
      <td>male</td>
      <td>1</td>
      <td>성인</td>
      <td>1</td>
    </tr>
    <tr>
      <th>890</th>
      <td>male</td>
      <td>3</td>
      <td>성인</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 4 columns</p>
</div>




```python
result_frm = sex_class_category_subset.pivot_table(index = ['sex', 'age_category'],
                                                  columns = 'pclass',
                                                  values = 'survived',
                                                  aggfunc = 'count',
                                                  margins = True,
                                                  margins_name = 'survived_ratio')
result_frm
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
      <th>pclass</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>survived_ratio</th>
    </tr>
    <tr>
      <th>sex</th>
      <th>age_category</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">female</th>
      <th>청소년</th>
      <td>22</td>
      <td>20</td>
      <td>59</td>
      <td>101</td>
    </tr>
    <tr>
      <th>성인</th>
      <td>19</td>
      <td>33</td>
      <td>27</td>
      <td>79</td>
    </tr>
    <tr>
      <th>노인</th>
      <td>44</td>
      <td>21</td>
      <td>16</td>
      <td>81</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">male</th>
      <th>청소년</th>
      <td>10</td>
      <td>28</td>
      <td>107</td>
      <td>145</td>
    </tr>
    <tr>
      <th>성인</th>
      <td>22</td>
      <td>39</td>
      <td>92</td>
      <td>153</td>
    </tr>
    <tr>
      <th>노인</th>
      <td>69</td>
      <td>32</td>
      <td>54</td>
      <td>155</td>
    </tr>
    <tr>
      <th>survived_ratio</th>
      <th></th>
      <td>186</td>
      <td>173</td>
      <td>355</td>
      <td>714</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(type(result_frm/result_frm.loc['survived_ratio', 'survived_ratio']))
display(result_frm/result_frm.loc['survived_ratio', 'survived_ratio'][0])
```

    <class 'pandas.core.frame.DataFrame'>
    


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
      <th>pclass</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>survived_ratio</th>
    </tr>
    <tr>
      <th>sex</th>
      <th>age_category</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">female</th>
      <th>청소년</th>
      <td>0.030812</td>
      <td>0.028011</td>
      <td>0.082633</td>
      <td>0.141457</td>
    </tr>
    <tr>
      <th>성인</th>
      <td>0.026611</td>
      <td>0.046218</td>
      <td>0.037815</td>
      <td>0.110644</td>
    </tr>
    <tr>
      <th>노인</th>
      <td>0.061625</td>
      <td>0.029412</td>
      <td>0.022409</td>
      <td>0.113445</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">male</th>
      <th>청소년</th>
      <td>0.014006</td>
      <td>0.039216</td>
      <td>0.149860</td>
      <td>0.203081</td>
    </tr>
    <tr>
      <th>성인</th>
      <td>0.030812</td>
      <td>0.054622</td>
      <td>0.128852</td>
      <td>0.214286</td>
    </tr>
    <tr>
      <th>노인</th>
      <td>0.096639</td>
      <td>0.044818</td>
      <td>0.075630</td>
      <td>0.217087</td>
    </tr>
    <tr>
      <th>survived_ratio</th>
      <th></th>
      <td>0.260504</td>
      <td>0.242297</td>
      <td>0.497199</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



```python
result_frm = sex_class_category_subset.pivot_table(index = 'sex',
                                                  columns = 'pclass',
                                                  values = 'survived',
                                                  aggfunc = 'count',
                                                  margins = True,
                                                  margins_name = 'survived_ratio')
```


```python
result_frm / titanic_frm.shape[0]
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
      <th>pclass</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>survived_ratio</th>
    </tr>
    <tr>
      <th>sex</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>0.105499</td>
      <td>0.085297</td>
      <td>0.161616</td>
      <td>0.352413</td>
    </tr>
    <tr>
      <th>male</th>
      <td>0.136925</td>
      <td>0.121212</td>
      <td>0.389450</td>
      <td>0.647587</td>
    </tr>
    <tr>
      <th>survived_ratio</th>
      <td>0.242424</td>
      <td>0.206510</td>
      <td>0.551066</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



- fillna(): 결측값을 원하는 값으로 채워넣을 때 사용


```python
print('타이타닉 승객 중 나이를 명시하지 않는 고객의 나이를 평균으로 대체 - ')
print('mean- ', int(titanic_frm['age'].mean()))
titanic_frm['age'].fillna(int(titanic_frm['age'].mean())).astype('int')
```

    타이타닉 승객 중 나이를 명시하지 않는 고객의 나이를 평균으로 대체 - 
    mean-  29
    




    0      22
    1      38
    2      26
    3      35
    4      35
           ..
    886    27
    887    19
    888    29
    889    26
    890    32
    Name: age, Length: 891, dtype: int32




```python
titanic_frm.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 16 columns):
     #   Column        Non-Null Count  Dtype   
    ---  ------        --------------  -----   
     0   survived      891 non-null    int64   
     1   pclass        891 non-null    int64   
     2   sex           891 non-null    object  
     3   age           714 non-null    float64 
     4   sibsp         891 non-null    int64   
     5   parch         891 non-null    int64   
     6   fare          891 non-null    float64 
     7   embarked      889 non-null    object  
     8   class         891 non-null    category
     9   who           891 non-null    object  
     10  adult_male    891 non-null    bool    
     11  deck          203 non-null    category
     12  embark_town   889 non-null    object  
     13  alive         891 non-null    object  
     14  alone         891 non-null    bool    
     15  age_category  714 non-null    category
    dtypes: bool(2), category(3), float64(2), int64(4), object(5)
    memory usage: 81.7+ KB
    


```python
print('문 - age_gender_category 추가')
print('조건 - 성별뒤에 나이를 붙여서 ex)male20 ')

titanic_frm['age_gender_category']= titanic_frm['sex']+titanic_frm['age'].astype(str)
titanic_frm
```

    문 - age_gender_category 추가
    조건 - 성별뒤에 나이를 붙여서 ex)male20 
    




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
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>adult_male</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
      <th>age_category</th>
      <th>age_gender_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
      <td>청소년</td>
      <td>male22.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>False</td>
      <td>노인</td>
      <td>female38.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
      <td>성인</td>
      <td>female26.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
      <td>노인</td>
      <td>female35.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
      <td>노인</td>
      <td>male35.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>2</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>13.0000</td>
      <td>S</td>
      <td>Second</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
      <td>성인</td>
      <td>male27.0</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>B</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
      <td>청소년</td>
      <td>female19.0</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>23.4500</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
      <td>NaN</td>
      <td>femalenan</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1</td>
      <td>1</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>C</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>True</td>
      <td>성인</td>
      <td>male26.0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7500</td>
      <td>Q</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Queenstown</td>
      <td>no</td>
      <td>True</td>
      <td>성인</td>
      <td>male32.0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 17 columns</p>
</div>



#### 다중 인덱스

- 열
- 행


```python
col_multi_idx_frm = pd.DataFrame(np.round(np.random.randn(5,4), 2),
                                columns = 
                                [['Grp01', 'Grp01', 'Grp02', 'Grp02'],
                                ['feature01', 'feature02', 'feature01', 'feature02']])
col_multi_idx_frm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">Grp01</th>
      <th colspan="2" halign="left">Grp02</th>
    </tr>
    <tr>
      <th></th>
      <th>feature01</th>
      <th>feature02</th>
      <th>feature01</th>
      <th>feature02</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.85</td>
      <td>0.37</td>
      <td>0.16</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.83</td>
      <td>-1.04</td>
      <td>-0.44</td>
      <td>-0.99</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.23</td>
      <td>-1.24</td>
      <td>-1.11</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.93</td>
      <td>0.53</td>
      <td>-2.21</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.12</td>
      <td>-0.39</td>
      <td>0.21</td>
      <td>-1.08</td>
    </tr>
  </tbody>
</table>
</div>



- 열 인덱스에 이름을 부여해서 사용의 편리성을 높일 수 있음


```python
col_multi_idx_frm.columns.names = ['GrpIdx', 'FeatureIdx']

col_multi_idx_frm.columns
```




    MultiIndex([('Grp01', 'feature01'),
                ('Grp01', 'feature02'),
                ('Grp02', 'feature01'),
                ('Grp02', 'feature02')],
               names=['GrpIdx', 'FeatureIdx'])




```python
col_multi_idx_frm = pd.DataFrame(np.round(np.random.randn(6,4), 2),
                                columns = 
                                [['Grp01', 'Grp01', 'Grp02', 'Grp02'],
                                ['feature01', 'feature02', 'feature01', 'feature02']],
                                index = [['M', 'M', 'M', 'F', 'F', 'F'],
                                         ['id_'+str(idx+1) for idx in range(6)]])
col_multi_idx_frm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">Grp01</th>
      <th colspan="2" halign="left">Grp02</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>feature01</th>
      <th>feature02</th>
      <th>feature01</th>
      <th>feature02</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">M</th>
      <th>id_1</th>
      <td>0.94</td>
      <td>-1.49</td>
      <td>0.38</td>
      <td>-0.53</td>
    </tr>
    <tr>
      <th>id_2</th>
      <td>0.06</td>
      <td>-1.38</td>
      <td>-0.49</td>
      <td>-0.26</td>
    </tr>
    <tr>
      <th>id_3</th>
      <td>-0.05</td>
      <td>-2.13</td>
      <td>-0.60</td>
      <td>-1.16</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">F</th>
      <th>id_4</th>
      <td>0.53</td>
      <td>0.97</td>
      <td>1.71</td>
      <td>0.68</td>
    </tr>
    <tr>
      <th>id_5</th>
      <td>-0.31</td>
      <td>1.28</td>
      <td>0.94</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>id_6</th>
      <td>-0.74</td>
      <td>0.72</td>
      <td>-1.58</td>
      <td>0.34</td>
    </tr>
  </tbody>
</table>
</div>




```python
col_multi_idx_frm.index.names = ['Gender', 'UserId']
col_multi_idx_frm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">Grp01</th>
      <th colspan="2" halign="left">Grp02</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>feature01</th>
      <th>feature02</th>
      <th>feature01</th>
      <th>feature02</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th>UserId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">M</th>
      <th>id_1</th>
      <td>0.94</td>
      <td>-1.49</td>
      <td>0.38</td>
      <td>-0.53</td>
    </tr>
    <tr>
      <th>id_2</th>
      <td>0.06</td>
      <td>-1.38</td>
      <td>-0.49</td>
      <td>-0.26</td>
    </tr>
    <tr>
      <th>id_3</th>
      <td>-0.05</td>
      <td>-2.13</td>
      <td>-0.60</td>
      <td>-1.16</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">F</th>
      <th>id_4</th>
      <td>0.53</td>
      <td>0.97</td>
      <td>1.71</td>
      <td>0.68</td>
    </tr>
    <tr>
      <th>id_5</th>
      <td>-0.31</td>
      <td>1.28</td>
      <td>0.94</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>id_6</th>
      <td>-0.74</td>
      <td>0.72</td>
      <td>-1.58</td>
      <td>0.34</td>
    </tr>
  </tbody>
</table>
</div>




```python
col_multi_idx_frm.index[0]
```




    ('M', 'id_1')




```python
col_multi_idx_frm.loc[col_multi_idx_frm.index[0], :]
```




    Grp01  feature01    0.94
           feature02   -1.49
    Grp02  feature01    0.38
           feature02   -0.53
    Name: (M, id_1), dtype: float64



- 인덱스를 변경할 수 있다.
- stack(): 열 -> 행
- unstack(): 행 -> 열


```python
col_multi_idx_frm.stack()
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
      <th></th>
      <th></th>
      <th>Grp01</th>
      <th>Grp02</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th>UserId</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="6" valign="top">M</th>
      <th rowspan="2" valign="top">id_1</th>
      <th>feature01</th>
      <td>0.94</td>
      <td>0.38</td>
    </tr>
    <tr>
      <th>feature02</th>
      <td>-1.49</td>
      <td>-0.53</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">id_2</th>
      <th>feature01</th>
      <td>0.06</td>
      <td>-0.49</td>
    </tr>
    <tr>
      <th>feature02</th>
      <td>-1.38</td>
      <td>-0.26</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">id_3</th>
      <th>feature01</th>
      <td>-0.05</td>
      <td>-0.60</td>
    </tr>
    <tr>
      <th>feature02</th>
      <td>-2.13</td>
      <td>-1.16</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">F</th>
      <th rowspan="2" valign="top">id_4</th>
      <th>feature01</th>
      <td>0.53</td>
      <td>1.71</td>
    </tr>
    <tr>
      <th>feature02</th>
      <td>0.97</td>
      <td>0.68</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">id_5</th>
      <th>feature01</th>
      <td>-0.31</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>feature02</th>
      <td>1.28</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">id_6</th>
      <th>feature01</th>
      <td>-0.74</td>
      <td>-1.58</td>
    </tr>
    <tr>
      <th>feature02</th>
      <td>0.72</td>
      <td>0.34</td>
    </tr>
  </tbody>
</table>
</div>




```python
stack_frm = col_multi_idx_frm.stack()
stack_frm
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
      <th></th>
      <th></th>
      <th>Grp01</th>
      <th>Grp02</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th>UserId</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="6" valign="top">M</th>
      <th rowspan="2" valign="top">id_1</th>
      <th>feature01</th>
      <td>0.94</td>
      <td>0.38</td>
    </tr>
    <tr>
      <th>feature02</th>
      <td>-1.49</td>
      <td>-0.53</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">id_2</th>
      <th>feature01</th>
      <td>0.06</td>
      <td>-0.49</td>
    </tr>
    <tr>
      <th>feature02</th>
      <td>-1.38</td>
      <td>-0.26</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">id_3</th>
      <th>feature01</th>
      <td>-0.05</td>
      <td>-0.60</td>
    </tr>
    <tr>
      <th>feature02</th>
      <td>-2.13</td>
      <td>-1.16</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">F</th>
      <th rowspan="2" valign="top">id_4</th>
      <th>feature01</th>
      <td>0.53</td>
      <td>1.71</td>
    </tr>
    <tr>
      <th>feature02</th>
      <td>0.97</td>
      <td>0.68</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">id_5</th>
      <th>feature01</th>
      <td>-0.31</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>feature02</th>
      <td>1.28</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">id_6</th>
      <th>feature01</th>
      <td>-0.74</td>
      <td>-1.58</td>
    </tr>
    <tr>
      <th>feature02</th>
      <td>0.72</td>
      <td>0.34</td>
    </tr>
  </tbody>
</table>
</div>




```python
stack_frm.index[0]
```




    ('M', 'id_1', 'feature01')




```python
stack_frm.loc[stack_frm.index[0],:]
```




    Grp01    0.94
    Grp02    0.38
    Name: (M, id_1, feature01), dtype: float64




```python
col_multi_idx_frm.unstack()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="10" halign="left">Grp01</th>
      <th>...</th>
      <th colspan="10" halign="left">Grp02</th>
    </tr>
    <tr>
      <th></th>
      <th colspan="6" halign="left">feature01</th>
      <th colspan="4" halign="left">feature02</th>
      <th>...</th>
      <th colspan="4" halign="left">feature01</th>
      <th colspan="6" halign="left">feature02</th>
    </tr>
    <tr>
      <th>UserId</th>
      <th>id_1</th>
      <th>id_2</th>
      <th>id_3</th>
      <th>id_4</th>
      <th>id_5</th>
      <th>id_6</th>
      <th>id_1</th>
      <th>id_2</th>
      <th>id_3</th>
      <th>id_4</th>
      <th>...</th>
      <th>id_3</th>
      <th>id_4</th>
      <th>id_5</th>
      <th>id_6</th>
      <th>id_1</th>
      <th>id_2</th>
      <th>id_3</th>
      <th>id_4</th>
      <th>id_5</th>
      <th>id_6</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>F</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.53</td>
      <td>-0.31</td>
      <td>-0.74</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.97</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.71</td>
      <td>0.94</td>
      <td>-1.58</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.68</td>
      <td>0.67</td>
      <td>0.34</td>
    </tr>
    <tr>
      <th>M</th>
      <td>0.94</td>
      <td>0.06</td>
      <td>-0.05</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1.49</td>
      <td>-1.38</td>
      <td>-2.13</td>
      <td>NaN</td>
      <td>...</td>
      <td>-0.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.53</td>
      <td>-0.26</td>
      <td>-1.16</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 24 columns</p>
</div>



- 다중인덱싱 어떻게?
- 튜플형식으로 접근


```python
col_multi_idx_frm[[('Grp01', 'feature01'), ('Grp02', 'feature02')]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th>Grp01</th>
      <th>Grp02</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>feature01</th>
      <th>feature02</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th>UserId</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">M</th>
      <th>id_1</th>
      <td>0.94</td>
      <td>-0.53</td>
    </tr>
    <tr>
      <th>id_2</th>
      <td>0.06</td>
      <td>-0.26</td>
    </tr>
    <tr>
      <th>id_3</th>
      <td>-0.05</td>
      <td>-1.16</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">F</th>
      <th>id_4</th>
      <td>0.53</td>
      <td>0.68</td>
    </tr>
    <tr>
      <th>id_5</th>
      <td>-0.31</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>id_6</th>
      <td>-0.74</td>
      <td>0.34</td>
    </tr>
  </tbody>
</table>
</div>




```python
col_multi_idx_frm[('Grp01', 'feature01')][0]
```




    0.94




```python
col_multi_idx_frm.loc[('M', 'id_1'), ('Grp01', 'feature01')]
```




    0.94




```python
col_multi_idx_frm.loc[('M', 'id_1'), :]
```




    Grp01  feature01    0.94
           feature02   -1.49
    Grp02  feature01    0.38
           feature02   -0.53
    Name: (M, id_1), dtype: float64



- 다중 인덱스 정렬?
- level 속성을 이용해서 기준을 정의
- axis 지정이 필요하다


```python
col_multi_idx_frm.sort_index(level = 1, axis = 0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">Grp01</th>
      <th colspan="2" halign="left">Grp02</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>feature01</th>
      <th>feature02</th>
      <th>feature01</th>
      <th>feature02</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th>UserId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">M</th>
      <th>id_1</th>
      <td>0.94</td>
      <td>-1.49</td>
      <td>0.38</td>
      <td>-0.53</td>
    </tr>
    <tr>
      <th>id_2</th>
      <td>0.06</td>
      <td>-1.38</td>
      <td>-0.49</td>
      <td>-0.26</td>
    </tr>
    <tr>
      <th>id_3</th>
      <td>-0.05</td>
      <td>-2.13</td>
      <td>-0.60</td>
      <td>-1.16</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">F</th>
      <th>id_4</th>
      <td>0.53</td>
      <td>0.97</td>
      <td>1.71</td>
      <td>0.68</td>
    </tr>
    <tr>
      <th>id_5</th>
      <td>-0.31</td>
      <td>1.28</td>
      <td>0.94</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>id_6</th>
      <td>-0.74</td>
      <td>0.72</td>
      <td>-1.58</td>
      <td>0.34</td>
    </tr>
  </tbody>
</table>
</div>



#### 실습


```python
ratings_data = pd.read_csv('./data/ratings.dat' , delimiter = "::", dtype=np.int64)
```


```python
ratings_data
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
      <th>1</th>
      <th>1193</th>
      <th>5</th>
      <th>978300760</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>661</td>
      <td>3</td>
      <td>978302109</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>914</td>
      <td>3</td>
      <td>978301968</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3408</td>
      <td>4</td>
      <td>978300275</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2355</td>
      <td>5</td>
      <td>978824291</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1197</td>
      <td>3</td>
      <td>978302268</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1000203</th>
      <td>6040</td>
      <td>1091</td>
      <td>1</td>
      <td>956716541</td>
    </tr>
    <tr>
      <th>1000204</th>
      <td>6040</td>
      <td>1094</td>
      <td>5</td>
      <td>956704887</td>
    </tr>
    <tr>
      <th>1000205</th>
      <td>6040</td>
      <td>562</td>
      <td>5</td>
      <td>956704746</td>
    </tr>
    <tr>
      <th>1000206</th>
      <td>6040</td>
      <td>1096</td>
      <td>4</td>
      <td>956715648</td>
    </tr>
    <tr>
      <th>1000207</th>
      <td>6040</td>
      <td>1097</td>
      <td>4</td>
      <td>956715569</td>
    </tr>
  </tbody>
</table>
<p>1000208 rows × 4 columns</p>
</div>




```python
ratings_data.index()
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Input In [113], in <cell line: 1>()
    ----> 1 ratings_data.index()
    

    TypeError: 'RangeIndex' object is not callable



```python
users_data = np.loadtxt('./data/users.dat',
                     
                     dtype = 'U',
                     delimiter = '::')
```


```python
users_data
```




    array([['1', 'F', '1', '10', '48067'],
           ['2', 'M', '56', '16', '70072'],
           ['3', 'M', '25', '15', '55117'],
           ...,
           ['6038', 'F', '56', '1', '14706'],
           ['6039', 'F', '45', '0', '01060'],
           ['6040', 'M', '25', '6', '11106']], dtype='<U10')




```python
movies_data = np.loadtxt('./data/movies.dat',
                     dtype = 'U',
                     delimiter = '::')
```


```python
movies_data
```




    array([['1', 'Toy Story (1995)', "Animation|Children's|Comedy"],
           ['2', 'Jumanji (1995)', "Adventure|Children's|Fantasy"],
           ['3', 'Grumpier Old Men (1995)', 'Comedy|Romance'],
           ...,
           ['3712', 'Soapdish (1991)', 'Comedy'],
           ['3713', 'Long Walk Home, The (1990)', 'Drama'],
           ['3714', "Clara's Heart (1988)", 'Drama']], dtype='<U82')




```python

```
