#### 학습목표
- 분석하려는 데이터는 대부분 .csv 파일을 테이블형식으로 처리할 수 있는 타입
- 제공되는 2 type: Series, DataFrame
- Series: index + value


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
s = pd.Series([1,2,3,4,5], dtype=np.float)
seriesInfo(s)
```

    index -  RangeIndex(start=0, stop=5, step=1) <class 'pandas.core.indexes.range.RangeIndex'>
    value -  [1. 2. 3. 4. 5.] <class 'numpy.ndarray'>
    
    data - 
    0    1.0
    1    2.0
    2    3.0
    3    4.0
    4    5.0
    dtype: float64
    


```python
s = pd.Series({'a':1, 'b':2, 'c':3}, dtype=np.float)
seriesInfo(s)
```

    index -  Index(['a', 'b', 'c'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    value -  [1. 2. 3.] <class 'numpy.ndarray'>
    
    data - 
    a    1.0
    b    2.0
    c    3.0
    dtype: float64
    

- 인덱스의 라벨은 정수, 문자, 날짜, 시간


```python
s = pd.Series([1,2,3,4,5],
             index = ['강남', '서초', '방배', '동작', '신도림'])
seriesInfo(s)
```

    index -  Index(['강남', '서초', '방배', '동작', '신도림'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    value -  [1 2 3 4 5] <class 'numpy.ndarray'>
    
    data - 
    강남     1
    서초     2
    방배     3
    동작     4
    신도림    5
    dtype: int64
    

- name: 시리즈 데이터에 이름을 부여해주는 역할
- index.name: 시리즈의 인덱스에 이름을 부여해주는 역할


```python
s.name = '데이터에 이름 부여'
s.index.name = '구 구별'
seriesInfo(s)
```

    index -  Index(['강남', '서초', '방배', '동작', '신도림'], dtype='object', name='구 구별') <class 'pandas.core.indexes.base.Index'>
    value -  [1 2 3 4 5] <class 'numpy.ndarray'>
    
    data - 
    구 구별
    강남     1
    서초     2
    방배     3
    동작     4
    신도림    5
    Name: 데이터에 이름 부여, dtype: int64
    


```python
tmp_data = ('섭섭해', '2022-11-10', '남자', True)
s = pd.Series(tmp_data, dtype = np.object,
             index = ['이름', '생년월일', '성별', '결혼여부'])
seriesInfo(s)
```

    index -  Index(['이름', '생년월일', '성별', '결혼여부'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    value -  ['섭섭해' '2022-11-10' '남자' True] <class 'numpy.ndarray'>
    
    data - 
    이름             섭섭해
    생년월일    2022-11-10
    성별              남자
    결혼여부          True
    dtype: object
    

- series에서 원소를 선택하기 위해서는?
- 정수형 위치 인덱스 또는 인덱스 이름을 활용해야 한다



```python
print('정수 인덱싱 - ', s[0], type(s[0]))
print('인덱스 이름 - ', s['이름'], type(s['이름']))
print()

print(s[[0,1]])
print(s[['이름', '생년월일']])
print()

print('범위지정 - slicing')
print(s['이름': '성별'])
print(s[0:3])
```

    정수 인덱싱 -  섭섭해 <class 'str'>
    인덱스 이름 -  섭섭해 <class 'str'>
    
    이름             섭섭해
    생년월일    2022-11-10
    dtype: object
    이름             섭섭해
    생년월일    2022-11-10
    dtype: object
    
    범위지정 - slicing
    이름             섭섭해
    생년월일    2022-11-10
    성별              남자
    dtype: object
    이름             섭섭해
    생년월일    2022-11-10
    성별              남자
    dtype: object
    


```python
for idx, value in s.items() :
    print('idx : {}, \t\tvalue : {}'.format(idx, value))
print()

for idx in s.keys() :
    print('idx : {}'.format(idx))
print()

for value in s.values:
    print('value: {}'.format(value))
```

    idx : 이름, 		value : 섭섭해
    idx : 생년월일, 		value : 2022-11-10
    idx : 성별, 		value : 남자
    idx : 결혼여부, 		value : True
    
    idx : 이름
    idx : 생년월일
    idx : 성별
    idx : 결혼여부
    
    value: 섭섭해
    value: 2022-11-10
    value: 남자
    value: True
    


```python
s.index
```




    Index(['이름', '생년월일', '성별', '결혼여부'], dtype='object')




```python
s.values
```




    array(['섭섭해', '2022-11-10', '남자', True], dtype=object)




```python
s= pd.Series(range(10,21))
seriesInfo(s)
```

    index -  RangeIndex(start=0, stop=11, step=1) <class 'pandas.core.indexes.range.RangeIndex'>
    value -  [10 11 12 13 14 15 16 17 18 19 20] <class 'numpy.ndarray'>
    
    data - 
    0     10
    1     11
    2     12
    3     13
    4     14
    5     15
    6     16
    7     17
    8     18
    9     19
    10    20
    dtype: int64
    


```python
print('시리즈의 데이터는 배열이므로 벡터화 연산이 가능하다 - ')
s * 20
```

    시리즈의 데이터는 배열이므로 벡터화 연산이 가능하다 - 
    




    0     200
    1     220
    2     240
    3     260
    4     280
    5     300
    6     320
    7     340
    8     360
    9     380
    10    400
    dtype: int64




```python
print('2의 배수인 것만 추출 - ')
s[s%2 == 0]
```

    2의 배수인 것만 추출 - 
    




    0     10
    2     12
    4     14
    6     16
    8     18
    10    20
    dtype: int64




```python
from datetime import date, datetime, timedelta
from dateutil.parser import parse
```


```python
day = datetime(2022, 11, 10)
print(day + timedelta(days=1))
```

    2022-11-11 00:00:00
    

- 평균이 50이고 편차 5 정규분포 데이터를 10일간 만들고 싶다면?


```python
np.random.normal(50, 5, (10,))
```




    array([60.62544598, 47.32405731, 56.72439685, 48.30823329, 49.46910552,
           48.28829455, 54.00162691, 59.91710431, 47.83661914, 47.2786089 ])




```python
series01 = pd.Series(np.random.normal(50, 5, (10,)),
                    index = [ day + timedelta(days=d) for d in range(10)])
series01
```




    2022-11-10    47.349038
    2022-11-11    47.887336
    2022-11-12    46.803442
    2022-11-13    43.369385
    2022-11-14    49.373680
    2022-11-15    56.821507
    2022-11-16    52.494063
    2022-11-17    56.681184
    2022-11-18    47.260392
    2022-11-19    48.572947
    dtype: float64



- 데이터 갱신


```python
price_series = pd.Series([4000, 3000, 3500, 2000],
                       index = ['a', 'b', 'c', 'd'])
seriesInfo(price_series)
print()

price_series[0] = 5000
seriesInfo(price_series)
```

    index -  Index(['a', 'b', 'c', 'd'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    value -  [4000 3000 3500 2000] <class 'numpy.ndarray'>
    
    data - 
    a    4000
    b    3000
    c    3500
    d    2000
    dtype: int64
    
    index -  Index(['a', 'b', 'c', 'd'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    value -  [5000 3000 3500 2000] <class 'numpy.ndarray'>
    
    data - 
    a    5000
    b    3000
    c    3500
    d    2000
    dtype: int64
    

- 데이터 삭제
- del


```python
price_series['e'] = 9000
seriesInfo(price_series)
print()

del price_series['e']
seriesInfo(price_series)
```

    index -  Index(['a', 'b', 'c', 'd', 'e'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    value -  [5000 3000 3500 2000 9000] <class 'numpy.ndarray'>
    
    data - 
    a    5000
    b    3000
    c    3500
    d    2000
    e    9000
    dtype: int64
    
    index -  Index(['a', 'b', 'c', 'd'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    value -  [5000 3000 3500 2000] <class 'numpy.ndarray'>
    
    data - 
    a    5000
    b    3000
    c    3500
    d    2000
    dtype: int64
    


```python
price_series['e'] = np.NaN
seriesInfo(price_series)
```

    index -  Index(['a', 'b', 'c', 'd', 'e'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    value -  [5000. 3000. 3500. 2000.   nan] <class 'numpy.ndarray'>
    
    data - 
    a    5000.0
    b    3000.0
    c    3500.0
    d    2000.0
    e       NaN
    dtype: float64
    


```python
print('isnull - ')
print(pd.isnull(price_series))
print()

print(price_series[pd.isnull(price_series)])
print()

print(price_series[price_series.isnull()])
print()

print('notnull - ')
print(price_series[price_series.notnull()])
```

    isnull - 
    a    False
    b    False
    c    False
    d    False
    e     True
    dtype: bool
    
    e   NaN
    dtype: float64
    
    e   NaN
    dtype: float64
    
    notnull - 
    a    5000.0
    b    3000.0
    c    3500.0
    d    2000.0
    dtype: float64
    


```python
series01 = pd.Series([400,200,350,500],
                    index = ['a', 'o', 'k', 'm'])
series01
```




    a    400
    o    200
    k    350
    m    500
    dtype: int64




```python
series02 = pd.Series([400,200,350,500],
                    index = ['o', 'a', 'h', 'm'])
series02
```




    o    400
    a    200
    h    350
    m    500
    dtype: int64




```python
series03 = series01 + series02
print(series03)
print()

print('mean - ', np.mean(series03))
print('mean - ', series03.mean())
print('mean - ', series03.sum() / len(series03))

print('fillna() - ')
print(series03.fillna(0))
series04 = series03.fillna(0)
print('mean - ', series04.mean())
print()

series03 = series01.add(series02, fill_value = 0)
print(series03)
print()

print('len() - ', len(series03))
```

    a     600.0
    h       NaN
    k       NaN
    m    1000.0
    o     600.0
    dtype: float64
    
    mean -  733.3333333333334
    mean -  733.3333333333334
    mean -  440.0
    fillna() - 
    a     600.0
    h       0.0
    k       0.0
    m    1000.0
    o     600.0
    dtype: float64
    mean -  440.0
    
    a     600.0
    h     350.0
    k     350.0
    m    1000.0
    o     600.0
    dtype: float64
    
    len() -  5
    


```python
print(series03)
print('결측값을 제외한 subset - ')
series05 = series03[pd.notnull(series03)]
print(series05)
```

    a     600.0
    h     350.0
    k     350.0
    m    1000.0
    o     600.0
    dtype: float64
    결측값을 제외한 subset - 
    a     600.0
    h     350.0
    k     350.0
    m    1000.0
    o     600.0
    dtype: float64
    

- DataFrame: pd.DataFrame()
- 행 인덱스(문자), 열 인덱스(문자)


```python
print('dict 이용한 생성 - ')
dict_data = {
    'feature01': [1,2,3],
    'feature02': [1,2,3],
    'feature03': [1,2,3],
    'feature04': [1,2,3],
    'feature05': [1,2,3]    
}
tmp_frm = pd.DataFrame(dict_data)
tmp_frm
```

    dict 이용한 생성 - 
    




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
      <th>feature01</th>
      <th>feature02</th>
      <th>feature03</th>
      <th>feature04</th>
      <th>feature05</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
def frmInfo(df):
    print('shape - ', df.shape)
    print('size - ', df.size)
    print('ndim - ', df.ndim)
    print('row index - ', df.index, type(df.index))
    print('col index - ', df.columns, type(df.columns))
    print('values - ', df.values)
    print()
    print('data - ')
    display(df)
```


```python
frmInfo(tmp_frm)
```

    shape -  (3, 5)
    size -  15
    ndim -  2
    row index -  RangeIndex(start=0, stop=3, step=1) <class 'pandas.core.indexes.range.RangeIndex'>
    col index -  Index(['feature01', 'feature02', 'feature03', 'feature04', 'feature05'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    values -  [[1 1 1 1 1]
     [2 2 2 2 2]
     [3 3 3 3 3]]
    
    data - 
    


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
      <th>feature01</th>
      <th>feature02</th>
      <th>feature03</th>
      <th>feature04</th>
      <th>feature05</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



```python
print('리스트를 이용한 생성 - ')
list_frm = pd.DataFrame([['임정섭', 'm', True], ['임재원', 'm', False]],
                       index = ['user_' + str(idx) for idx in range(1,3)],
                       columns = ['이름', '성별', '결혼여부'])
frmInfo(list_frm)
```

    리스트를 이용한 생성 - 
    shape -  (2, 3)
    size -  6
    ndim -  2
    row index -  Index(['user_1', 'user_2'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    col index -  Index(['이름', '성별', '결혼여부'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    values -  [['임정섭' 'm' True]
     ['임재원' 'm' False]]
    
    data - 
    


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
      <th>이름</th>
      <th>성별</th>
      <th>결혼여부</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>user_1</th>
      <td>임정섭</td>
      <td>m</td>
      <td>True</td>
    </tr>
    <tr>
      <th>user_2</th>
      <td>임재원</td>
      <td>m</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



```python
print('전처리 - 열 이름 변경(영문 -> 한글, 한글 -> 영문), rename()')
print('원본에 반영 - inplace')

list_frm.rename(columns = {'이름': 'name', '성별' : 'gender', '결혼여부': 'marriage'}, inplace = True)
```

    전처리 - 열 이름 변경(영문 -> 한글, 한글 -> 영문), rename()
    원본에 반영 - inplace
    


```python
list_frm
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
      <th>name</th>
      <th>gender</th>
      <th>marriage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>user_1</th>
      <td>임정섭</td>
      <td>m</td>
      <td>True</td>
    </tr>
    <tr>
      <th>user_2</th>
      <td>임재원</td>
      <td>m</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
list_frm.index.name = 'customer'
list_frm.columns.name = 'feature'
frmInfo(list_frm)
```

    shape -  (2, 3)
    size -  6
    ndim -  2
    row index -  Index(['user_1', 'user_2'], dtype='object', name='customer') <class 'pandas.core.indexes.base.Index'>
    col index -  Index(['name', 'gender', 'marriage'], dtype='object', name='feature') <class 'pandas.core.indexes.base.Index'>
    values -  [['임정섭' 'm' True]
     ['임재원' 'm' False]]
    
    data - 
    


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
      <th>feature</th>
      <th>name</th>
      <th>gender</th>
      <th>marriage</th>
    </tr>
    <tr>
      <th>customer</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>user_1</th>
      <td>임정섭</td>
      <td>m</td>
      <td>True</td>
    </tr>
    <tr>
      <th>user_2</th>
      <td>임재원</td>
      <td>m</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



```python
frmInfo(list_frm)
print()

print(list_frm.index.name)
print(list_frm.columns.name)
```

    shape -  (2, 3)
    size -  6
    ndim -  2
    row index -  Index(['user_1', 'user_2'], dtype='object', name='customer') <class 'pandas.core.indexes.base.Index'>
    col index -  Index(['name', 'gender', 'marriage'], dtype='object', name='feature') <class 'pandas.core.indexes.base.Index'>
    values -  [['임정섭' 'm' True]
     ['임재원' 'm' False]]
    
    data - 
    


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
      <th>feature</th>
      <th>name</th>
      <th>gender</th>
      <th>marriage</th>
    </tr>
    <tr>
      <th>customer</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>user_1</th>
      <td>임정섭</td>
      <td>m</td>
      <td>True</td>
    </tr>
    <tr>
      <th>user_2</th>
      <td>임재원</td>
      <td>m</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


    
    customer
    feature
    


```python

```


```python
print('data extract - \n', list_frm['name'], type(list_frm['name']))
```

    data extract - 
     customer
    user_1    임정섭
    user_2    임재원
    Name: name, dtype: object <class 'pandas.core.series.Series'>
    


```python
print('add feature - ')
list_frm['age'] = [20, 30]
list_frm
```

    add feature - 
    




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
      <th>feature</th>
      <th>name</th>
      <th>gender</th>
      <th>marriage</th>
      <th>age</th>
    </tr>
    <tr>
      <th>customer</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>user_1</th>
      <td>임정섭</td>
      <td>m</td>
      <td>True</td>
      <td>20</td>
    </tr>
    <tr>
      <th>user_2</th>
      <td>임재원</td>
      <td>m</td>
      <td>False</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('del feature - ')
del list_frm['age']
list_frm
```

    del feature - 
    




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
      <th>feature</th>
      <th>name</th>
      <th>gender</th>
      <th>marriage</th>
    </tr>
    <tr>
      <th>customer</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>user_1</th>
      <td>임정섭</td>
      <td>m</td>
      <td>True</td>
    </tr>
    <tr>
      <th>user_2</th>
      <td>임재원</td>
      <td>m</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



- 행 인덱싱: 무조건 슬라이싱
- 배열 인덱싱, 라벨 인덱싱, 숫자 인덱싱 가능


```python
list_frm[:1]
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
      <th>feature</th>
      <th>name</th>
      <th>gender</th>
      <th>marriage</th>
    </tr>
    <tr>
      <th>customer</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>user_1</th>
      <td>임정섭</td>
      <td>m</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
list_frm[ : 'user_1']
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
      <th>feature</th>
      <th>name</th>
      <th>gender</th>
      <th>marriage</th>
    </tr>
    <tr>
      <th>customer</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>user_1</th>
      <td>임정섭</td>
      <td>m</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
list_frm['name']['user_2']
```




    '임재원'




```python
import urllib
url = 'http://kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json?key=f5eef3421c602c6cb7ea224104795888&targetDt=20120101'
response_page = urllib.request.urlopen(url)
print('response - ')
print(response_page.read())
```

    response - 
    b'{"boxOfficeResult":{"boxofficeType":"\xec\x9d\xbc\xeb\xb3\x84 \xeb\xb0\x95\xec\x8a\xa4\xec\x98\xa4\xed\x94\xbc\xec\x8a\xa4","showRange":"20120101~20120101","dailyBoxOfficeList":[{"rnum":"1","rank":"1","rankInten":"0","rankOldAndNew":"OLD","movieCd":"20112207","movieNm":"\xeb\xaf\xb8\xec\x85\x98\xec\x9e\x84\xed\x8c\x8c\xec\x84\x9c\xeb\xb8\x94:\xea\xb3\xa0\xec\x8a\xa4\xed\x8a\xb8\xed\x94\x84\xeb\xa1\x9c\xed\x86\xa0\xec\xbd\x9c","openDt":"2011-12-15","salesAmt":"2776060500","salesShare":"36.3","salesInten":"-415699000","salesChange":"-13","salesAcc":"40541108500","audiCnt":"353274","audiInten":"-60106","audiChange":"-14.5","audiAcc":"5328435","scrnCnt":"697","showCnt":"3223"},{"rnum":"2","rank":"2","rankInten":"1","rankOldAndNew":"OLD","movieCd":"20110295","movieNm":"\xeb\xa7\x88\xec\x9d\xb4 \xec\x9b\xa8\xec\x9d\xb4","openDt":"2011-12-21","salesAmt":"1189058500","salesShare":"15.6","salesInten":"-105894500","salesChange":"-8.2","salesAcc":"13002897500","audiCnt":"153501","audiInten":"-16465","audiChange":"-9.7","audiAcc":"1739543","scrnCnt":"588","showCnt":"2321"},{"rnum":"3","rank":"3","rankInten":"-1","rankOldAndNew":"OLD","movieCd":"20112621","movieNm":"\xec\x85\x9c\xeb\xa1\x9d\xed\x99\x88\xec\xa6\x88 : \xea\xb7\xb8\xeb\xa6\xbc\xec\x9e\x90 \xea\xb2\x8c\xec\x9e\x84","openDt":"2011-12-21","salesAmt":"1176022500","salesShare":"15.4","salesInten":"-210328500","salesChange":"-15.2","salesAcc":"10678327500","audiCnt":"153004","audiInten":"-31283","audiChange":"-17","audiAcc":"1442861","scrnCnt":"360","showCnt":"1832"},{"rnum":"4","rank":"4","rankInten":"0","rankOldAndNew":"OLD","movieCd":"20113260","movieNm":"\xed\x8d\xbc\xed\x8e\x99\xed\x8a\xb8 \xea\xb2\x8c\xec\x9e\x84","openDt":"2011-12-21","salesAmt":"644532000","salesShare":"8.4","salesInten":"-75116500","salesChange":"-10.4","salesAcc":"6640940000","audiCnt":"83644","audiInten":"-12225","audiChange":"-12.8","audiAcc":"895416","scrnCnt":"396","showCnt":"1364"},{"rnum":"5","rank":"5","rankInten":"0","rankOldAndNew":"OLD","movieCd":"20113271","movieNm":"\xed\x94\x84\xeb\xa0\x8c\xec\xa6\x88: \xeb\xaa\xac\xec\x8a\xa4\xed\x84\xb0\xec\x84\xac\xec\x9d\x98\xeb\xb9\x84\xeb\xb0\x80 ","openDt":"2011-12-29","salesAmt":"436753500","salesShare":"5.7","salesInten":"-89051000","salesChange":"-16.9","salesAcc":"1523037000","audiCnt":"55092","audiInten":"-15568","audiChange":"-22","audiAcc":"202909","scrnCnt":"290","showCnt":"838"},{"rnum":"6","rank":"6","rankInten":"1","rankOldAndNew":"OLD","movieCd":"19940256","movieNm":"\xeb\x9d\xbc\xec\x9d\xb4\xec\x98\xa8 \xed\x82\xb9","openDt":"1994-07-02","salesAmt":"507115500","salesShare":"6.6","salesInten":"-114593500","salesChange":"-18.4","salesAcc":"1841625000","audiCnt":"45750","audiInten":"-11699","audiChange":"-20.4","audiAcc":"171285","scrnCnt":"244","showCnt":"895"},{"rnum":"7","rank":"7","rankInten":"-1","rankOldAndNew":"OLD","movieCd":"20113381","movieNm":"\xec\x98\xa4\xec\x8b\xb9\xed\x95\x9c \xec\x97\xb0\xec\x95\xa0","openDt":"2011-12-01","salesAmt":"344871000","salesShare":"4.5","salesInten":"-107005500","salesChange":"-23.7","salesAcc":"20634684500","audiCnt":"45062","audiInten":"-15926","audiChange":"-26.1","audiAcc":"2823060","scrnCnt":"243","showCnt":"839"},{"rnum":"8","rank":"8","rankInten":"0","rankOldAndNew":"OLD","movieCd":"20112709","movieNm":"\xea\xb7\xb9\xec\x9e\xa5\xed\x8c\x90 \xed\x8f\xac\xec\xbc\x93\xeb\xaa\xac\xec\x8a\xa4\xed\x84\xb0 \xeb\xb2\xa0\xec\x8a\xa4\xed\x8a\xb8 \xec\x9c\x84\xec\x8b\x9c\xe3\x80\x8c\xeb\xb9\x84\xed\x81\xac\xed\x8b\xb0\xeb\x8b\x88\xec\x99\x80 \xeb\xb0\xb1\xec\x9d\x98 \xec\x98\x81\xec\x9b\x85 \xeb\xa0\x88\xec\x8b\x9c\xeb\x9d\xbc\xeb\xac\xb4\xe3\x80\x8d","openDt":"2011-12-22","salesAmt":"167809500","salesShare":"2.2","salesInten":"-45900500","salesChange":"-21.5","salesAcc":"1897120000","audiCnt":"24202","audiInten":"-7756","audiChange":"-24.3","audiAcc":"285959","scrnCnt":"186","showCnt":"348"},{"rnum":"9","rank":"9","rankInten":"0","rankOldAndNew":"OLD","movieCd":"20113311","movieNm":"\xec\x95\xa8\xeb\xb9\x88\xea\xb3\xbc \xec\x8a\x88\xed\x8d\xbc\xeb\xb0\xb4\xeb\x93\x9c3","openDt":"2011-12-15","salesAmt":"137030000","salesShare":"1.8","salesInten":"-35408000","salesChange":"-20.5","salesAcc":"3416675000","audiCnt":"19729","audiInten":"-6461","audiChange":"-24.7","audiAcc":"516289","scrnCnt":"169","showCnt":"359"},{"rnum":"10","rank":"10","rankInten":"0","rankOldAndNew":"OLD","movieCd":"20112708","movieNm":"\xea\xb7\xb9\xec\x9e\xa5\xed\x8c\x90 \xed\x8f\xac\xec\xbc\x93\xeb\xaa\xac\xec\x8a\xa4\xed\x84\xb0 \xeb\xb2\xa0\xec\x8a\xa4\xed\x8a\xb8 \xec\x9c\x84\xec\x8b\x9c \xe3\x80\x8c\xeb\xb9\x84\xed\x81\xac\xed\x8b\xb0\xeb\x8b\x88\xec\x99\x80 \xed\x9d\x91\xec\x9d\x98 \xec\x98\x81\xec\x9b\x85 \xec\xa0\x9c\xed\x81\xac\xeb\xa1\x9c\xeb\xac\xb4\xe3\x80\x8d","openDt":"2011-12-22","salesAmt":"125535500","salesShare":"1.6","salesInten":"-40756000","salesChange":"-24.5","salesAcc":"1595695000","audiCnt":"17817","audiInten":"-6554","audiChange":"-26.9","audiAcc":"235070","scrnCnt":"175","showCnt":"291"}]}}'
    


```python
res_json = json.loads(response_page.read())
res_json
```




    {'boxOfficeResult': {'boxofficeType': '일별 박스오피스',
      'showRange': '20120101~20120101',
      'dailyBoxOfficeList': [{'rnum': '1',
        'rank': '1',
        'rankInten': '0',
        'rankOldAndNew': 'OLD',
        'movieCd': '20112207',
        'movieNm': '미션임파서블:고스트프로토콜',
        'openDt': '2011-12-15',
        'salesAmt': '2776060500',
        'salesShare': '36.3',
        'salesInten': '-415699000',
        'salesChange': '-13',
        'salesAcc': '40541108500',
        'audiCnt': '353274',
        'audiInten': '-60106',
        'audiChange': '-14.5',
        'audiAcc': '5328435',
        'scrnCnt': '697',
        'showCnt': '3223'},
       {'rnum': '2',
        'rank': '2',
        'rankInten': '1',
        'rankOldAndNew': 'OLD',
        'movieCd': '20110295',
        'movieNm': '마이 웨이',
        'openDt': '2011-12-21',
        'salesAmt': '1189058500',
        'salesShare': '15.6',
        'salesInten': '-105894500',
        'salesChange': '-8.2',
        'salesAcc': '13002897500',
        'audiCnt': '153501',
        'audiInten': '-16465',
        'audiChange': '-9.7',
        'audiAcc': '1739543',
        'scrnCnt': '588',
        'showCnt': '2321'},
       {'rnum': '3',
        'rank': '3',
        'rankInten': '-1',
        'rankOldAndNew': 'OLD',
        'movieCd': '20112621',
        'movieNm': '셜록홈즈 : 그림자 게임',
        'openDt': '2011-12-21',
        'salesAmt': '1176022500',
        'salesShare': '15.4',
        'salesInten': '-210328500',
        'salesChange': '-15.2',
        'salesAcc': '10678327500',
        'audiCnt': '153004',
        'audiInten': '-31283',
        'audiChange': '-17',
        'audiAcc': '1442861',
        'scrnCnt': '360',
        'showCnt': '1832'},
       {'rnum': '4',
        'rank': '4',
        'rankInten': '0',
        'rankOldAndNew': 'OLD',
        'movieCd': '20113260',
        'movieNm': '퍼펙트 게임',
        'openDt': '2011-12-21',
        'salesAmt': '644532000',
        'salesShare': '8.4',
        'salesInten': '-75116500',
        'salesChange': '-10.4',
        'salesAcc': '6640940000',
        'audiCnt': '83644',
        'audiInten': '-12225',
        'audiChange': '-12.8',
        'audiAcc': '895416',
        'scrnCnt': '396',
        'showCnt': '1364'},
       {'rnum': '5',
        'rank': '5',
        'rankInten': '0',
        'rankOldAndNew': 'OLD',
        'movieCd': '20113271',
        'movieNm': '프렌즈: 몬스터섬의비밀 ',
        'openDt': '2011-12-29',
        'salesAmt': '436753500',
        'salesShare': '5.7',
        'salesInten': '-89051000',
        'salesChange': '-16.9',
        'salesAcc': '1523037000',
        'audiCnt': '55092',
        'audiInten': '-15568',
        'audiChange': '-22',
        'audiAcc': '202909',
        'scrnCnt': '290',
        'showCnt': '838'},
       {'rnum': '6',
        'rank': '6',
        'rankInten': '1',
        'rankOldAndNew': 'OLD',
        'movieCd': '19940256',
        'movieNm': '라이온 킹',
        'openDt': '1994-07-02',
        'salesAmt': '507115500',
        'salesShare': '6.6',
        'salesInten': '-114593500',
        'salesChange': '-18.4',
        'salesAcc': '1841625000',
        'audiCnt': '45750',
        'audiInten': '-11699',
        'audiChange': '-20.4',
        'audiAcc': '171285',
        'scrnCnt': '244',
        'showCnt': '895'},
       {'rnum': '7',
        'rank': '7',
        'rankInten': '-1',
        'rankOldAndNew': 'OLD',
        'movieCd': '20113381',
        'movieNm': '오싹한 연애',
        'openDt': '2011-12-01',
        'salesAmt': '344871000',
        'salesShare': '4.5',
        'salesInten': '-107005500',
        'salesChange': '-23.7',
        'salesAcc': '20634684500',
        'audiCnt': '45062',
        'audiInten': '-15926',
        'audiChange': '-26.1',
        'audiAcc': '2823060',
        'scrnCnt': '243',
        'showCnt': '839'},
       {'rnum': '8',
        'rank': '8',
        'rankInten': '0',
        'rankOldAndNew': 'OLD',
        'movieCd': '20112709',
        'movieNm': '극장판 포켓몬스터 베스트 위시「비크티니와 백의 영웅 레시라무」',
        'openDt': '2011-12-22',
        'salesAmt': '167809500',
        'salesShare': '2.2',
        'salesInten': '-45900500',
        'salesChange': '-21.5',
        'salesAcc': '1897120000',
        'audiCnt': '24202',
        'audiInten': '-7756',
        'audiChange': '-24.3',
        'audiAcc': '285959',
        'scrnCnt': '186',
        'showCnt': '348'},
       {'rnum': '9',
        'rank': '9',
        'rankInten': '0',
        'rankOldAndNew': 'OLD',
        'movieCd': '20113311',
        'movieNm': '앨빈과 슈퍼밴드3',
        'openDt': '2011-12-15',
        'salesAmt': '137030000',
        'salesShare': '1.8',
        'salesInten': '-35408000',
        'salesChange': '-20.5',
        'salesAcc': '3416675000',
        'audiCnt': '19729',
        'audiInten': '-6461',
        'audiChange': '-24.7',
        'audiAcc': '516289',
        'scrnCnt': '169',
        'showCnt': '359'},
       {'rnum': '10',
        'rank': '10',
        'rankInten': '0',
        'rankOldAndNew': 'OLD',
        'movieCd': '20112708',
        'movieNm': '극장판 포켓몬스터 베스트 위시 「비크티니와 흑의 영웅 제크로무」',
        'openDt': '2011-12-22',
        'salesAmt': '125535500',
        'salesShare': '1.6',
        'salesInten': '-40756000',
        'salesChange': '-24.5',
        'salesAcc': '1595695000',
        'audiCnt': '17817',
        'audiInten': '-6554',
        'audiChange': '-26.9',
        'audiAcc': '235070',
        'scrnCnt': '175',
        'showCnt': '291'}]}}




```python
data = res_json['boxOfficeResult']['dailyBoxOfficeList']
```


```python
#case1
lst = []
for tmp_dict in data :
    lst.append( [tmp_dict['rnum'], tmp_dict['movieNm'], tmp_dict['salesAmt']] )
print()

movie_frm = pd.DataFrame(lst,
                        columns = ['랭킹', '영화제목', '판매금액'])
display(movie_frm)
print()
frmInfo(movie_frm)
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
      <th>랭킹</th>
      <th>영화제목</th>
      <th>판매금액</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>미션임파서블:고스트프로토콜</td>
      <td>2776060500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>마이 웨이</td>
      <td>1189058500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>셜록홈즈 : 그림자 게임</td>
      <td>1176022500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>퍼펙트 게임</td>
      <td>644532000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>프렌즈: 몬스터섬의비밀</td>
      <td>436753500</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>라이온 킹</td>
      <td>507115500</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>오싹한 연애</td>
      <td>344871000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>극장판 포켓몬스터 베스트 위시「비크티니와 백의 영웅 레시라무」</td>
      <td>167809500</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>앨빈과 슈퍼밴드3</td>
      <td>137030000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>극장판 포켓몬스터 베스트 위시 「비크티니와 흑의 영웅 제크로무」</td>
      <td>125535500</td>
    </tr>
  </tbody>
</table>
</div>


    
    shape -  (10, 3)
    size -  30
    ndim -  2
    row index -  RangeIndex(start=0, stop=10, step=1) <class 'pandas.core.indexes.range.RangeIndex'>
    col index -  Index(['랭킹', '영화제목', '판매금액'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    values -  [['1' '미션임파서블:고스트프로토콜' '2776060500']
     ['2' '마이 웨이' '1189058500']
     ['3' '셜록홈즈 : 그림자 게임' '1176022500']
     ['4' '퍼펙트 게임' '644532000']
     ['5' '프렌즈: 몬스터섬의비밀 ' '436753500']
     ['6' '라이온 킹' '507115500']
     ['7' '오싹한 연애' '344871000']
     ['8' '극장판 포켓몬스터 베스트 위시「비크티니와 백의 영웅 레시라무」' '167809500']
     ['9' '앨빈과 슈퍼밴드3' '137030000']
     ['10' '극장판 포켓몬스터 베스트 위시 「비크티니와 흑의 영웅 제크로무」' '125535500']]
    
    data - 
    


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
      <th>랭킹</th>
      <th>영화제목</th>
      <th>판매금액</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>미션임파서블:고스트프로토콜</td>
      <td>2776060500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>마이 웨이</td>
      <td>1189058500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>셜록홈즈 : 그림자 게임</td>
      <td>1176022500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>퍼펙트 게임</td>
      <td>644532000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>프렌즈: 몬스터섬의비밀</td>
      <td>436753500</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>라이온 킹</td>
      <td>507115500</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>오싹한 연애</td>
      <td>344871000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>극장판 포켓몬스터 베스트 위시「비크티니와 백의 영웅 레시라무」</td>
      <td>167809500</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>앨빈과 슈퍼밴드3</td>
      <td>137030000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>극장판 포켓몬스터 베스트 위시 「비크티니와 흑의 영웅 제크로무」</td>
      <td>125535500</td>
    </tr>
  </tbody>
</table>
</div>



```python
#case2
pd.DataFrame(data)[['rnum', 'movieNm', 'salesAmt']]
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
      <th>rnum</th>
      <th>movieNm</th>
      <th>salesAmt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>미션임파서블:고스트프로토콜</td>
      <td>2776060500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>마이 웨이</td>
      <td>1189058500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>셜록홈즈 : 그림자 게임</td>
      <td>1176022500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>퍼펙트 게임</td>
      <td>644532000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>프렌즈: 몬스터섬의비밀</td>
      <td>436753500</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>라이온 킹</td>
      <td>507115500</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>오싹한 연애</td>
      <td>344871000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>극장판 포켓몬스터 베스트 위시「비크티니와 백의 영웅 레시라무」</td>
      <td>167809500</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>앨빈과 슈퍼밴드3</td>
      <td>137030000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>극장판 포켓몬스터 베스트 위시 「비크티니와 흑의 영웅 제크로무」</td>
      <td>125535500</td>
    </tr>
  </tbody>
</table>
</div>



문1) 다음 조건을 만족하는 임의의 데이터 프레임을 작성해 보자
- 조건) 열의 갯수와 행의 갯수가 각각 5개 이상이어야 한다.
- 조건) 열에는 정수, 문자, 실수, 날짜 데이터가 각각 1개 이상 포함되어야 한다.

문2) 제공된 booklist_json.json 파일로부터 데이터를 읽어들여서 데이터프레임을 작성해보자
- hint) open() , json.load()
- json.loads(): json으로 된 문자열 파일을 읽어들일 때
- json.load(): json 파일을 읽어들일 때


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
