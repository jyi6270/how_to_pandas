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
    display(df)
```

DataFrame: row_index, 데이터 조작, 인덱스 조작
- loc[]: 라벨값 기반
- iloc[]: 정수값 기반


```python
sample_frm = pd.DataFrame(np.arange(10,22).reshape(3,4),
                         index = ['row01', 'row02', 'row03'],
                         columns = ['col01', 'col02', 'col03', 'col04'])
frmInfo(sample_frm)
```

    shape -  (3, 4)
    size -  12
    ndim -  2
    row index -  Index(['row01', 'row02', 'row03'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    col index -  Index(['col01', 'col02', 'col03', 'col04'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    values -  [[10 11 12 13]
     [14 15 16 17]
     [18 19 20 21]]
    
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
      <th>col01</th>
      <th>col02</th>
      <th>col03</th>
      <th>col04</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>row01</th>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>13</td>
    </tr>
    <tr>
      <th>row02</th>
      <td>14</td>
      <td>15</td>
      <td>16</td>
      <td>17</td>
    </tr>
    <tr>
      <th>row03</th>
      <td>18</td>
      <td>19</td>
      <td>20</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>



```python
print('행 인덱스를 사용하며 행 1개 선택 - ')
print('라벨기반 - ', sample_frm.loc['row01'])
print('정수기반 - ', sample_frm.iloc[0])
```

    행 인덱스를 사용하며 행 1개 선택 - 
    라벨기반 -  col01    10
    col02    11
    col03    12
    col04    13
    Name: row01, dtype: int32
    정수기반 -  col01    10
    col02    11
    col03    12
    col04    13
    Name: row01, dtype: int32
    


```python
# 행 여러개 선택
print('라벨기반 - \n', sample_frm.loc[['row01', 'row03']])
print('정수기반 - \n', sample_frm.iloc[[0,2]])
```

    라벨기반 - 
            col01  col02  col03  col04
    row01     10     11     12     13
    row03     18     19     20     21
    정수기반 - 
            col01  col02  col03  col04
    row01     10     11     12     13
    row03     18     19     20     21
    


```python
# 범위를 지정(slicing)하여 행 여러 개 선택
print('라벨기반 - \n', sample_frm.loc['row01' : 'row03'])
print('정수기반 - \n', sample_frm.iloc[0:2])
```

    라벨기반 - 
            col01  col02  col03  col04
    row01     10     11     12     13
    row02     14     15     16     17
    row03     18     19     20     21
    정수기반 - 
            col01  col02  col03  col04
    row01     10     11     12     13
    row02     14     15     16     17
    


```python
sample_frm['col01'] > 15
```




    row01    False
    row02    False
    row03     True
    Name: col01, dtype: bool




```python
sample_frm.loc[sample_frm['col01'] > 15 ]
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
      <th>col01</th>
      <th>col02</th>
      <th>col03</th>
      <th>col04</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>row03</th>
      <td>18</td>
      <td>19</td>
      <td>20</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('행 인덱스와 열 인덱스를 사용하여 행 선택 - frm.loc[행, 열]')
print('frm[열][행]')
# 문) 첫 번째 row 모든 열을 추출
print('라벨기반 - \n ', sample_frm.loc['row01', : ])
print('정수기반 - \n', sample_frm.iloc[0, : ])
print()

print('특정 컬럼을 추출 - ')
print('라벨기반 - \n ', sample_frm.loc['row01', 'col03'])
print('정수기반 - \n', sample_frm.iloc[0, [2,3] ])
```

    행 인덱스와 열 인덱스를 사용하여 행 선택 - frm.loc[행, 열]
    frm[열][행]
    라벨기반 - 
      col01    10
    col02    11
    col03    12
    col04    13
    Name: row01, dtype: int32
    정수기반 - 
     col01    10
    col02    11
    col03    12
    col04    13
    Name: row01, dtype: int32
    
    특정 컬럼을 추출 - 
    라벨기반 - 
      12
    정수기반 - 
     col03    12
    col04    13
    Name: row01, dtype: int32
    


```python
print('프레임에서 새로운 열이 아닌 행을 추가한다면 - ')
sample_frm.loc['row04'] = 0
frmInfo(sample_frm)
```

    프레임에서 새로운 열이 아닌 행을 추가한다면 - 
    shape -  (4, 4)
    size -  16
    ndim -  2
    row index -  Index(['row01', 'row02', 'row03', 'row04'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    col index -  Index(['col01', 'col02', 'col03', 'col04'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    values -  [[10 11 12 13]
     [14 15 16 17]
     [18 19 20 21]
     [ 0  0  0  0]]
    
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
      <th>col01</th>
      <th>col02</th>
      <th>col03</th>
      <th>col04</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>row01</th>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>13</td>
    </tr>
    <tr>
      <th>row02</th>
      <td>14</td>
      <td>15</td>
      <td>16</td>
      <td>17</td>
    </tr>
    <tr>
      <th>row03</th>
      <td>18</td>
      <td>19</td>
      <td>20</td>
      <td>21</td>
    </tr>
    <tr>
      <th>row04</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
print('기존 행을 복사해서 행 추가 - ')
sample_frm.loc['row05'] = sample_frm.loc['row04']
frmInfo(sample_frm)
```

    기존 행을 복사해서 행 추가
    shape -  (5, 4)
    size -  20
    ndim -  2
    row index -  Index(['row01', 'row02', 'row03', 'row04', 'row05'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    col index -  Index(['col01', 'col02', 'col03', 'col04'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    values -  [[10 11 12 13]
     [14 15 16 17]
     [18 19 20 21]
     [ 0  0  0  0]
     [ 0  0  0  0]]
    
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
      <th>col01</th>
      <th>col02</th>
      <th>col03</th>
      <th>col04</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>row01</th>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>13</td>
    </tr>
    <tr>
      <th>row02</th>
      <td>14</td>
      <td>15</td>
      <td>16</td>
      <td>17</td>
    </tr>
    <tr>
      <th>row03</th>
      <td>18</td>
      <td>19</td>
      <td>20</td>
      <td>21</td>
    </tr>
    <tr>
      <th>row04</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>row05</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
print('행 추가와 비슷하게 열 추가 - ') #열추가는 인덱서 X
sample_frm['col05'] = sample_frm['col04']
frmInfo(sample_frm)
sample_frm['col06'] = 10
frmInfo(sample_frm)
```

    행 추가와 비슷하게 열 추가 - 
    shape -  (5, 5)
    size -  25
    ndim -  2
    row index -  Index(['row01', 'row02', 'row03', 'row04', 'row05'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    col index -  Index(['col01', 'col02', 'col03', 'col04', 'col05'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    values -  [[10 11 12 13 13]
     [14 15 16 17 17]
     [18 19 20 21 21]
     [ 0  0  0  0  0]
     [ 0  0  0  0  0]]
    
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
      <th>col01</th>
      <th>col02</th>
      <th>col03</th>
      <th>col04</th>
      <th>col05</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>row01</th>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>13</td>
      <td>13</td>
    </tr>
    <tr>
      <th>row02</th>
      <td>14</td>
      <td>15</td>
      <td>16</td>
      <td>17</td>
      <td>17</td>
    </tr>
    <tr>
      <th>row03</th>
      <td>18</td>
      <td>19</td>
      <td>20</td>
      <td>21</td>
      <td>21</td>
    </tr>
    <tr>
      <th>row04</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>row05</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


    shape -  (5, 6)
    size -  30
    ndim -  2
    row index -  Index(['row01', 'row02', 'row03', 'row04', 'row05'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    col index -  Index(['col01', 'col02', 'col03', 'col04', 'col05', 'col06'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    values -  [[10 11 12 13 13 10]
     [14 15 16 17 17 10]
     [18 19 20 21 21 10]
     [ 0  0  0  0  0 10]
     [ 0  0  0  0  0 10]]
    
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
      <th>col01</th>
      <th>col02</th>
      <th>col03</th>
      <th>col04</th>
      <th>col05</th>
      <th>col06</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>row01</th>
      <td>10</td>
      <td>11</td>
      <td>12</td>
      <td>13</td>
      <td>13</td>
      <td>10</td>
    </tr>
    <tr>
      <th>row02</th>
      <td>14</td>
      <td>15</td>
      <td>16</td>
      <td>17</td>
      <td>17</td>
      <td>10</td>
    </tr>
    <tr>
      <th>row03</th>
      <td>18</td>
      <td>19</td>
      <td>20</td>
      <td>21</td>
      <td>21</td>
      <td>10</td>
    </tr>
    <tr>
      <th>row04</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>row05</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



```python
score_data = {
    'kor': [90, 80, 79, 96, 85],
    'eng': [88, 78, 85, 93, 92],
    'math': [84, 95, 85, 94, 93]
}
score_frm = pd.DataFrame(score_data,
                        index = ['문승환', '최진형', '오한샘', '조용일', '김가영'])
score_frm
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
      <th>kor</th>
      <th>eng</th>
      <th>math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>문승환</th>
      <td>90</td>
      <td>88</td>
      <td>84</td>
    </tr>
    <tr>
      <th>최진형</th>
      <td>80</td>
      <td>78</td>
      <td>95</td>
    </tr>
    <tr>
      <th>오한샘</th>
      <td>79</td>
      <td>85</td>
      <td>85</td>
    </tr>
    <tr>
      <th>조용일</th>
      <td>96</td>
      <td>93</td>
      <td>94</td>
    </tr>
    <tr>
      <th>김가영</th>
      <td>85</td>
      <td>92</td>
      <td>93</td>
    </tr>
  </tbody>
</table>
</div>



위 데이터를 활용한 실습


```python
#- 모든 학생의 각 과목 평균 점수를 새로운 열(average)로 추가
score_frm['average'] = np.mean(score_frm, axis = 1).round(2)
score_frm
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
      <th>kor</th>
      <th>eng</th>
      <th>math</th>
      <th>average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>문승환</th>
      <td>90</td>
      <td>88</td>
      <td>84</td>
      <td>87.33</td>
    </tr>
    <tr>
      <th>최진형</th>
      <td>80</td>
      <td>78</td>
      <td>95</td>
      <td>84.33</td>
    </tr>
    <tr>
      <th>오한샘</th>
      <td>79</td>
      <td>85</td>
      <td>85</td>
      <td>83.00</td>
    </tr>
    <tr>
      <th>조용일</th>
      <td>96</td>
      <td>93</td>
      <td>94</td>
      <td>94.33</td>
    </tr>
    <tr>
      <th>김가영</th>
      <td>85</td>
      <td>92</td>
      <td>93</td>
      <td>90.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
#- 최진형 학생의 영어 점수를 90점으로 수정하고 평균 점수도 다시 계산
score_frm.loc['최진형', 'eng']= 90
score_frm

score_frm['average'] = np.mean(score_frm[['kor','eng','math']], axis=1).round(2)
score_frm
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
      <th>kor</th>
      <th>eng</th>
      <th>math</th>
      <th>average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>문승환</th>
      <td>90</td>
      <td>88</td>
      <td>84</td>
      <td>87.33</td>
    </tr>
    <tr>
      <th>최진형</th>
      <td>80</td>
      <td>90</td>
      <td>95</td>
      <td>88.33</td>
    </tr>
    <tr>
      <th>오한샘</th>
      <td>79</td>
      <td>85</td>
      <td>85</td>
      <td>83.00</td>
    </tr>
    <tr>
      <th>조용일</th>
      <td>96</td>
      <td>93</td>
      <td>94</td>
      <td>94.33</td>
    </tr>
    <tr>
      <th>김가영</th>
      <td>85</td>
      <td>92</td>
      <td>93</td>
      <td>90.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
#- 조용일 학생의 점수를 데이터 프레임으로 추출
score_frm.loc[['조용일']]
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
      <th>kor</th>
      <th>eng</th>
      <th>math</th>
      <th>average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>조용일</th>
      <td>96</td>
      <td>93</td>
      <td>94</td>
      <td>94.33</td>
    </tr>
  </tbody>
</table>
</div>




```python
#- 문승환 학생의 점수를 시리즈 타입으로 추출
score_frm.loc['문승환']
```




    kor        90.00
    eng        88.00
    math       84.00
    average    87.33
    Name: 문승환, dtype: float64




```python
#- 오한샘 학생의 국,수,영 점수를 100으로 수정하고 평균 점수도 다시 계산
score_frm.loc['오한샘', : ] = 100
score_frm['average'] = np.mean(score_frm[['kor','eng','math']], axis=1).round(2)
score_frm
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
      <th>kor</th>
      <th>eng</th>
      <th>math</th>
      <th>average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>문승환</th>
      <td>90</td>
      <td>88</td>
      <td>84</td>
      <td>87.33</td>
    </tr>
    <tr>
      <th>최진형</th>
      <td>80</td>
      <td>90</td>
      <td>95</td>
      <td>88.33</td>
    </tr>
    <tr>
      <th>오한샘</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>조용일</th>
      <td>96</td>
      <td>93</td>
      <td>94</td>
      <td>94.33</td>
    </tr>
    <tr>
      <th>김가영</th>
      <td>85</td>
      <td>92</td>
      <td>93</td>
      <td>90.00</td>
    </tr>
  </tbody>
</table>
</div>



- Series and DataFrame: count()
- Series: value_counts()


```python
tmp_series = pd.Series(range(10))
print('count - ', tmp_series.count())
print('len - ', len(tmp_series))
print()

tmp_series[5] = np.NaN
print('data - ', tmp_series)
print('count - ', tmp_series.count() )
print('len - ', len(tmp_series))
```

    count -  10
    len -  10
    
    data -  0    0.0
    1    1.0
    2    2.0
    3    3.0
    4    4.0
    5    NaN
    6    6.0
    7    7.0
    8    8.0
    9    9.0
    dtype: float64
    count -  9
    len -  10
    


```python
sample_frm = pd.DataFrame(np.random.randint(5, size=(4,4)))
sample_frm
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('dataframe count - \n', sample_frm.count())
print('shape - ', sample_frm.shape)
print('dataframe len - \n', len(sample_frm))
print()

sample_frm.iloc[1,0] = np.NaN
sample_frm.iloc[3,0] = np.NaN
sample_frm.iloc[2,3] = np.NaN
display(sample_frm)
print()
print('dataframe count - \n ', sample_frm.count())
print('shape - ', sample_frm.shape)
print('dataframe len - \n ', len(sample_frm))
```

    dataframe count - 
     0    4
    1    4
    2    4
    3    4
    dtype: int64
    shape -  (4, 4)
    dataframe len - 
     4
    
    


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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>3</td>
      <td>3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>3</td>
      <td>2</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>


    
    dataframe count - 
      0    2
    1    4
    2    4
    3    3
    dtype: int64
    shape -  (4, 4)
    dataframe len - 
      4
    


```python
sample_frm[2].value_counts()
```

- 타이타닉 데이터를 활용한 실습


```python
import seaborn as sns
titanic_dataset = sns.load_dataset('titanic')
print('type - ', type(titanic_dataset))
print()
display(titanic_dataset.head())
print(titanic_dataset.info())
```

    type -  <class 'pandas.core.frame.DataFrame'>
    
    


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
    </tr>
  </tbody>
</table>
</div>


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 15 columns):
     #   Column       Non-Null Count  Dtype   
    ---  ------       --------------  -----   
     0   survived     891 non-null    int64   
     1   pclass       891 non-null    int64   
     2   sex          891 non-null    object  
     3   age          714 non-null    float64 
     4   sibsp        891 non-null    int64   
     5   parch        891 non-null    int64   
     6   fare         891 non-null    float64 
     7   embarked     889 non-null    object  
     8   class        891 non-null    category
     9   who          891 non-null    object  
     10  adult_male   891 non-null    bool    
     11  deck         203 non-null    category
     12  embark_town  889 non-null    object  
     13  alive        891 non-null    object  
     14  alone        891 non-null    bool    
    dtypes: bool(2), category(2), float64(2), int64(4), object(5)
    memory usage: 80.7+ KB
    None
    


```python
titanic_dataset.count()
```




    survived       891
    pclass         891
    sex            891
    age            714
    sibsp          891
    parch          891
    fare           891
    embarked       889
    class          891
    who            891
    adult_male     891
    deck           203
    embark_town    889
    alive          891
    alone          891
    dtype: int64




```python
titanic_dataset['class'].value_counts()
```




    Third     491
    First     216
    Second    184
    Name: class, dtype: int64




```python
titanic_dataset['new_col'] = 0
del titanic_dataset['new_col']
```


```python
# 문1 age 컬럼의 값에 10살을 더한 값으로 age_by_10 컬럼 추가
# 문2 parch와 sibsp 더한 값에 1을 더해서 family_no 컬럼 추가
# 문3 새로 만든 두 개의 컬럼을 삭제- drop 원본에 적용
```


```python
titanic_dataset['age_by_10'] = titanic_dataset['age'] + 10
titanic_dataset.head()

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
      <th>age_by_10</th>
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
      <td>32.0</td>
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
      <td>48.0</td>
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
      <td>36.0</td>
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
      <td>45.0</td>
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
      <td>45.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic_dataset['family_no'] = titanic_dataset['parch'] + titanic_dataset['sibsp'] +1
titanic_dataset.head()

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
      <th>age_by_10</th>
      <th>family_no</th>
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
      <td>32.0</td>
      <td>2</td>
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
      <td>48.0</td>
      <td>2</td>
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
      <td>36.0</td>
      <td>1</td>
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
      <td>45.0</td>
      <td>2</td>
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
      <td>45.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# drop(label = , axis = 0(행) 1(열), inplace = )
titanic_dataset.drop(['age_by_10', 'family_no'], axis =1, inplace = True)
titanic_dataset.head()
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
    </tr>
  </tbody>
</table>
</div>




```python
titanic_dataset.index[:10].values
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int64)




```python
print('요금(fare)에 대한 통계 정보 확인(최대,최소,평균,합계)')
fare_series = titanic_dataset['fare']
print('max - ', fare_series.values.max())
print('min - ', fare_series.values.min())
print('mean - ', fare_series.values.mean())
print('sum - ', fare_series.values.sum())
```

    요금(fare)에 대한 통계 정보 확인(최대,최소,평균,합계)
    max -  512.3292
    min -  0.0
    mean -  32.204207968574636
    sum -  28693.9493
    


```python
print('선실등급(pclass == 3) 데이터만 추출하고 싶다면')
```


```python
display(titanic_dataset[titanic_dataset['pclass']==3])

print(titanic_dataset[titanic_dataset['pclass']==3]['pclass'].value_counts())
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
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>8.4583</td>
      <td>Q</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Queenstown</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>21.0750</td>
      <td>S</td>
      <td>Third</td>
      <td>child</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
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
    </tr>
    <tr>
      <th>882</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>22.0</td>
      <td>0</td>
      <td>0</td>
      <td>10.5167</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>884</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.0500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>885</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>39.0</td>
      <td>0</td>
      <td>5</td>
      <td>29.1250</td>
      <td>Q</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Queenstown</td>
      <td>no</td>
      <td>False</td>
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
    </tr>
  </tbody>
</table>
<p>491 rows × 15 columns</p>
</div>


    3    491
    Name: pclass, dtype: int64
    


```python
print('선실등급(pclass==3) sex, who 데이터만 추출하고 싶다면 - ')
display(titanic_dataset[titanic_dataset['pclass']==3][['sex','who']].head())
```

    선실등급(pclass==3) sex, who 데이터만 추출하고 싶다면 - 
    


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
      <th>who</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>male</td>
      <td>man</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>woman</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>man</td>
    </tr>
    <tr>
      <th>5</th>
      <td>male</td>
      <td>man</td>
    </tr>
    <tr>
      <th>7</th>
      <td>male</td>
      <td>child</td>
    </tr>
  </tbody>
</table>
</div>



```python
pclass_subset_frm = titanic_dataset[titanic_dataset['pclass']==3]
pclass_subset_frm.head()
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
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>8.4583</td>
      <td>Q</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Queenstown</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>21.0750</td>
      <td>S</td>
      <td>Third</td>
      <td>child</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



- set_index(): 특정 열을 데이터프레임의 행 인덱스로 설정
- reset_index(): 원복


```python
pclass_subset_frm.reset_index(inplace=True)
pclass_subset_frm
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
      <th>index</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
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
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
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
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
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
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>8.4583</td>
      <td>Q</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Queenstown</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>21.0750</td>
      <td>S</td>
      <td>Third</td>
      <td>child</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
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
      <th>486</th>
      <td>882</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>22.0</td>
      <td>0</td>
      <td>0</td>
      <td>10.5167</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>487</th>
      <td>884</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.0500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>488</th>
      <td>885</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>39.0</td>
      <td>0</td>
      <td>5</td>
      <td>29.1250</td>
      <td>Q</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Queenstown</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>489</th>
      <td>888</td>
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
    </tr>
    <tr>
      <th>490</th>
      <td>890</td>
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
    </tr>
  </tbody>
</table>
<p>491 rows × 16 columns</p>
</div>




```python
pclass_subset_frm.set_index('index', inplace = True)
pclass_subset_frm
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
    </tr>
    <tr>
      <th>index</th>
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
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>8.4583</td>
      <td>Q</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Queenstown</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>21.0750</td>
      <td>S</td>
      <td>Third</td>
      <td>child</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
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
    </tr>
    <tr>
      <th>882</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>22.0</td>
      <td>0</td>
      <td>0</td>
      <td>10.5167</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>884</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.0500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>885</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>39.0</td>
      <td>0</td>
      <td>5</td>
      <td>29.1250</td>
      <td>Q</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Queenstown</td>
      <td>no</td>
      <td>False</td>
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
    </tr>
  </tbody>
</table>
<p>491 rows × 15 columns</p>
</div>




```python
pclass_subset_frm.reset_index(inplace=True)
pclass_subset_frm.set_index(['index','pclass'], inplace = True)
pclass_subset_frm
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
      <th>survived</th>
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
    </tr>
    <tr>
      <th>index</th>
      <th>pclass</th>
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
      <th>0</th>
      <th>3</th>
      <td>0</td>
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
    </tr>
    <tr>
      <th>2</th>
      <th>3</th>
      <td>1</td>
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
    </tr>
    <tr>
      <th>4</th>
      <th>3</th>
      <td>0</td>
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
    </tr>
    <tr>
      <th>5</th>
      <th>3</th>
      <td>0</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>8.4583</td>
      <td>Q</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Queenstown</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <th>3</th>
      <td>0</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>21.0750</td>
      <td>S</td>
      <td>Third</td>
      <td>child</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
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
    </tr>
    <tr>
      <th>882</th>
      <th>3</th>
      <td>0</td>
      <td>female</td>
      <td>22.0</td>
      <td>0</td>
      <td>0</td>
      <td>10.5167</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>884</th>
      <th>3</th>
      <td>0</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.0500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>885</th>
      <th>3</th>
      <td>0</td>
      <td>female</td>
      <td>39.0</td>
      <td>0</td>
      <td>5</td>
      <td>29.1250</td>
      <td>Q</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Queenstown</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>888</th>
      <th>3</th>
      <td>0</td>
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
    </tr>
    <tr>
      <th>890</th>
      <th>3</th>
      <td>0</td>
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
    </tr>
  </tbody>
</table>
<p>491 rows × 14 columns</p>
</div>




```python
pclass_subset_frm.index
```




    MultiIndex([(  0, 3),
                (  2, 3),
                (  4, 3),
                (  5, 3),
                (  7, 3),
                (  8, 3),
                ( 10, 3),
                ( 12, 3),
                ( 13, 3),
                ( 14, 3),
                ...
                (875, 3),
                (876, 3),
                (877, 3),
                (878, 3),
                (881, 3),
                (882, 3),
                (884, 3),
                (885, 3),
                (888, 3),
                (890, 3)],
               names=['index', 'pclass'], length=491)




```python
titanic_dataset.head()
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
    </tr>
  </tbody>
</table>
</div>



요구사항4
- 나이가 60보다 크고 선실등급이 1등급이고 성별이 여자인 데이터만 추출


```python
titanic_dataset[(titanic_dataset['age'] > 60) &
                (titanic_dataset['pclass'] == 1) & 
                (titanic_dataset['sex'] == 'female')]
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>275</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>63.0</td>
      <td>1</td>
      <td>0</td>
      <td>77.9583</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>D</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>829</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>80.0000</td>
      <td>NaN</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>B</td>
      <td>NaN</td>
      <td>yes</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



요구사항5


```python
#- 승객의 평균나이
print(int(titanic_dataset['age'].mean()))

#- 여성승객의 평균나이
print(titanic_dataset['age'][titanic_dataset['sex']=='female'].mean())
print(titanic_dataset.loc[titanic_dataset['sex']=='female', 'age'].mean())

#- 1등급 선실의 여성승객의 평균나이
print(titanic_dataset['age'][(titanic_dataset['sex']=='female') & (titanic_dataset['pclass']==1)].mean())
print(titanic_dataset.loc[(titanic_dataset['pclass'] == 1)&
                         (titanic_dataset['sex'] == 'female') , 'age'].mean())
```

    29
    27.915708812260537
    27.915708812260537
    34.61176470588235
    34.61176470588235
    

정렬
- sort_index(axis = , ascending = )
- sort_values(by = , ascending = ) : 열 값을 기준으로 행 정렬

요구사항 6


```python
print('타이타닉에서 승객의 나이를 기준으로 내림차순 정렬 - ')
display(titanic_dataset.sort_values(by = 'age', ascending = False))

titanic_age_subset_frm = titanic_dataset.sort_values(by = 'age', ascending = False)
titanic_age_subset_frm.reset_index(inplace = True)
display(titanic_age_subset_frm)
print()

titanic_age_subset_frm.drop('index', axis = 1, inplace = True)
display(titanic_age_subset_frm)
```

    타이타닉에서 승객의 나이를 기준으로 내림차순 정렬 - 
    


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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>630</th>
      <td>1</td>
      <td>1</td>
      <td>male</td>
      <td>80.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>S</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>A</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
    </tr>
    <tr>
      <th>851</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>74.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7750</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>493</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>71.0</td>
      <td>0</td>
      <td>0</td>
      <td>49.5042</td>
      <td>C</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Cherbourg</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>71.0</td>
      <td>0</td>
      <td>0</td>
      <td>34.6542</td>
      <td>C</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>A</td>
      <td>Cherbourg</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>116</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>70.5</td>
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
    </tr>
    <tr>
      <th>859</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.2292</td>
      <td>C</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Cherbourg</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>863</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>69.5500</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>868</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>9.5000</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>878</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.8958</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
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
    </tr>
  </tbody>
</table>
<p>891 rows × 15 columns</p>
</div>



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
      <th>index</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>630</td>
      <td>1</td>
      <td>1</td>
      <td>male</td>
      <td>80.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>S</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>A</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>851</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>74.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7750</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>493</td>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>71.0</td>
      <td>0</td>
      <td>0</td>
      <td>49.5042</td>
      <td>C</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Cherbourg</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>96</td>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>71.0</td>
      <td>0</td>
      <td>0</td>
      <td>34.6542</td>
      <td>C</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>A</td>
      <td>Cherbourg</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>116</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>70.5</td>
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
      <td>859</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.2292</td>
      <td>C</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Cherbourg</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>887</th>
      <td>863</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>69.5500</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>888</th>
      <td>868</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>9.5000</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>889</th>
      <td>878</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.8958</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>890</th>
      <td>888</td>
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
    </tr>
  </tbody>
</table>
<p>891 rows × 16 columns</p>
</div>


    
    


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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>male</td>
      <td>80.0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0000</td>
      <td>S</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>A</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>74.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7750</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>71.0</td>
      <td>0</td>
      <td>0</td>
      <td>49.5042</td>
      <td>C</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Cherbourg</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>71.0</td>
      <td>0</td>
      <td>0</td>
      <td>34.6542</td>
      <td>C</td>
      <td>First</td>
      <td>man</td>
      <td>True</td>
      <td>A</td>
      <td>Cherbourg</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>70.5</td>
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
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.2292</td>
      <td>C</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Cherbourg</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>887</th>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>NaN</td>
      <td>8</td>
      <td>2</td>
      <td>69.5500</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>9.5000</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>889</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>7.8958</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
    <tr>
      <th>890</th>
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
    </tr>
  </tbody>
</table>
<p>891 rows × 15 columns</p>
</div>


요구사항 7


```python
print('타이타닉에서 성별의 인원수 내림차순 정렬 - ', titanic_dataset['sex'].value_counts().sort_values(ascending=False))
print('타이타닉에서 나이별 인원수 내림차순 정렬 - ',titanic_dataset['age'].value_counts().sort_values(ascending=False))
print('타이타닉에서 선실별 인원수 내림차순 정렬 - ',titanic_dataset['pclass'].value_counts().sort_values(ascending=False))
print('타이타닉에서 사망/생존별 인원수 내림차순 정렬 - ', titanic_dataset['alive'].value_counts().sort_values(ascending=False))
```

    타이타닉에서 성별의 인원수 내림차순 정렬 -  male      577
    female    314
    Name: sex, dtype: int64
    타이타닉에서 나이별 인원수 내림차순 정렬 -  24.00    30
    22.00    27
    18.00    26
    19.00    25
    28.00    25
             ..
    66.00     1
    0.67      1
    0.42      1
    34.50     1
    74.00     1
    Name: age, Length: 88, dtype: int64
    타이타닉에서 선실별 인원수 내림차순 정렬 -  3    491
    1    216
    2    184
    Name: pclass, dtype: int64
    타이타닉에서 사망/생존별 인원수 내림차순 정렬 -  no     549
    yes    342
    Name: alive, dtype: int64
    

요구사항 8


```python
print('승객 나이를 기준으로 내림차순 정렬하고 순위를 부여한다면 - ')
titanic_dataset.sort_values(by = 'age', ascending = False)['age'].rank(ascending = False)
```

    승객 나이를 기준으로 내림차순 정렬하고 순위를 부여한다면 - 
    




    630    1.0
    851    2.0
    493    3.5
    96     3.5
    116    5.0
          ... 
    859    NaN
    863    NaN
    868    NaN
    878    NaN
    888    NaN
    Name: age, Length: 891, dtype: float64




```python
titanic_dataset.sort_values(by = 'rank', ascending = True).head()
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    Input In [136], in <cell line: 1>()
    ----> 1 titanic_dataset.sort_values(by = 'rank', ascending = True).head()
    

    File ~\anaconda3\lib\site-packages\pandas\util\_decorators.py:311, in deprecate_nonkeyword_arguments.<locals>.decorate.<locals>.wrapper(*args, **kwargs)
        305 if len(args) > num_allow_args:
        306     warnings.warn(
        307         msg.format(arguments=arguments),
        308         FutureWarning,
        309         stacklevel=stacklevel,
        310     )
    --> 311 return func(*args, **kwargs)
    

    File ~\anaconda3\lib\site-packages\pandas\core\frame.py:6313, in DataFrame.sort_values(self, by, axis, ascending, inplace, kind, na_position, ignore_index, key)
       6309 elif len(by):
       6310     # len(by) == 1
       6312     by = by[0]
    -> 6313     k = self._get_label_or_level_values(by, axis=axis)
       6315     # need to rewrap column in Series to apply key function
       6316     if key is not None:
       6317         # error: Incompatible types in assignment (expression has type
       6318         # "Series", variable has type "ndarray")
    

    File ~\anaconda3\lib\site-packages\pandas\core\generic.py:1840, in NDFrame._get_label_or_level_values(self, key, axis)
       1838     values = self.axes[axis].get_level_values(key)._values
       1839 else:
    -> 1840     raise KeyError(key)
       1842 # Check for duplicates
       1843 if values.ndim > 1:
    

    KeyError: 'rank'



```python
sample_frm = pd.read_csv('./data/pandas_data/year2020_baby_name.csv',
                        sep = ',')
print('type - ', type(sample_frm))
print(sample_frm.info())
display(sample_frm.head())
```

    type -  <class 'pandas.core.frame.DataFrame'>
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 33838 entries, 0 to 33837
    Data columns (total 3 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   NAME    33838 non-null  object
     1   GENDER  33838 non-null  object
     2   COUNT   33838 non-null  int64 
    dtypes: int64(1), object(2)
    memory usage: 793.2+ KB
    None
    


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
      <th>NAME</th>
      <th>GENDER</th>
      <th>COUNT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Isabella</td>
      <td>F</td>
      <td>22731</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sophia</td>
      <td>F</td>
      <td>20477</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Emma</td>
      <td>F</td>
      <td>17179</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Olivia</td>
      <td>F</td>
      <td>16860</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ava</td>
      <td>F</td>
      <td>15300</td>
    </tr>
  </tbody>
</table>
</div>



```python
sort_frm = sample_frm.sort_values(by='COUNT', ascending=False)
sort_frm.head()
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
      <th>NAME</th>
      <th>GENDER</th>
      <th>COUNT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Isabella</td>
      <td>F</td>
      <td>22731</td>
    </tr>
    <tr>
      <th>19698</th>
      <td>Jacob</td>
      <td>M</td>
      <td>21875</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sophia</td>
      <td>F</td>
      <td>20477</td>
    </tr>
    <tr>
      <th>19699</th>
      <td>Ethan</td>
      <td>M</td>
      <td>17866</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Emma</td>
      <td>F</td>
      <td>17179</td>
    </tr>
  </tbody>
</table>
</div>




```python
sort_frm.reset_index(inplace = True)
sort_frm.drop(['index'], axis = 1, inplace = True)
sort_frm
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
      <th>NAME</th>
      <th>GENDER</th>
      <th>COUNT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Isabella</td>
      <td>F</td>
      <td>22731</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jacob</td>
      <td>M</td>
      <td>21875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sophia</td>
      <td>F</td>
      <td>20477</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ethan</td>
      <td>M</td>
      <td>17866</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Emma</td>
      <td>F</td>
      <td>17179</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>33833</th>
      <td>Mccauley</td>
      <td>F</td>
      <td>5</td>
    </tr>
    <tr>
      <th>33834</th>
      <td>Mazal</td>
      <td>F</td>
      <td>5</td>
    </tr>
    <tr>
      <th>33835</th>
      <td>Mayzee</td>
      <td>F</td>
      <td>5</td>
    </tr>
    <tr>
      <th>33836</th>
      <td>Maythe</td>
      <td>F</td>
      <td>5</td>
    </tr>
    <tr>
      <th>33837</th>
      <td>Zzyzx</td>
      <td>M</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>33838 rows × 3 columns</p>
</div>




```python
m_gender_frm = sort_frm[sort_frm['GENDER'] == 'M']
f_gender_frm = sort_frm[sort_frm['GENDER'] == 'F']
```


```python
m_gender_frm.head()
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
      <th>NAME</th>
      <th>GENDER</th>
      <th>COUNT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Jacob</td>
      <td>M</td>
      <td>21875</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ethan</td>
      <td>M</td>
      <td>17866</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Michael</td>
      <td>M</td>
      <td>17133</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Jayden</td>
      <td>M</td>
      <td>17030</td>
    </tr>
    <tr>
      <th>7</th>
      <td>William</td>
      <td>M</td>
      <td>16870</td>
    </tr>
  </tbody>
</table>
</div>




```python
f_gender_frm.head()
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
      <th>NAME</th>
      <th>GENDER</th>
      <th>COUNT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Isabella</td>
      <td>F</td>
      <td>22731</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sophia</td>
      <td>F</td>
      <td>20477</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Emma</td>
      <td>F</td>
      <td>17179</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Olivia</td>
      <td>F</td>
      <td>16860</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Ava</td>
      <td>F</td>
      <td>15300</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
