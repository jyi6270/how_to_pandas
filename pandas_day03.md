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

파일입출력과 병합, 문자열 관련된 데이터 조작
- csv, excel, json, html, scraping 비정형 파일의 종류
- read_xxxx(), to_xxxx()


```python
file_path = './data/pandas_data/read_csv_sample.csv'
sample_frm = pd.read_csv(file_path)
print('type - ', type(sample_frm))
frmInfo(sample_frm)
```

    type -  <class 'pandas.core.frame.DataFrame'>
    shape -  (3, 5)
    size -  15
    ndim -  2
    row index -  RangeIndex(start=0, stop=3, step=1) <class 'pandas.core.indexes.range.RangeIndex'>
    col index -  Index(['c0', 'c1', 'c2', 'c3', 'c4'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    values - 
     [[0 1 4 7 6]
     [1 2 5 8 4]
     [2 3 6 9 2]] <class 'numpy.ndarray'>
    
    data - 
    





```python
sample_frm = pd.read_csv(file_path, index_col = 'c4')
print('type - ', type(sample_frm))
frmInfo(sample_frm)
```

    type -  <class 'pandas.core.frame.DataFrame'>
    shape -  (3, 4)
    size -  12
    ndim -  2
    row index -  Int64Index([6, 4, 2], dtype='int64', name='c4') <class 'pandas.core.indexes.numeric.Int64Index'>
    col index -  Index(['c0', 'c1', 'c2', 'c3'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    values - 
     [[0 1 4 7]
     [1 2 5 8]
     [2 3 6 9]] <class 'numpy.ndarray'>
    
    data - 
    





```python
file_path = './data/pandas_data/남북한발전전력량.xlsx'
sample_frm = pd.read_excel(file_path)
print('type - ', type(sample_frm))
frmInfo(sample_frm)
```

    type -  <class 'pandas.core.frame.DataFrame'>
    shape -  (9, 29)
    size -  261
    ndim -  2
    row index -  RangeIndex(start=0, stop=9, step=1) <class 'pandas.core.indexes.range.RangeIndex'>
    col index -  Index(['전력량 (억㎾h)', '발전 전력별', '1990', '1991', '1992', '1993', '1994', '1995',
           '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
           '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
           '2014', '2015', '2016'],
          dtype='object') <class 'pandas.core.indexes.base.Index'>
    values - 
     [['남한' '합계' 1077 1186 1310 1444 1650 1847 2055 2244 2153 2393 2664 2852
      3065 3225 3421 3646 3812 4031 4224 4336 4747 4969 5096 5171 5220 5281
      5404]
     [nan '수력' 64 51 49 60 41 55 52 54 61 61 56 42 53 69 59 52 52 50 56 56 65
      78 77 84 78 58 66]
     [nan '화력' 484 573 696 803 1022 1122 1264 1420 1195 1302 1518 1689 1821
      1859 2056 2127 2272 2551 2658 2802 3196 3343 3430 3581 3427 3402 3523]
     [nan '원자력' 529 563 565 581 587 670 739 771 897 1031 1090 1121 1191 1297
      1307 1468 1487 1429 1510 1478 1486 1547 1503 1388 1564 1648 1620]
     [nan '신재생' '-' '-' '-' '-' '-' '-' '-' '-' '-' '-' '-' '-' '-' '-' '-'
      '-' '-' '-' '-' '-' '-' '-' 86 118 151 173 195]
     ['북한' '합계' 277 263 247 221 231 230 213 193 170 186 194 202 190 196 206
      215 225 236 255 235 237 211 215 221 216 190 239]
     [nan '수력' 156 150 142 133 138 142 125 107 102 103 102 106 106 117 125
      131 126 133 141 125 134 132 135 139 130 100 128]
     [nan '화력' 121 113 105 88 93 88 88 86 68 83 92 96 84 79 81 84 99 103 114
      110 103 79 80 82 86 90 111]
     [nan '원자력' '-' '-' '-' '-' '-' '-' '-' '-' '-' '-' '-' '-' '-' '-' '-'
      '-' '-' '-' '-' '-' '-' '-' '-' '-' '-' '-' '-']] <class 'numpy.ndarray'>
    
    data - 
    




```python
file_path = './data/pandas_data/read_json_sample.json'
sample_frm = pd.read_json(file_path)
print('type - ', type(sample_frm))
frmInfo(sample_frm)
```

    type -  <class 'pandas.core.frame.DataFrame'>
    shape -  (3, 4)
    size -  12
    ndim -  2
    row index -  Index(['pandas', 'NumPy', 'matplotlib'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    col index -  Index(['name', 'year', 'developer', 'opensource'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    values - 
     [['' 2008 'Wes Mckinneye' 'True']
     ['' 2006 'Travis Oliphant' 'True']
     ['' 2003 'John D. Hunter' 'True']] <class 'numpy.ndarray'>
    
    data - 
    




```python
file_path = './data/pandas_data/sample.html'
sample_tables = pd.read_html(file_path)
print('type - ', type(sample_tables))
print('len - ', len(sample_tables))
print()
for idx in range(len(sample_tables)):
    print('table index - ', idx)
    print(sample_tables[idx])

sample_frm = sample_tables[1]
print('type - ', type(sample_frm))
sample_frm.set_index('name', inplace = True)
print(sample_frm)
```

    type -  <class 'list'>
    len -  2
    
    table index -  0
       Unnamed: 0  c0  c1  c2  c3
    0           0   0   1   4   7
    1           1   1   2   5   8
    2           2   2   3   6   9
    table index -  1
             name  year        developer  opensource
    0       NumPy  2006  Travis Oliphant        True
    1  matplotlib  2003   John D. Hunter        True
    2      pandas  2008    Wes Mckinneye        True
    

- scraping 통한 데이터 로딩
- BeautifulSoup(정적페이지)


```python
from bs4 import BeautifulSoup
import requests
import re
```


```python
response = requests.get('https://en.wikipedia.org/wiki/List_of_American_exchange-traded_funds')
print('response code - ', response)
print()
print('response - ', response.text)
```

    


```python
soup = BeautifulSoup(response.text)
soup
```





```python
soup.select('div > ul > li')
```




```python
print('type - ', type(soup.select('div > ul > li')))

lst = soup.select('div > ul > li')
for row in lst :
#     print(row.text)
    etf_name = re.findall('^(.*) \(NYSE', row.text)
#     print(etf_name)
#     print()
    etf_market = re.findall('NYSE Arca\|(.*)\)', row.text)
    print(etf_market)
```


- google 지오코딩 API
- conda install -c conda-forge googlemaps


```python
import googlemaps
maps = googlemaps.Client(key = 'AIzaSyDdki495w62u7ziBMaVuzyYf4n-VZ8tsGA')
location = maps.geocode('서울시청')
print('location - ', type(location))
print()
print('location - ', location[0].get('geometry'))
```

    
     API queries_quota: 60 
    
    location -  <class 'list'>
    
    location -  {'bounds': {'northeast': {'lat': 37.7017495, 'lng': 127.1835899}, 'southwest': {'lat': 37.4259627, 'lng': 126.7645827}}, 'location': {'lat': 37.566535, 'lng': 126.9779692}, 'location_type': 'APPROXIMATE', 'viewport': {'northeast': {'lat': 37.7017495, 'lng': 127.1835899}, 'southwest': {'lat': 37.4259627, 'lng': 126.7645827}}}
    

실습
- 문) places = ['서울시청', '해운대해수욕장', '국립대전현충원', '한라산', '인천국제공항']
- 제공된 위치에 해당하는 위도, 경도를 추출하고 데이터 프레임을 작성해보자


```python
places = ['서울시청', '해운대해수욕장', '국립대전현충원', '한라산', '인천국제공항']
lat = []
lng = []

for place in places :
    print(place)
    location = maps.geocode(place)
    lat.append(location[0].get('geometry')['location']['lat'])
    lng.append(location[0].get('geometry')['location']['lng'])
print()
print('lat - ', lat)
print('lng - ', lng)
```

    서울시청
    해운대해수욕장
    국립대전현충원
    한라산
    인천국제공항
    
    lat -  [37.566535, 35.1586975, 36.3686021, 33.3616666, 37.4601908]
    lng -  [126.9779692, 129.1603842, 127.296685, 126.5291666, 126.4406957]
    


```python
location_frm = pd.DataFrame({
    '위도': lat,
    '경도': lng
},
index = places)
location_frm
```






```python
location_frm.to_csv('./data/pandas_data/googlemaps.csv')
```


```python
sample_frm = pd.read_csv('./data/pandas_data/googlemaps.csv')
frmInfo(sample_frm)
```

    shape -  (5, 3)
    size -  15
    ndim -  2
    row index -  RangeIndex(start=0, stop=5, step=1) <class 'pandas.core.indexes.range.RangeIndex'>
    col index -  Index(['Unnamed: 0', '위도', '경도'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    values - 
     [['서울시청' 37.566535 126.9779692]
     ['해운대해수욕장' 35.1586975 129.1603842]
     ['국립대전현충원' 36.3686021 127.296685]
     ['한라산' 33.3616666 126.5291666]
     ['인천국제공항' 37.4601908 126.4406957]] <class 'numpy.ndarray'>
    
    data - 
    





```python
sample_frm.rename(columns={'Unnamed: 0': '위치'}, inplace = True)
```


```python
sample_frm.set_index('위치', inplace=True)
```


```python
sample_frm
```





프레임 병합
- pd.merge(): 공통의 열을 기준으로 병합
- frm.join(): 인덱스 기준으로 병합
- concat(): 연결


```python
data01 = {
    '학번': [1,2,3,4],
    '이름': ['오한샘','장수빈','김가영','조용일'],
    '학년': [2,4,1,3]
}
data02 = {
    '학번': [1,2,4,5],
    '학과': ['CS', 'AI', 'AI', 'CS'],
    '학점': [2.8, 4.5, 1.9, 3.5]
}
```


```python
stu_frm = pd.DataFrame(data01)
major_frm = pd.DataFrame(data02)
```


```python
stu_frm
```






```python
major_frm
```





```python
pd.merge(stu_frm, major_frm, how = 'inner')
```





```python
pd.merge(stu_frm, major_frm, how = 'outer')
```




- 컬럼 인덱스가 다른 경우
- 속성: left_on = , right_on = 


```python
data01 = {
    '학번': [1,2,3,4],
    '이름': ['오한샘','장수빈','김가영','조용일'],
    '학년': [2,4,1,3]
}
data02 = {
    '과목코드': [1,2,4,5],
    '학과': ['CS', 'AI', 'AI', 'CS'],
    '학점': [2.8, 4.5, 1.9, 3.5]
}
```


```python
stu_frm = pd.DataFrame(data01)
major_frm = pd.DataFrame(data02)
```


```python
pd.merge(stu_frm, major_frm, how = 'inner', left_on = '학번', right_on = '과목코드')
```






```python
import seaborn as sns
iris_frm = sns.load_dataset('iris')
print('type - ', type(iris_frm))
```

    type -  <class 'pandas.core.frame.DataFrame'>
    


```python
frmInfo(iris_frm)
```

    shape -  (150, 5)
    size -  750
    ndim -  2
    row index -  RangeIndex(start=0, stop=150, step=1) <class 'pandas.core.indexes.range.RangeIndex'>
    col index -  Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width',
           'species'],
          dtype='object') <class 'pandas.core.indexes.base.Index'>
    values - 
     [[5.1 3.5 1.4 0.2 'setosa']
     [4.9 3.0 1.4 0.2 'setosa']
     ...
     [6.2 3.4 5.4 2.3 'virginica']
     [5.9 3.0 5.1 1.8 'virginica']] <class 'numpy.ndarray'>
    
    data - 
    




```python
data01 = {
    'species': ['setosa', 'virginica', 'virginica', 'versicolor'],
    'sepal_length': [5.1, 4.9, 4.7, 4.6]
}
data02 = {
    'species': ['setosa', 'setosa', 'virginica', 'verginica'],
    'sepal_width': [3.5 , 3.0, 3.2, 3.1]
}
iris01_frm = pd.DataFrame(data01)
iris02_frm = pd.DataFrame(data02)
display(iris01_frm)
print()
display(iris02_frm)
```

    



```python
#중복되는 키값이 존재하면 how 대신 on
pd.merge(iris01_frm, iris02_frm, on='species')
```







```python
data01 = {
    'species': ['setosa', 'virginica', 'virginica', 'versicolor'],
    'sepal_length': [5.1, 4.9, 4.7, 4.6]
}
data02 = {
    'species': ['setosa', 'setosa', 'virginica', 'verginica'],
    'sepal_length': [3.5 , 3.0, 3.2, 3.1],
    'petal_width' : [3.5 , 3.0, 3.2, 3.1]
}
iris01_frm = pd.DataFrame(data01)
iris02_frm = pd.DataFrame(data02)
display(iris01_frm)
print()
display(iris02_frm)
```


    
    





```python
pd.merge(iris01_frm, iris02_frm, on=['species'])
```






- merge: 열 병합
- 인덱스를 기준으로 merge: left_index, right_index


```python
sample_frm01 = pd.DataFrame({
    'city': ['seoul', 'seoul', 'seoul', 'kwangju', 'kwangju'],
    'year': [2010,2050,2020,2018,2022],
    'pop': [100,200,300,400,500]
})
sample_frm01
```








```python
# 다중인덱스

sample_frm02 = pd.DataFrame(np.arange(12).reshape(6,2),
                           columns = ['feature01', 'feature02'],
                           index = [['seoul', 'seoul', 'kwangju', 'kwangju', 'kwangju', 'kwangju'],
                                   [2010, 2050, 2020, 2018, 2022, 2021]])
sample_frm02
```







```python
# error
# pd.merge(sample_frm01, sample_frm02, left_index= True, right_index = True)
pd.merge(sample_frm01, sample_frm02, right_index= True, left_on=['city', 'year'])

```







```python
data01 = {
    '학번': [1,2,3,4],
    '이름': ['오한샘','장수빈','김가영','조용일']
}
data02 = {
    '과목코드': [1,2,4,5],
    '학과': ['CS', 'AI', 'AI', 'CS']
}
stu_frm = pd.DataFrame(data01)
major_frm = pd.DataFrame(data02)
```


```python
pd.merge(stu_frm, major_frm, left_index = True , right_index = True)
```





```python
stu_frm.join(major_frm, how = 'inner')
```







- concat : 열 인덱스와 인덱스를 기준으로 하지않고 단순한 데이터 연결


```python
pd.concat([pd.Series([0,1]), pd.Series([2,3,4,5])])
```




    0    0
    1    1
    0    2
    1    3
    2    4
    3    5
    dtype: int64




```python
pd.concat([stu_frm, major_frm], axis = 0)
```






```python
stock_frm01 = pd.read_excel('./data/pandas_data/stock_price.xlsx')
stock_frm02 = pd.read_excel('./data/pandas_data/stock_valuation.xlsx')
```


```python
display(stock_frm01)
print()
display(stock_frm02)
```

