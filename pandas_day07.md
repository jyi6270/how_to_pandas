```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import json

import warnings
warnings.filterwarnings("ignore")

print('numpy version - ', np.__version__)
print('pandas version - ', pd.__version__)

def seriesInfo(s):
    print('index - ', s.index, type(s.index))
    print('value - ', s.values, type(s.values))
    print()
    print('data - ')
    print(s)
    
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

    numpy version -  1.21.5
    pandas version -  1.4.2
    

- 주유소 평균가격 확인
- 셀프 주유소는 정말 저렴할까?

[실습01]
지역으로 시작하는 모든 파일을 로드하고 병합



```python
from glob import glob
print('glob(): 파일경로 등을 쉽게 접근할 수 있도록 도와주는 모듈의 함수')
stations_files = glob('./data/pandas_data/oil_data/지역*.xls')
stations_files
```

    glob(): 파일경로 등을 쉽게 접근할 수 있도록 도와주는 모듈의 함수
    




    ['./data/pandas_data/oil_data\\지역_위치별(주유소) (1).xls',
     './data/pandas_data/oil_data\\지역_위치별(주유소) (10).xls',
     './data/pandas_data/oil_data\\지역_위치별(주유소) (11).xls',
     './data/pandas_data/oil_data\\지역_위치별(주유소) (12).xls',
     './data/pandas_data/oil_data\\지역_위치별(주유소) (13).xls',
     './data/pandas_data/oil_data\\지역_위치별(주유소) (14).xls',
     './data/pandas_data/oil_data\\지역_위치별(주유소) (15).xls',
     './data/pandas_data/oil_data\\지역_위치별(주유소) (16).xls',
     './data/pandas_data/oil_data\\지역_위치별(주유소) (17).xls',
     './data/pandas_data/oil_data\\지역_위치별(주유소) (18).xls',
     './data/pandas_data/oil_data\\지역_위치별(주유소) (19).xls',
     './data/pandas_data/oil_data\\지역_위치별(주유소) (2).xls',
     './data/pandas_data/oil_data\\지역_위치별(주유소) (20).xls',
     './data/pandas_data/oil_data\\지역_위치별(주유소) (21).xls',
     './data/pandas_data/oil_data\\지역_위치별(주유소) (22).xls',
     './data/pandas_data/oil_data\\지역_위치별(주유소) (23).xls',
     './data/pandas_data/oil_data\\지역_위치별(주유소) (24).xls',
     './data/pandas_data/oil_data\\지역_위치별(주유소) (3).xls',
     './data/pandas_data/oil_data\\지역_위치별(주유소) (4).xls',
     './data/pandas_data/oil_data\\지역_위치별(주유소) (5).xls',
     './data/pandas_data/oil_data\\지역_위치별(주유소) (6).xls',
     './data/pandas_data/oil_data\\지역_위치별(주유소) (7).xls',
     './data/pandas_data/oil_data\\지역_위치별(주유소) (8).xls',
     './data/pandas_data/oil_data\\지역_위치별(주유소) (9).xls',
     './data/pandas_data/oil_data\\지역_위치별(주유소).xls']




```python
tmp_raw = []
for filePath in stations_files:
    tmp = pd.read_excel(filePath, header=2)
    tmp_raw.append(tmp)
```


```python
# [실습02]
# 기본정보 확인 및 결측치 확인
stations_raw = pd.concat(tmp_raw)
stations_raw
```







```python
stations_raw.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 537 entries, 0 to 45
    Data columns (total 10 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   지역      537 non-null    object
     1   상호      537 non-null    object
     2   주소      537 non-null    object
     3   상표      537 non-null    object
     4   전화번호    537 non-null    object
     5   셀프여부    537 non-null    object
     6   고급휘발유   537 non-null    object
     7   휘발유     537 non-null    object
     8   경유      537 non-null    object
     9   실내등유    537 non-null    object
    dtypes: object(10)
    memory usage: 46.1+ KB
    

[실습03]
휘발유를 대상으로 분석 진행하기위해서 컬럼명 변경하여 서브셋 만들기

         상호,       주소,     휘발유, 셀프여부 , 상표
rename   oil_store , address , price , self_y_n , company



```python
gasoline_frm = stations_raw.loc[:,['상호', '주소', '휘발유', '셀프여부', '상표']]
gasoline_frm.rename(columns = {'상호': 'oil_store', 
                                      '주소': 'address', 
                                      '휘발유': 'price', 
                                      '셀프여부': 'self_y_n',
                                      '상표': 'company'}, inplace = True)
gasoline_frm
```





[실습04]
구별 주유소 가격을 조사하기 위해서 파생변수 생성(direct)
구별 정보확인
서울특별시 -> 성동구 대체
특별시 -> 도봉구 대체



```python
# lst = []
# for address in gasoline_frm['address']:
#     lst.append(address.split()[1])
    
# or

gasoline_frm['direct'] = [address.split()[1] for address in gasoline_frm['address']]
```


```python
gasoline_frm.head()
```







```python
gasoline_frm['direct'].unique()
```




    array(['강동구', '동대문구', '동작구', '마포구', '서대문구', '서초구', '성동구', '서울특별시', '성북구',
           '송파구', '양천구', '영등포구', '강북구', '용산구', '은평구', '종로구', '중구', '중랑구',
           '강서구', '관악구', '광진구', '구로구', '금천구', '노원구', '도봉구', '특별시', '강남구'],
          dtype=object)




```python
# 방법1
gasoline_frm.replace({'direct': '서울특별시'}, '성동구', inplace=True)
gasoline_frm.replace({'direct': '특별시'}, '도봉구', inplace=True)
```


```python
# 방법2
gasoline_frm.loc[gasoline_frm['direct'] == '서울특별시', 'direct'] = '성동구'
gasoline_frm.loc[gasoline_frm['direct'] == '특별시', 'direct'] = '도봉구'
```

[실습05]
가격 정보 확인[ - 들어있음] 후
가격 정보가 기입되지 않은 주유소는 제거
컬럼 타입을 숫자형으로 변경
인덱스를 새롭게 정의(reset_index)
인덱스열을 제거


```python
gasoline_frm.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 537 entries, 0 to 45
    Data columns (total 6 columns):
     #   Column     Non-Null Count  Dtype 
    ---  ------     --------------  ----- 
     0   oil_store  537 non-null    object
     1   address    537 non-null    object
     2   price      537 non-null    object
     3   self_y_n   537 non-null    object
     4   company    537 non-null    object
     5   direct     537 non-null    object
    dtypes: object(6)
    memory usage: 29.4+ KB
    


```python
gasoline_frm.drop(list(gasoline_frm[gasoline_frm['price']=='-'].index) , axis = 0 , inplace=True)
gasoline_frm.reset_index(drop=True , inplace=True)
gasoline_frm['price'] = gasoline_frm['price'].astype(int)
gasoline_frm.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 477 entries, 0 to 476
    Data columns (total 6 columns):
     #   Column     Non-Null Count  Dtype 
    ---  ------     --------------  ----- 
     0   oil_store  477 non-null    object
     1   address    477 non-null    object
     2   price      477 non-null    int32 
     3   self_y_n   477 non-null    object
     4   company    477 non-null    object
     5   direct     477 non-null    object
    dtypes: int32(1), object(5)
    memory usage: 20.6+ KB
    

[실습06]
구별 휘발유 평균가격 확인


```python
gasoline_frm.groupby('direct').mean()
```






```python
gasoline_pivot_frm = pd.pivot_table(gasoline_frm, index = 'direct', 
                                   values = 'price',
                                   aggfunc = np.mean)
gasoline_pivot_frm
```





[실습07]
서울지역 주유가격 상위 10, 하위 10 확인



```python
price_top_10_frm = gasoline_frm.sort_values(by = 'price', ascending = False).head(10)
price_bottom_10_frm = gasoline_frm.sort_values(by = 'price', ascending = True).head(10)
```


```python
price_top_10_frm
```






```python
price_bottom_10_frm
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>oil_store</th>
      <th>address</th>
      <th>price</th>
      <th>self_y_n</th>
      <th>company</th>
      <th>direct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>74</th>
      <td>만남의광장주유소</td>
      <td>서울 서초구 양재대로12길 73-71 (원지동)</td>
      <td>1490</td>
      <td>N</td>
      <td>알뜰(ex)</td>
      <td>서초구</td>
    </tr>
    <tr>
      <th>310</th>
      <td>강서오곡셀프주유소</td>
      <td>서울특별시 강서구 벌말로 254 (오곡동)</td>
      <td>1497</td>
      <td>Y</td>
      <td>SK에너지</td>
      <td>강서구</td>
    </tr>
    <tr>
      <th>404</th>
      <td>태릉솔밭주유소</td>
      <td>서울특별시 노원구 노원로 49 (공릉동)</td>
      <td>1497</td>
      <td>Y</td>
      <td>S-OIL</td>
      <td>노원구</td>
    </tr>
    <tr>
      <th>231</th>
      <td>수유동주유소</td>
      <td>서울특별시 강북구  도봉로 395 (수유동)</td>
      <td>1498</td>
      <td>Y</td>
      <td>GS칼텍스</td>
      <td>강북구</td>
    </tr>
    <tr>
      <th>201</th>
      <td>도림주유소</td>
      <td>서울 영등포구 도림로 343 (도림동)</td>
      <td>1499</td>
      <td>Y</td>
      <td>알뜰주유소</td>
      <td>영등포구</td>
    </tr>
    <tr>
      <th>373</th>
      <td>풀페이주유소</td>
      <td>서울특별시 구로구 경인로 41 (온수동)</td>
      <td>1499</td>
      <td>N</td>
      <td>SK에너지</td>
      <td>구로구</td>
    </tr>
    <tr>
      <th>203</th>
      <td>(주)강서오일</td>
      <td>서울 영등포구 도신로 151 (도림동)</td>
      <td>1499</td>
      <td>N</td>
      <td>현대오일뱅크</td>
      <td>영등포구</td>
    </tr>
    <tr>
      <th>35</th>
      <td>서경주유소</td>
      <td>서울 동작구 대림로 46 (신대방동)</td>
      <td>1499</td>
      <td>N</td>
      <td>현대오일뱅크</td>
      <td>동작구</td>
    </tr>
    <tr>
      <th>202</th>
      <td>(주)대청에너지 대청주유소</td>
      <td>서울 영등포구 가마산로 328 (대림동)</td>
      <td>1499</td>
      <td>N</td>
      <td>GS칼텍스</td>
      <td>영등포구</td>
    </tr>
    <tr>
      <th>294</th>
      <td>신일셀프주유소</td>
      <td>서울 중랑구 상봉로 58 (망우동)</td>
      <td>1499</td>
      <td>Y</td>
      <td>SK에너지</td>
      <td>중랑구</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('각각의 프레임에 위도, 경도 추가 - googlemaps')
import googlemaps
```

    각각의 프레임에 위도, 경도 추가 - googlemaps
    


```python
gmaps = googlemaps.Client(key='AIzaSyDdki495w62u7ziBMaVuzyYf4n-VZ8tsGA')
gmaps
```

    
     API queries_quota: 60 
    
    




    <googlemaps.client.Client at 0x1e7e56a4cd0>




```python
lat = []
lng = []

for address in price_top_10_frm['address']:
    tmp_lst = gmaps.geocode(address, language = 'ko')
    lat.append(tmp_lst[0].get('geometry')['location']['lat'])
    lng.append(tmp_lst[0].get('geometry')['location']['lng'])
```


```python
price_top_10_frm['lat'] = lat
price_top_10_frm['lng'] = lng
```


```python
price_top_10_frm
```






```python
lat = []
lng = []

for address in price_bottom_10_frm['address']:
    tmp_lst = gmaps.geocode(address, language = 'ko')
    lat.append(tmp_lst[0].get('geometry')['location']['lat'])
    lng.append(tmp_lst[0].get('geometry')['location']['lng'])
```


```python
price_bottom_10_frm['lat'] = lat
price_bottom_10_frm['lng'] = lng
```


```python
price_bottom_10_frm
```





```python
price_top_10_frm.reset_index(inplace=True)
del price_top_10_frm['index']
price_top_10_frm
```







```python
import folium
```


```python
map = folium.Map(location = [37.5202, 126.975])
map
```





```python
for idx in price_top_10_frm.index : 
    folium.CircleMarker([price_top_10_frm['lat'][idx] , price_top_10_frm['lng'][idx]] , 
                        fill_color = 'red' , 
                        fill = True).add_to(map)
map
```

