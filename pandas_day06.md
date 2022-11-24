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
    

#### 1. 서울시 구별 cctv 현황분석
- 서울시 각 구별 CCTV수를 파악하고, 인구대비 CCTV 비율을 파악해서 순위 비교
- 인구대비 CCTV의 평균치를 확인하고 그로부터 CCtv가 과하게 부족한 구를 확인
- service_data_cctv_in_seoul.csv(cctv현황)
- service_data_population_in_seoul.xlsx(인구현황)


```python
print('step 01. load 후 데이터 스키마 확인')

seoul_cctv_frm = pd.read_csv('./data/pandas_data/service_data_cctv_in_seoul.csv')
seoul_cctv_frm
```

    step 01. load 후 데이터 스키마 확인
    







```python
print('기관명 -> 구별 변경 - rename()')
seoul_cctv_frm.rename(columns = {'기관명': '구별'}, inplace = True)
seoul_cctv_frm
```

    기관명 -> 구별 변경 - rename()
    








```python
print('인구현황 데이터 로드 - ')
seoul_pop_frm = pd.read_excel('./data/pandas_data/service_data_population_in_seoul.xls',
                             header = 2,
                             usecols = 'B,D,G,J,N')
seoul_pop_frm
```

    인구현황 데이터 로드 - 
    



```python
seoul_pop_frm.columns
```




    Index(['자치구', '계', '계.1', '계.2', '65세이상고령자'], dtype='object')




```python
print('자치구 -> 구별, 인구수, 한국인, 외국인, 고령자 rename()')
seoul_pop_frm.rename(columns = {'자치구': '구별', '계': '인구수',
                                '계.1': '한국인', '계.2':'외국인',
                               '65세이상고령자':'고령자'}, inplace = True)
seoul_pop_frm
```

    자치구 -> 구별, 인구수, 한국인, 외국인, 고령자 rename()
    





```python
print('step02. cctv와 인구 현황 데이터 파악')
print('소계를 기준으로 오름차순 정렬 - cctv가 가장 많은 구 확인')
print(seoul_cctv_frm.sort_values(by = '소계', ascending=True))

print('2014-2016년 cctv수를 더하고 2013년 이전 cctv수로 나누어서 최근 3년간 cctv 증가율 계산 - ')
seoul_cctv_frm['증가율'] = round((seoul_cctv_frm['2014년'] + seoul_cctv_frm['2015년'] + seoul_cctv_frm['2016년']) / seoul_cctv_frm['2013년도 이전'] * 100 ,2)
print(seoul_cctv_frm)
```

    step02. cctv와 인구 현황 데이터 파악
    소계를 기준으로 오름차순 정렬 - cctv가 가장 많은 구 확인
          구별    소계  2013년도 이전  2014년  2015년  2016년
    9    도봉구   485        238    159     42    386
    12   마포구   574        314    118    169    379
    17   송파구   618        529     21     68    463
    24   중랑구   660        509    121    177    109
    23    중구   671        413    190     72    348
    5    광진구   707        573     78     53    174
    2    강북구   748        369    120    138    204
    1    강동구   773        379     99    155    377
    3    강서구   884        388    258    184     81
    19  영등포구   904        495    214    195    373
    13  서대문구   962        844     50     68    292
    22   종로구  1002        464    314    211    630
    7    금천구  1015        674     51    269    354
    15   성동구  1062        730     91    241    265
    11   동작구  1091        544    341    103    314
    8    노원구  1265        542     57    451    516
    10  동대문구  1294       1070     23    198    579
    16   성북구  1464       1009     78    360    204
    4    관악구  1496        846    260    390    613
    6    구로구  1561       1142    173    246    323
    20   용산구  1624       1368    218    112    398
    21   은평구  1873       1138    224    278    468
    14   서초구  1930       1406    157    336    398
    18   양천구  2034       1843    142     30    467
    0    강남구  2780       1292    430    584    932
    2014-2016년 cctv수를 더하고 2013년 이전 cctv수로 나누어서 최근 3년간 cctv 증가율 계산 - 
          구별    소계  2013년도 이전  2014년  2015년  2016년     증가율
    0    강남구  2780       1292    430    584    932  150.62
    1    강동구   773        379     99    155    377  166.49
    2    강북구   748        369    120    138    204  125.20
    3    강서구   884        388    258    184     81  134.79
    4    관악구  1496        846    260    390    613  149.29
    5    광진구   707        573     78     53    174   53.23
    6    구로구  1561       1142    173    246    323   64.97
    7    금천구  1015        674     51    269    354  100.00
    8    노원구  1265        542     57    451    516  188.93
    9    도봉구   485        238    159     42    386  246.64
    10  동대문구  1294       1070     23    198    579   74.77
    11   동작구  1091        544    341    103    314  139.34
    12   마포구   574        314    118    169    379  212.10
    13  서대문구   962        844     50     68    292   48.58
    14   서초구  1930       1406    157    336    398   63.37
    15   성동구  1062        730     91    241    265   81.78
    16   성북구  1464       1009     78    360    204   63.63
    17   송파구   618        529     21     68    463  104.35
    18   양천구  2034       1843    142     30    467   34.67
    19  영등포구   904        495    214    195    373  157.98
    20   용산구  1624       1368    218    112    398   53.22
    21   은평구  1873       1138    224    278    468   85.24
    22   종로구  1002        464    314    211    630  248.92
    23    중구   671        413    190     72    348  147.70
    24   중랑구   660        509    121    177    109   79.96
    


```python
print('증가율이 가장 높은 구 확인')
seoul_cctv_frm.sort_values(by='증가율', ascending = False).head(1)
```

    증가율이 가장 높은 구 확인
    





```python
print('step03 - 서울시 인구 데이터 파악')
print('drop - 합계 삭제')
print('구별 결측값 확인 - 만약, 결측값이 있다면 행 삭제')
print('각 구별 전체 인구를 이용하여 구별 외국인 비율과 고령자 비율 계산하여 반영')
print('인구수로 정렬하여 인사이트 찾아보자')

seoul_pop_frm.drop(0, axis=0, inplace = True)
seoul_pop_frm
```

    step03 - 서울시 인구 데이터 파악
    drop - 합계 삭제
    구별 결측값 확인 - 만약, 결측값이 있다면 행 삭제
    각 구별 전체 인구를 이용하여 구별 외국인 비율과 고령자 비율 계산하여 반영
    인구수로 정렬하여 인사이트 찾아보자
    








```python
seoul_pop_frm['구별'].unique()
```




    array(['종로구', '중구', '용산구', '성동구', '광진구', '동대문구', '중랑구', '성북구', '강북구',
           '도봉구', '노원구', '은평구', '서대문구', '마포구', '양천구', '강서구', '구로구', '금천구',
           '영등포구', '동작구', '관악구', '서초구', '강남구', '송파구', '강동구', nan],
          dtype=object)




```python
seoul_pop_frm[seoul_pop_frm['구별'].isnull()]
seoul_pop_frm.drop(26, axis = 0, inplace = True)
```


```python
seoul_pop_frm['구별'].unique()
```




    array(['종로구', '중구', '용산구', '성동구', '광진구', '동대문구', '중랑구', '성북구', '강북구',
           '도봉구', '노원구', '은평구', '서대문구', '마포구', '양천구', '강서구', '구로구', '금천구',
           '영등포구', '동작구', '관악구', '서초구', '강남구', '송파구', '강동구'], dtype=object)




```python
seoul_pop_frm['외국인비율'] = seoul_pop_frm['외국인']/seoul_pop_frm['인구수'] * 100
seoul_pop_frm['고령자비율'] = seoul_pop_frm['고령자']/seoul_pop_frm['인구수'] * 100
seoul_pop_frm.head()
```




```python
seoul_pop_frm.sort_values(by = '인구수', ascending=False).head()
```






```python
seoul_pop_frm.sort_values(by = '외국인', ascending=False).head()
```






```python
print('step04 - 인구대비 cctv 현황을 분석하기 위해서 테이블 병합 - ')
merge_outer_frm = pd.merge(seoul_cctv_frm, seoul_pop_frm, how='outer', on = '구별')
merge_outer_frm.head()
```

    step04 - 인구대비 cctv 현황을 분석하기 위해서 테이블 병합 - 
    





```python
merge_outer_frm.drop(['2013년도 이전','2014년','2015년','2016년'], axis=1, inplace=True)
merge_outer_frm
```






```python
print('시각화를 위해서 구별을 인덱스로 설정 - set_index')
merge_outer_frm.set_index('구별', inplace = True)
merge_outer_frm
```

    시각화를 위해서 구별을 인덱스로 설정 - 
    



```python
print('상관관계 지수를 확인하는 함수 - np.crrcoef()')
print(np.corrcoef(merge_outer_frm['인구수'], merge_outer_frm['소계']))
print(np.corrcoef(merge_outer_frm['고령자'], merge_outer_frm['소계']))
print(np.corrcoef(merge_outer_frm['외국인비율'], merge_outer_frm['소계']))
print(np.corrcoef(merge_outer_frm['고령자비율'], merge_outer_frm['소계']))
```

    상관관계 지수를 확인하는 함수 - np.crrcoef()
    [[1.         0.30634228]
     [0.30634228 1.        ]]
    [[1.         0.25519598]
     [0.25519598 1.        ]]
    [[ 1.         -0.13607433]
     [-0.13607433  1.        ]]
    [[ 1.         -0.28078554]
     [-0.28078554  1.        ]]
    

#### 실습2
- 강남 3구의 주민들이 자신들이 거주하는 구의 체감 안전도를 높게 생각한다는 기사
- http://news1.kr/articles/?1911504



```python
print('step01. load 후 데이터 스키마 확인')
seoul_crime_frm = pd.read_csv('./data/pandas_data/service_data_crime_in_seoul.csv',
                            encoding= 'cp949')
seoul_crime_frm.info()
```

    step01. load 후 데이터 스키마 확인
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 31 entries, 0 to 30
    Data columns (total 11 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   관서명     31 non-null     object
     1   살인 발생   31 non-null     int64 
     2   살인 검거   31 non-null     int64 
     3   강도 발생   31 non-null     int64 
     4   강도 검거   31 non-null     int64 
     5   강간 발생   31 non-null     int64 
     6   강간 검거   31 non-null     int64 
     7   절도 발생   31 non-null     object
     8   절도 검거   31 non-null     object
     9   폭력 발생   31 non-null     object
     10  폭력 검거   31 non-null     object
    dtypes: int64(6), object(5)
    memory usage: 2.8+ KB
    


```python
seoul_crime_frm.head()
```




```python
import googlemaps
```


```python
gmaps = googlemaps.Client(key='AIzaSyDdki495w62u7ziBMaVuzyYf4n-VZ8tsGA')
gmaps
```

    
     API queries_quota: 60 
    
    




    <googlemaps.client.Client at 0x2867d466940>




```python
gmaps.geocode('중부서', language='ko')
```




    [{'address_components': [{'long_name': '２７',
        'short_name': '２７',
        'types': ['premise']},
       {'long_name': '수표로',
        'short_name': '수표로',
        'types': ['political', 'sublocality', 'sublocality_level_4']},
       {'long_name': '중구',
        'short_name': '중구',
        'types': ['political', 'sublocality', 'sublocality_level_1']},
       {'long_name': '서울특별시',
        'short_name': '서울특별시',
        'types': ['administrative_area_level_1', 'political']},
       {'long_name': '대한민국',
        'short_name': 'KR',
        'types': ['country', 'political']},
       {'long_name': '100-032',
        'short_name': '100-032',
        'types': ['postal_code']}],
      'formatted_address': '대한민국 서울특별시 중구 수표로 27',
      'geometry': {'location': {'lat': 37.56361709999999, 'lng': 126.9896517},
       'location_type': 'ROOFTOP',
       'viewport': {'northeast': {'lat': 37.5649660802915,
         'lng': 126.9910006802915},
        'southwest': {'lat': 37.5622681197085, 'lng': 126.9883027197085}}},
      'partial_match': True,
      'place_id': 'ChIJc-9q5uSifDURLhQmr5wkXmc',
      'plus_code': {'compound_code': 'HX7Q+CV 대한민국 서울특별시',
       'global_code': '8Q98HX7Q+CV'},
      'types': ['establishment', 'point_of_interest', 'police']}]




```python
print('관서명이 **서이므로 주소검색에 문제 발생')
print('관서명을 서울**경찰서로 만들어서 station_name_list 리스트에 담기')
station_name_lst = []
for station in seoul_crime_frm['관서명']:
    station_name_lst.append('서울'+station[:-1]+'경찰서')
```

    관서명이 **서이므로 주소검색에 문제 발생
    관서명을 서울**경찰서로 만들어서 station_name_list 리스트에 담기
    


```python
station_name_lst
```




    ['서울중부경찰서',
     '서울종로경찰서',
     '서울남대문경찰서',
     '서울서대문경찰서',
     '서울혜화경찰서',
     '서울용산경찰서',
     '서울성북경찰서',
     '서울동대문경찰서',
     '서울마포경찰서',
     '서울영등포경찰서',
     '서울성동경찰서',
     '서울동작경찰서',
     '서울광진경찰서',
     '서울서부경찰서',
     '서울강북경찰서',
     '서울금천경찰서',
     '서울중랑경찰서',
     '서울강남경찰서',
     '서울관악경찰서',
     '서울강서경찰서',
     '서울강동경찰서',
     '서울종암경찰서',
     '서울구로경찰서',
     '서울서초경찰서',
     '서울양천경찰서',
     '서울송파경찰서',
     '서울노원경찰서',
     '서울방배경찰서',
     '서울은평경찰서',
     '서울도봉경찰서',
     '서울수서경찰서']




```python
print('경찰서이름의 풀 주소와 위도, 경도를 구해보자 - ')
station_address = []
station_lat = []
station_lng = []

for station in station_name_lst:
    address = gmaps.geocode(station, language='ko')
    station_address.append(address[0].get("formatted_address"))
    
    address_geo = address[0].get('geometry')
    
    station_lat.append(address_geo['location']['lat'])
    station_lng.append(address_geo['location']['lng'])
    
print(station_address)
print(station_lat)
print(station_lng)
```

    경찰서이름의 풀 주소와 위도, 경도를 구해보자 - 
    ['대한민국 서울특별시 중구 수표로 27', '대한민국 서울특별시 종로구 인사동5길 41', '대한민국 서울특별시 중구 한강대로 410', '대한민국 서울특별시 서대문구 통일로 113', '대한민국 서울특별시 종로구 창경궁로 112-16', '대한민국 서울특별시 용산구 백범로 329', '대한민국 서울특별시 성북구 삼선동 보문로 170', '대한민국 서울특별시 동대문구 약령시로21길 29', '대한민국 서울특별시 마포구 마포대로 183', '대한민국 서울특별시 영등포구 국회대로 608', '대한민국 서울특별시 성동구 행당동 왕십리광장로 9', '대한민국 서울특별시 동작구 노량진로 148', '대한민국 서울특별시 광진구 구의동 자양로 167', '대한민국 서울특별시 은평구 진흥로 58', '대한민국 서울특별시 강북구 오패산로 406', '대한민국 서울특별시 금천구 시흥대로73길 50', '대한민국 서울특별시 중랑구 묵2동 249-2', '대한민국 서울특별시 강남구 테헤란로114길 11', '대한민국 서울특별시 관악구 관악로5길 33', '대한민국 서울특별시 양천구 신월동 화곡로 73', '대한민국 서울특별시 강동구 성내로 57', '대한민국 서울특별시 성북구 종암로 135', '대한민국 서울특별시 구로구 가마산로 235', '대한민국 서울특별시 서초구 서초3동 반포대로 179', '대한민국 서울특별시 양천구 목동동로 99', '대한민국 서울특별시 송파구 중대로 221', '대한민국 서울특별시 노원구 노원로 283', '대한민국 서울특별시 서초구 동작대로 204', '대한민국 서울특별시 은평구 연서로 365', '대한민국 서울특별시 도봉구 노해로 403', '대한민국 서울특별시 강남구 개포로 617']
    [37.56361709999999, 37.571824, 37.5547584, 37.5647439, 37.5719679, 37.5387099, 37.58977830000001, 37.58506149999999, 37.550814, 37.5260441, 37.5617303, 37.5130866, 37.542873, 37.6020914, 37.63730390000001, 37.4568722, 37.6056429, 37.5094352, 37.4743945, 37.5397827, 37.528511, 37.6019994, 37.494931, 37.4956054, 37.5167711, 37.5016941, 37.6421389, 37.4945959, 37.6280204, 37.6533589, 37.49349]
    [126.9896517, 126.9841533, 126.9734981, 126.9667705, 126.9989574, 126.9659183, 127.016589, 127.0457679, 126.954028, 126.9008091, 127.0364217, 126.9428498, 127.083821, 126.9213528, 127.0273399, 126.8970429, 127.0764866, 127.0669578, 126.9513489, 126.8299968, 127.1268224, 127.0322276, 126.886731, 127.0052504, 126.8656996, 127.1272481, 127.0710473, 126.9831279, 126.9287899, 127.052682, 127.0772119]
    


```python
seoul_crime_frm['관서명'].unique()
```




    array(['중부서', '종로서', '남대문서', '서대문서', '혜화서', '용산서', '성북서', '동대문서', '마포서',
           '영등포서', '성동서', '동작서', '광진서', '서부서', '강북서', '금천서', '중랑서', '강남서',
           '관악서', '강서서', '강동서', '종암서', '구로서', '서초서', '양천서', '송파서', '노원서',
           '방배서', '은평서', '도봉서', '수서서'], dtype=object)




```python
print('구별 컬럼추가 - station_address를 이용해서 ')
station_gu = []
for address in station_address:
    station_gu.append(address.split()[2])
```

    구별 컬럼추가 - station_address를 이용해서 
    


```python
print(station_gu)
```

    ['중구', '종로구', '중구', '서대문구', '종로구', '용산구', '성북구', '동대문구', '마포구', '영등포구', '성동구', '동작구', '광진구', '은평구', '강북구', '금천구', '중랑구', '강남구', '관악구', '양천구', '강동구', '성북구', '구로구', '서초구', '양천구', '송파구', '노원구', '서초구', '은평구', '도봉구', '강남구']
    


```python
seoul_crime_frm['구별'] = station_gu
```


```python
seoul_crime_frm.head()
```





```python
print('파일 저장 - service_data_crime_gu_in_seoul.csv')
seoul_crime_frm.to_csv('./data/pandas_data/service_data_crime_gu_in_seoul.csv',
                       encoding= 'cp949')

```

    파일 저장 - service_data_crime_gu_in_seoul.csv
    저장한 파일 로드 - 
    


```python
print('저장한 파일 로드 - ')
seoul_crime_frm = pd.read_csv('./data/pandas_data/service_data_crime_gu_in_seoul.csv',
                             encoding = 'cp949',
                             thousands = ',',
                             index_col = 0)
seoul_crime_frm.head()
```

    저장한 파일 로드 - 
    







```python
seoul_crime_frm.drop('Unnamed: 0', axis = 1, inplace = True)
seoul_crime_frm.head()
```





```python
print('step03. 범죄데이터를 구별로 정리하기')
print('피봇테이블을 이용해서 관서별에서 구별로 바꿔보자')

seoul_crime_frm.pivot_table(index='구별',
                           aggfunc = np.sum)
```

    step03. 범죄데이터를 구별로 정리하기
    피봇테이블을 이용해서 관서별에서 구별로 바꿔보자
    





```python
print('step04 - 범죄별 검거율계산해서 검거 건수를 검거율로 대체')
print('강간검거율, 살인검거율, 강도검거율, 절도검거율, 폭력검거율 대체하고 검거는 삭제')

seoul_crime_frm['강도검거율'] = seoul_crime_frm['강도 검거']/seoul_crime_frm['강도 발생']*100
seoul_crime_frm['살인검거율'] = seoul_crime_frm['살인 검거']/seoul_crime_frm['살인 발생']*100
seoul_crime_frm['강간검거율'] = seoul_crime_frm['강간 검거']/seoul_crime_frm['강간 발생']*100
seoul_crime_frm['절도검거율'] = seoul_crime_frm['절도 검거']/seoul_crime_frm['절도 발생']*100
seoul_crime_frm['폭력검거율'] = seoul_crime_frm['폭력 검거']/seoul_crime_frm['폭력 발생']*100
```

    step04 - 범죄별 검거율계산해서 검거 건수를 검거율로 대체
    강간검거율, 살인검거율, 강도검거율, 절도검거율, 폭력검거율 대체하고 검거는 삭제
    


```python
seoul_crime_pivot_frm = seoul_crime_frm.pivot_table(index='구별', aggfunc=np.sum)
seoul_crime_pivot_frm
```





```python
del seoul_crime_pivot_frm['강간 검거']
del seoul_crime_pivot_frm['강도 검거']
del seoul_crime_pivot_frm['살인 검거']
del seoul_crime_pivot_frm['절도 검거']
del seoul_crime_pivot_frm['폭력 검거']
```


```python
seoul_crime_pivot_frm
```





```python
cols = ['강간검거율', '살인검거율', '강도검거율', '폭력검거율', '절도검거율']
for col in cols:
    seoul_crime_pivot_frm[col][seoul_crime_pivot_frm[col]>100] =100
    
seoul_crime_pivot_frm
```





```python
print('step05 - xx발생 : xx rename')

loop = ['강간', '강도', '살인', '폭력', '절도']
for i in loop:
    seoul_crime_pivot_frm.rename(columns = {i + '발생' : i}, inplace = True)
    
seoul_crime_pivot_frm
```

    step05 - xx발생 : xx rename
    


```python
seoul_crime_pivot_frm
```

