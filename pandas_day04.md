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

#### 그룹 분석 및 피봇테이블
- groupby(열 또는 열리스트 | 행 인덱스)
- 그룹연산(size, count, mean, median, min, max, sum, std, var, quantile, first, last, aggregate, describe, transform)
- 분할(split): 특징에 따라서 분류
- 적용(apply): 집계, 변환, 필터링 함수의 적용
- 결합(combine): 처리된 결과를 하나로 결합


```python
grp_frm = pd.DataFrame({
    '학과': ['DA', 'AI', 'AI', 'DA', 'AI', 'AI'],
    '학년': [1,2,3,4,2,3],
    '이름': ['섭섭해', '최강희', '김희선', '신사임당', '임꺽정', '홍길동'],
    '학점': [4.5,3.5,3.2,4.3,2.1,1.9]
})
frmInfo(grp_frm)
```

    shape -  (6, 4)
    size -  24
    ndim -  2
    row index -  RangeIndex(start=0, stop=6, step=1) <class 'pandas.core.indexes.range.RangeIndex'>
    col index -  Index(['학과', '학년', '이름', '학점'], dtype='object') <class 'pandas.core.indexes.base.Index'>
    values - 
     [['DA' 1 '섭섭해' 4.5]
     ['AI' 2 '최강희' 3.5]
     ['AI' 3 '김희선' 3.2]
     ['DA' 4 '신사임당' 4.3]
     ['AI' 2 '임꺽정' 2.1]
     ['AI' 3 '홍길동' 1.9]] <class 'numpy.ndarray'>
    
    data - 
    





```python
dept_grp = grp_frm['학과'].groupby(grp_frm['학과'])
print(dept_grp.groups)
print(dept_grp.groups['AI'])
```

    {'AI': [1, 2, 4, 5], 'DA': [0, 3]}
    Int64Index([1, 2, 4, 5], dtype='int64')
    

- get_group()


```python
dept_grp.get_group('AI')
```




    1    AI
    2    AI
    4    AI
    5    AI
    Name: 학과, dtype: object




```python
dept_grp_frm = grp_frm.groupby(grp_frm['학과'])
dept_grp_frm.get_group('DA')
```





```python
dept_grp_frm.get_group('AI')
```





```python
grp_frm.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6 entries, 0 to 5
    Data columns (total 4 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   학과      6 non-null      object 
     1   학년      6 non-null      int64  
     2   이름      6 non-null      object 
     3   학점      6 non-null      float64
    dtypes: float64(1), int64(1), object(2)
    memory usage: 320.0+ bytes
    


```python
grp_frm.describe()
```





```python
print('type - ', type(grp_frm.mean()))
print()
print(grp_frm.mean())
print()
print(grp_frm.count())
```

    type -  <class 'pandas.core.series.Series'>
    
    학년    2.50
    학점    3.25
    dtype: float64
    
    학과    6
    학년    6
    이름    6
    학점    6
    dtype: int64
    


```python
print('학과별 학년평균과 학점평균 - 결과는 데이터프레임')
dept_grp_frm.mean()
```

    학과별 학년평균과 학점평균 - 결과는 데이터프레임
    




```python
dept_grp_frm.sum()
```






```python
dept_grp_frm.agg([np.mean,np.sum])
```





```python
print('type - ', type(dept_grp_frm.agg([np.mean, np.sum])))
```

    type -  <class 'pandas.core.frame.DataFrame'>
    

- 다중그룹


```python
multi_grp = grp_frm.groupby(['학과', '학년'])
multi_grp.groups
```




    {('AI', 2): [1, 4], ('AI', 3): [2, 5], ('DA', 1): [0], ('DA', 4): [3]}




```python
multi_grp.get_group(('AI',2))
```






```python
import seaborn as sns
titanic_frm = sns.load_dataset('titanic')
iris_frm = sns.load_dataset('iris')
```


```python
print('subset - age,sex,class,fare,survived')
titanic_subset_frm = titanic_frm.loc[:, ['age','sex','class','fare','survived']]
display(titanic_subset_frm)
```

    subset - age,sex,class,fare,survived
    



```python
titanic_subset_frm.describe()
```







```python
titanic_subset_frm.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 5 columns):
     #   Column    Non-Null Count  Dtype   
    ---  ------    --------------  -----   
     0   age       714 non-null    float64 
     1   sex       891 non-null    object  
     2   class     891 non-null    category
     3   fare      891 non-null    float64 
     4   survived  891 non-null    int64   
    dtypes: category(1), float64(2), int64(1), object(1)
    memory usage: 29.0+ KB
    


```python
print('승객수 - ', len(titanic_subset_frm))
```

    승객수 -  891
    


```python
print('선실등급에 따른 그룹작성 - ')
class_grp = titanic_subset_frm.groupby('class')
print(class_grp)
print(class_grp.groups)
print()

print(class_grp.groups['First'].values)
print()
print('1등급 승객만 데이터프레임 형식으로 출력 - ')
display(titanic_subset_frm.loc[class_grp.groups['First'].values, :])
```

    선실등급에 따른 그룹작성 - 
    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x000001CD94E13580>
    {'First': [1, 3, 6, 11, 23, 27, 30, 31, 34, 35, 52, 54, 55, 61, 62, 64, 83, 88, 92, 96, 97, 102, 110, 118, 124, 136, 137, 139, 151, 155, 166, 168, 170, 174, 177, 185, 187, 194, 195, 209, ...], 'Second': [9, 15, 17, 20, 21, 33, 41, 43, 53, 56, 58, 66, 70, 72, 78, 84, 98, 99, 117, 120, 122, 123, 133, 134, 135, ...], 'Third': [0, 2, 4, 5, 7, 8, 10, 12, 13, 14, 16, 18, 19, 22, 24, 25, 26, 28, 29, 32, 36, 37, 38, 39, 40, 42, 44, 45, 46, 47, 48, 49, 50, 51, 57, 59, 60, 63, 65, 67, 68, 69, ...]}
    
    [  1   3   6  11  23  27  30  31  34  35  52  54  55  61  62  64  83  88
      92  96  97 102 110 118 124 136 137 139 151 155 166 168 170 174 177 185
     187 194 195 209 ... 781 782 789 793 796 802 806 809
     815 820 822 829 835 839 842 849 853 856 857 862 867 871 872 879 887 889]
    
    1등급 승객만 데이터프레임 형식으로 출력 - 
    



```python
for key, data in class_grp :
    print('key - ', key)
    print('len - ', len(key))
    display(data)
    
    
```


```python
class_grp.agg([np.mean, np.sum])
```






```python
class_grp.groups['First']
```




    Int64Index([  1,   3,   6,  11,  23,  27,  30,  31,  34,  35,
                ...
                853, 856, 857, 862, 867, 871, 872, 879, 887, 889],
               dtype='int64', length=216)




```python
class_grp.get_group('First')
```





```python
print('선실등급과 성별에 따른 그룹 생성 - ')
class_sex_grp = titanic_subset_frm.groupby(['class', 'sex'])
print(class_sex_grp)
```

    선실등급과 성별에 따른 그룹 생성 - 
    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x000001CD94E00EB0>
    


```python
print('일등급에 여성 정보를 추출 - ')
class_sex_frm = class_sex_grp.get_group(('First', 'female'))
print('type - ', type(class_sex_frm))
class_sex_frm.reset_index(inplace = True)
display(class_sex_frm.drop('index', axis = 1))
```

    일등급에 여성 정보를 추출 - 
    type -  <class 'pandas.core.frame.DataFrame'>
    



```python
iris_frm

```




```python
print('품종별 그룹을 만들고 확인 - ')
titanic_frm.groupby('species')
print('품종별 가장 큰 값과 가장 작은 값의 비율을 구한다면? - ')
print('품종별 가장 큰 petal_length 3개만 추출한다면? - ')




```python
species_grp = iris_frm.groupby('species')
species_grp.groups
```




    {'setosa': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49], 'versicolor': [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99], 'virginica': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149]}




```python
for key, data in species_grp:
    print('key - ', key)
    print()
    display(data)
```

    key -  setosa
    
    




    key -  versicolor
    
    




    key -  virginica
    
    





```python
species_grp.describe()
```






```python
def get_ration(feature):
    return feature.max() / feature.min()
```


```python
species_grp.agg(get_ration)
```








```python
iris_frm.sort_values(by='petal_length', ascending=False)[:3]
iris_frm.sort_values(by='petal_length', ascending=False).groupby('species').head()

```





```python
def top03_petal_length_func(grp) :
    return grp.sort_values(by = 'petal_length', ascending=False)[:3]

species_grp.apply(top03_petal_length_func)
```






- index를 이용한 데이터 분할(set_index, reset_index)


```python
titanic_frm.set_index(['pclass', 'sex'])
```






```python
sex_grp = titanic_frm.set_index(['pclass', 'sex']).groupby(level=[0,1])
sex_grp.groups
```




    {(1, 'female'): [(1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female'), (1, 'female')], (1, 'male'): [(1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), (1, 'male'), ...], (2, 'female'): [(2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female'), (2, 'female')], (2, 'male'): [(2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), (2, 'male'), ...], (3, 'female'): [(3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), (3, 'female'), ...], (3, 'male'): [(3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), (3, 'male'), ...]}




```python
sex_grp.get_group((1,'female'))
```







```python
sex_grp.agg([np.mean, np.sum, np.max])
```



- transform(): 원본 인덱스와 데이터프레임을 그대로 유지하면서 각 범주별 통계치를 원본 데이터프레임에 대입하고 싶다면?
- cut()
- qcut()


```python
class_grp = titanic_frm.groupby('pclass').transform(np.mean)
class_grp
```






```python
print('각 붓꽃 꽃잎길이가 해당 종 내에서 소/중/대 어느것에 해당되는지에 대한 프레임을 만들고 싶다면 - ')
species_grp = iris_frm.groupby('species')
species_grp.petal_length.size()

def category_petal_length(s_grp):
    return pd.qcut(s_grp, 3, labels=['소','중','대'])

iris_frm['category'] = species_grp.petal_length.transform(category_petal_length)
iris_frm.head()
```

    각 붓꽃 꽃잎길이가 해당 종 내에서 소/중/대 어느것에 해당되는지에 대한 프레임을 만들고 싶다면 - 
    




```python
print('타이타닉 데이터를 이용해서 분석을 진행해 보자 - ')
print('조건1) qcut 이용해서 나이를 세 개의 그룹으로 만들어보자')
print('조건2) 성별, 선실, 나이 그룹에 의한 생존율 데이터 프레임으로 계산 - transform')
print('조건3) 행 인덱스로 성별과 나이로 다중 인덱스로 만들고 열 인덱스는 선실')
print('조건4) 생존율을 해당 그룹에 생존 인원수를 전체 인원수로 나눈 값으로 한다')
```

    타이타닉 데이터를 이용해서 분석을 진행해 보자 - 
    조건1) qcut 이용해서 나이를 세 개의 그룹으로 만들어보자
    조건2) 성별, 선실, 나이 그룹에 의한 생존율 데이터 프레임으로 계산 - transform
    조건3) 행 인덱스로 성별과 나이로 다중 인덱스로 만들고 열 인덱스는 선실
    조건4) 생존율을 해당 그룹에 생존 인원수를 전체 인원수로 나눈 값으로 한다
    

#### 피봇
- 데이터프레임에서 두 개의 열을 이용해서 행/열 인덱스로 reshape 된 테이블을 통한 집계
- 새로운 데이터 프레임이 생성
- pivot(index, columns, data.values)
- pivot_table(data, values, index, columns, aggfunc = )


```python
tmp_frm = pd.DataFrame({
    'person': ['A', 'A', 'A', 'B', 'B', 'A', 'A', 'C', 'B', 'B', 'B'],
    'day': ['monday', 'tuesday', 'wednesday', 'monday', 'tuesday', 'monday', 'thursday', 'friday', 'tuesday', 'wednesday', 'thursday'],
    'sport': ['baseball', 'basketball', 'soccer', 'golf', 'golf', 'basketball', 'soccer', 'tennis', 'baseball', 'basketball', 'baseball'],
    'time' : [240, 60, 90, 300, 20, 30, 70, 40, 40, 30, 70]    
})
tmp_frm
```





```python
person_day_frm = tmp_frm.groupby(['person', 'day']).size().reset_index(name='time')
person_day_frm
print('type - ', type(person_day_frm))
print()
display(person_day_frm)
```

    type -  <class 'pandas.core.frame.DataFrame'>
    
    





```python
person_day_frm.pivot('person', 'day', 'time')
```







```python
sex_class_frm = titanic_frm.groupby(['sex', 'pclass']).size().reset_index()
print('type - ', type(sex_class_frm))
print()
display(sex_class_frm)
```

    type -  <class 'pandas.core.frame.DataFrame'>
    
    


