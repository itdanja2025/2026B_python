## lightGBM 튜닝 / 서빙 예제 필요한 데이터셋 
import pandas as pd
url = 'https://raw.githubusercontent.com/rashida048/Datasets/master/USA_cars_datasets.csv'
df = pd.read_csv(url)
# *만약에 모델의 성능 향상 하기 위한 도메인 지식의 (계산식)특성 추가 좋다.*
df = df[df['price'] > 0]
num_cols = ['year', 'mileage']                                  # 수치형 
cat_cols = ['brand', 'model', 'title_status', 'color', 'state'] # 범주형 
# 범주형 자료들을 원핫인코딩이 아닌 판다스'카테고리' 타입 변경 # lightGBM 내장 원핫인코딩 사용 목적
for col in cat_cols: 
    df[col] = df[col].astype('category') 
X = df[num_cols + cat_cols]
Y = df['price']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 1) 튜닝 -> optuna
# 1. optuna 설치 : $env:PYYAML_FORCE_LIBYAML = "0" , pip install optuna shap
# 2) optuna 로그 숨기기/경고만 출력
import optuna
optuna.logging.set_verbosity( optuna.logging.WARNING )
# 4) 목적함수 만들기: 각 탐색 시도 마다 교차 검증 수행 후 평균 성능 반환, *콜백함수( trial )
def objective( trial ) :
    pass
# 3) optuna.create_study( direction='maximize/minimize' )
# 현재 실습은 회귀모델이며 r2결정계수 평가지표 사용할 예정 
sampler = optuna.samplers.TPESampler( seed=42 )
study = optuna.create_study( direction='maximize' , sampler= sampler )
# study.optimize( 목적함수 , n_trials=반복수 )
study.optimize( objective , n_trials=20 ) # -> 반복 수 만큼 목적함수 실행 
