# AI스터디

## 1. 딥러닝 개요
### 1.1. 딥러닝 프레임 워크

#### a. 텐서플로우(TensorFlow 2.x 버전)
 + Google, 핵심코드가 C++로 작성
 + 직관적인 고수준 API
 + 뛰어난 이식성 및 확장성, 즉 다양한 플랫폼으로 확장하여 활용 가능
 + Tensorflow-Lite, TensorFlow Extended
 + 진입장벽 다소 높음        

#### b. 케라스(Keras)
 + 직관적이고 쉬운 API
 + TensorFlow의 백엔드(Backend) 활용
 + 동일한 코드로 CPU, GPU에서 실행 가능
 + 파이토치(PyTorch)

#### c. 파이토치(PyTorch)
 + Facebook
 + C/CUDA Backend 사용
 + 진입장벽이 낮음. 파이썬 문법과 유사
 + GPU 가속연산

#### 1.2. 신경망 정의
 + 인공지능 분야에서 쓰이는 알고리즘
 + 인간의 뇌 구조를 모방
 + 뉴런과 뉴런 사이에는 전기신호를 통해 정보를 전달
 + 입력(inputs)과 가중치(weights)를 곱한 선형구조(linear)
 + 활성화 함수(activation function)를 통한 비선형 구조(non-linear) 표현 가능

##### 1.2.1. 인공뉴런
 + 노드(Node)와 엣지(Edge)로 표현
 + 하나의 노드안에서 입력(Inputs)와 가중치(Weights)를 곱하고 더하는 선형(Linear)계산
 + 활성화 함수(Activation Function) 통과를 모두 포함

##### 1.2.2. 인공 신경망 (Artificial Neural Network)
 + 여러 개의 인공뉴런들이 모여 연결된 형태
 + 뉴런들이 모인 하나의 단위를 층(Layer)이라고 하고, 여러 층(Layer)으로 이루어질 수 있음

##### 1.2.3. 완전 연결 계층(Fully-Connected Layer)
 + 모든 노드들이 서로 연결된 신경망
 + Dense Layer라고도 불림

#### 1.3. 신경망 활용
##### 1.3.1 회귀(Regression)
 + 잡음(Noise)을 포함한 학습 데이터로부터 어떤 규칙을 찾고 연속된 값의 출력을 추정
 + 아래의 식을 만족하는 적절한  𝑎 (기울기), 𝑏 (𝑦 절편)를 찾아야함

~~~
 𝑌=𝑎𝑋+𝑏 
> 𝑋: 입력
> 𝑌: 출력
> 𝑎: 기울기
> 𝑏: 𝑦 절편
~~~

+ 회귀 문제 예시
  + 나이, 키, 몸무게에 따른 기대수명
  + 아파트의 방의 개수, 크기, 주변 상가 등에 따른 아파트 가격

##### 1.3.2. 분류(Classification)
 + 입력값에 따라 특정 범주(category)로 구분하는 문제로, 분류 곡선(직선)을 찾아야 함

 + 범주의 개수에 따라서 이진 분류(Binary Classification), 로지스틱 회귀(Logistic Regression), 선형회귀와 비슷하지만, 범주형 데이터를 분류하는 방향으로 선을 그음

   + 면접점수, 실기점수, 필기점수에 따른 시험 합격 여부

 + 다중 분류(Multi-Class Classification)
   + 꽃잎 모양, 색깔에 따른 꽃의 종 분류

