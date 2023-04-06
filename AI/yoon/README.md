# AI-Python_Study
<h2>파이토치 기초</h2>

* torch: 텐서를 생성하는 라이브러리
* torch.autograd: 자동미분 기능을 제공하는 라이브러리
* torch.nn: 신경망을 생성하는 라이브러리
* torch.multiprocessing: 병렬처리 기능을 제공하는 라이브러리
* torch.utils: 데이터 조작 등 유틸리티 기능 제공
* torch.legacy(./nn/.optim): Torch로부터 포팅해온 코드
* torch.onnx: ONNX(Open Neural Network Exchange)
            서로 다른 프레임워크 간의 모델을 공유할 때 사용
<h2>텐서(Tensors)</h2>

* 넘파이(NumPy)의 ndarry(n차원 array)와 유사
* GPU를 사용한 연산 가속도 가능한 기능

<pre>
<code>
import torch
torch.__version__
</code>
</pre>

<h2>초기화 되지 않은 행렬</h2>

<pre>
<code>
x = torch.empty(4, 2)
print(x)
</code>
</pre>

<h2>무작위로 초기화된 행렬</h2>

<pre>
<code>
x = torch.rand(4, 2)
print(x)
</code>
</pre>

<h2>dtype이 long, 0으로 채워진 텐서</h2>

<pre>
<code>
x = torch.zeros(4, 2, dtype=torch.long)
print(x)
</code>
</pre>

<h2>텐서를 직접 만들기</h2>

<pre>
<code>
x = torch.tensor([3, 2])
print(x)
</code>
</pre>

<pre>
<code>
x = x.new_ones(2, 4, dtype=torch.double)
print(x)
-> tensor([[1., 1., 1., 1.],
          [1., 1., 1., 1.]], dtype=torch.float64)
  : double은 float64로 매칭됨을 알 수 있다.
</code>
</pre>

<pre>
<code>
x = torch.randn_like(x, dtype=torch.float)
print(x)
//기존의 x라는 shape(형)을 그대로 가져와서 random으로 값 채우고 그 값을 dtype의 torch를 float으로 바꿔줘
</code>
</pre>

<h2>텐서의 크기</h2>

<pre>
<code>
print(x.size())
</code>
</pre>

<h2>텐서의 연산(operations)</h2>
<h3>덧셈1</h3>

<pre>
<code>
x = torch.rand(2, 4)
y = torch.rand(2, 4)
print(x)
print(y)
print(x + y)
</code>
</pre>

<h3>덧셈 2</h3>

<pre>
<code>
x = torch.rand(2, 4)
y = torch.rand(2, 4)
print(x)
print(y)
print(torch.add(x, y))
</code>
</pre>

<h3>덧셈3 결과 텐서를 인자로 제공</h3>

<pre>
<code>
result = torch.empty(2, 4)  //result에 torch.empty(2, 4) 만들고
torch.add(x, y, out=result)  //torch.add 연산 하는데 x, y를 주고 이 x, y 결과값을 out=result로 뺄 수 있다
print(result)
</code>
</pre>

<h3>덧셈4</h3>

* in-place 방식
- in-place방식으로 텐서의 값을 변경하는 연산 뒤에는 _''가 붙음
- x.copy_(y), x.t_()

<pre>
<code>
print(x)
print(y)
y.add_(x) // y+=x
print(y)
</code>
</pre>

<h2>그 외의 연산</h2>

* torch.sub: 뺄셈
<pre>
<code>
x = torch.Tensor([[1, 3], [5, 7]])
y = torch.Tensor([[2, 4], [6, 8]])
print(x - y)
print(torch.sub(x, y))
print(x.sub(y))
// 세 방법 모두 결과 동일
</code>
</pre>

* torch.mul: 곱셈
<pre>
<code>
x = torch.Tensor([[1, 3], [5, 7]])
y = torch.Tensor([[2, 4], [6, 8]])
print(x * y)
print(torch.mul(x, y))
print(x.mul(y))
</code>
</pre>

* torch.div: 나눗셈
<pre>
<code>
x = torch.Tensor([[1, 3], [5, 7]])
y = torch.Tensor([[2, 4], [6, 8]])
print(x / y)
print(torch.div(x, y))
print(x.div(y))
</code>
</pre>

* torch.mm: 내적(dot product)
<pre>
<code>
x = torch.Tensor([[1, 3], [5, 7]])
y = torch.Tensor([[2, 4], [6, 8]])
print(torch.mm(x, y))
</code>
</pre>

<h2>인덱싱</h2>

* 넘파이처럼 인덱싱 사용 가능
<pre>
<code>
print(x[:, 1])
</code>
</pre>

<h2>view</h2>

* 텐서의 크기(size)나 모양(shape)을 변경
<pre>
<code>
x = torch.randn(4, 5) 
y = x.view(20) 
z = x.view(5, -1)

print(x.size()) //4, 5행렬
print(y.size()) //1차원 행렬
print(z.size) //5, random 행렬 -> 5, 4 형태
</code>
</pre>

<h2>item</h2>

* 텐서에 값이 단 하나라도 존재하면 숫자값을 얻을 수 있음
<pre>
<code>
x = torch.randn(1)
print(x)
print(x.item()) // x 안에 있는 실제값을 출력
print(x.dtype)
</code>
</pre>

* 스칼라값 하나만 존재해야 함
<pre>
<code>
x = torch.randn(2) // element tensor 2개라서 오류
print(x)
print(x.item()) 
print(x.dtype)
</code>
</pre>

<h2>squeeze</h2>

* 차원을 축소(제거)
<pre>
<code>
tensor = torch.randn(1, 3, 3)
print(tensor)
print(tensor.shape)

-> tensor([...])
   torch.Size([1, 3, 3])
</code>
</pre>

<pre>
<code>
t = tensor.squeeze()

print(t)
print(t.shape)

-> tensor([...])
   torch.Size([3, 3]) // 1, 3, 3에서 3, 3으로 차원 축소
</code>
</pre>

<h2>unsqueeze</h2>

* 차원을 증가(생성)
<pre>
<code>
tensor = torch.randn(1, 3, 3)
print(tensor)
print(tensor.shape)

-> tensor([...])
   torch.Size([1, 3, 3])
</code>
</pre>

<pre>
<code>
t = tensor.unsqueeze(dim=0) //unsqueeze할 때 dim 지정 가능

print(t)
print(t.shape)

-> tensor([...])
   torch.Size([1, 1, 3, 3]) // 1, 3, 3에서 1, 1, 3, 3으로 차원 증가
</code>
</pre>

<h2>stack</h2>

* 텐서 간 결합
<pre>
<code>
x = torch.FloatTensor([1, 4])
x = torch.FloatTensor([2, 5])
x = torch.FloatTensor([3, 6])

print(torch.stack([x, y, z]))

-> tensor([[1., 4.]
          [2., 5.]
          [3., 6.]])
</code>
</pre>

<h2>cat</h2>

* 텐서를 결합하는 메소드(concatenate)
* 넘파이의 stack과 유사하지만, 쌓을 dim이 존재해야 함
    - 예를 들어, 해당 차원을 늘려준 후 결합

<pre>
<code>
a = torch.randn(1, 1, 3, 3)
b = torch.randn(1, 1, 3, 3)
c = torch.cat((a, b), dim=0) //dim을 0으로 지정해줬기 때문에 그 단위로 concatenate 된 것

print(c)
print(c.size())
</code>
</pre>

<h2>chunk</h2>

* 텐서를 여러 개로 나눌 때 사용
* 몇 개의 텐서로 나눌 것이냐

<pre>
<code>
tenso = torch.rand(3, 6)
t1, t2, t3 = torch.chunk(tensor, 3, dim=1) // chunk의 개수(텐서를 나누는 개수) 3개로 지정

print(tensor)
print(t1)
print(t2)
print(t3)
</code>
</pre>

<h2>split</h2>

* chunk와 동일한 기능이지만 조금 다름
* 하나의 텐서 당 크기가 얼마이냐
<pre>
<code>
tenso = torch.rand(3, 6)
t1, t2= torch.split(tensor, 3, dim=1) // 하나의 텐서가 의미하는 크기가 3

print(tensor)
print(t1)
print(t2)
</code>
</pre>

<h2>torch<->numpy</h2>

* Torch Tensor(텐서)를 Numpy array(배열)로 변환 가능
    - numpy()
    - from_numpy()
* (참고)
    - Tensor가 CPU상에 있다면 Numpy 배열은 메모리 공간을 공유하므로 하나가 변하면, 다른 하나도 변함

<h2>4월 4일 딥러닝 개요</h2>

<h3>딥러닝 타임라인(Deep Learining Timeline)</h2>

<h3>대표적인 프레임워크</h3>

* 텐서플로우(TensorFlow 2.x버전)
   - Google에서 개발, 핵심코드가 C++로 작성
   - 직관적인 고수준 API
   - 뛰어난 이식성 및 확장성, 즉 다양한 플랫폼으로 확장하여 활용 가능
      + Tensorflow-Life, TensorFlow Extended
   - 실무에서 사용되는 거라 진입장벽 다소 높음
* 케라스(Keras)
   - 직관적이고 쉬운 API
   - TensorFlow의 백엔드(Backend) 활용
   - 동일한 코드로 CPU, GPU에서 실행 가능
* 파이토치(PyTorch)
   - Facebook에서 개발
   - C/CUDA Backend 사용
   - 진입장벽이 낮음, 파이썬 문법과 유사해서 파이썬 유저들이 많이 사용(연구적 목적으로도 사용)
   - GPU 가속연산

<h3>신경망(Neural Network)</h3>

* 인공지능 분야에서 쓰이는 알고리즘
* '인간의 뇌 구조를 모방
   - 인간의 뇌 구조에는 뉴런이 있는데, 뉴런과 뉴런 사이에는 전기신호를 통해 정보를 전달한다. 전기 신호를 통해 정보를 전달하는 구조를 수학적으로 모델링을 한다.  
* 단순화하여 수학적으로 모델링한 구조
   - 입력(inputs)과 가중치(weights)를 곱한 선형구조(linear)의 개념을 만들어 낸다. 
   - 활성화 함수(activation function)를 통한 비선형 구조(non-linear) 표현 가능.

<h3>인공뉴런 vs 인공 신경망</h3>

* 인공 뉴런 (Artificial Neuron)
   - 노드(Node)와 엣지(Edge)로 표현한다.
      + 하나의 노드안에서 입력(Inputs)와 가중치(Weights)를 곱하고 더하는 선형(Linear)계산이 이루어진다. 
      + 활성화 함수(Activation Function) 통과를 모두 포함
   - 인공 신경망 (Artificial Neural Network)
      + 여러 개의 인공뉴런들이 모여 연결된 형태.
      + 뉴런들이 모인 하나의 단위를 층(Layer)이라고 하고, 여러 층(Layer)으로 이루어질 수 있음 
      + ex) 입력층(Input Layer), 은닉층(Hidden Layer), 출력층(Output Layer)

<h3>완전 연결 계층(Fully-Connected Layer)</h3>

* 모든 노드들이 서로 연결된 신경망이다.(Layer 사이)
* 밀집되어 있다 해서 Dense Layer라고도 불림
* ex) 입력 노드 3개, 은닉층1 노드 4개, 은닉층2 노드 4개, 출력층 노드 1개 -> 총 3×4×4×1=48 개의 선으로 연결

<h3>신경망의 활용-회귀(Regression)</h3>

* 잡음(Noise)을 포함한 학습 데이터로부터 어떤 규칙을 찾고 연속된 값의 출력을 추정
* 아래의 식을 만족하는 적절한 a (기울기), b (y절편)를 찾아야함
   - Y=aX+b
      + X :입력
      + Y :출력
      + a :기울기
      + b :y절편
* 회귀 문제 예시
   - 나이, 키, 몸무게에 따른 기대수명
   - 아파트의 방의 개수, 크기, 주변 상가 등에 따른 아파트 가격

<h3>신경망의 활용-분류(Classification)</h3>

* 입력값에 따라 특정 범주(category)로 구분하는 문제
   - 분류 곡선(직선)을 찾아야 함

* 범주의 개수에 따라서 이진 분류(Binary Classification), 로지스틱 회귀(Logistic Regression), 선형회귀와 비슷하지만, 범주형 데이터를 분류하는 방향으로 선을 그음
   - 면접점수, 실기점수, 필기점수에 따른 시험 합격 여부
* 다중 분류(Multi-Class Classification)
   - 꽃잎 모양, 색깔에 따른 꽃의 종 분류

<h2>4월 4일 신경망 기초수학</h2>

<pre><code>
import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn-witegrid'])
</pre>
</code>

<h3>일차함수</h3>

* y = ax + b 
   - a : 기울기,  b : y절편
* 그래프 상에서 직선인 그래프(linear)
<pre><code> 
def linear_function(x):
   a = 0.5
   b = 2
   return a*x + b
</pre>
</code>
<pre><code> 
pirnt(linear_function(5)) //x에 5를 넣은 채로 return 식 계산
-> 4.5
</pre></code>
<pre><code>
x = np.arange(-5, 5, 0.1) //start: -5, stop: 5, step: 0.1
y = linear_funtion(x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Function')
plt.show()
</pre></code>

<h3>이차함수</h3>

* y = ax^2 + bx +c 
* 일반적으로 두 개의 실근을 가짐
<pre><code> 
def quadratic_function(x):
   a = 1
   b = -1
   c = -2

   return a*x**2 + b*x +c
</pre>
</code>
<pre><code> 
print(quadratic_function(2))
-> 0
</pre></code>
<pre><code> 
x = np.arange(-5, 5, 0.1)
y = quadratic_function(x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Quadratic Function')
plt.show()
</pre></code>

<h3>삼차함수(다항함수)</h3>

* y = ax^3 + bx^2 + cx +d
<pre><code>
def cubic_function(x):
   a = 4
   b = 0
   c = -1
   d = -8
   
   return a*x**3 + b*x**2 + c*x +d
</pre> 
</code>
<pre><code> 
print(cubic_function(3))
-> 97
</pre></code>
<pre><code> 
x = np.arange(-5, 5, 0.1)
y = cubic_function(x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cubic Function')
plt.show()
</pre></code>

<h3>함수의 최소값/최대값</h3>
<pre><code> 
def my_function(x):
   a = 1
   b = -3
   c = 10

   return a*x**2 + b*x + c
</pre></code>
<pre><code> 
x = np.arange(-10, 10, 0.1)
y = my_function(x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(1.5, my_function(1.5)) //최저점 점으로 표시(1.5, 1.5 넣었을 때 y값)
plt.text(1.5-1.5, my_function(1.5)+10, 'min value of f(x)\n{}'.format(1.5, my_function(1.5)), fontsize=10)
plt.title('My function')
plt.show()
</pre></code>
<pre><code> 
min_val = min(y) //min함수: 최소값
print(min_val)
-> 7.75
</pre></code>

<h3>특정 구간 내에서 최소값 구하기</h3>
<pre><code> 
def get_minimum(x1, x2, x3)
   x = np.arange(x1, x2, 0.01)
   y = f(x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('get_minimum')
plt.show()

return min(y)
</pre></code>
<pre><code> 
print(get_minimum(1, 4, my_function))
-> 7.75
</pre></code>

<h3>지수함수/로그함수</h3>

* 지수함수-로그함수는 역함수 관계 ( y=x  직선 대칭 단, 밑이 같을 때)
* 파이썬으로 직접 구현 가능

<h3>지수함수</h3>

* y=a^x  ( a≠0 ) (기본형)
* y=e^x  ( e=2.71828... )
<pre><code> 
def exponential_function(x):
   a = 4
   return a**x
</pre>
</code>
<pre><code> 
print(exponential_function(4))
print(exponential_function(0))
</pre></code>
<pre><code> 
x = np.arange(-3, 2, 0.1)
y = exponential_function(x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-1, 15) //y값 -1~15로 범위 지정
plt.xlim(-4, 3) //x값 -4~3로 범위 지정
plt.title('exponential_function')
plt.show()
</pre></code>
<pre><code> 
def exponential_function2(x):
   a = 4
   return math.pow(a, x)
</pre></code>
<pre><code> 
print(exponential_function(4))
print(exponential_function(0))

-> 256.0
   1.0
</pre></code>

<h3>밑이 e인 지수함수 표현</h3>
<pre><code> 
print(math.exp(4))
print(np.exp(4))

-> 54.598150033144236
   54.598150033144236
</pre></code>

<h3>로그함수</h3>

* y=loga(x)  (a≠1) (기본형)
* y =log10(x) (상용로그)
* y =ln(x) (밑이 e인 자연로그)
<pre><code> 
print(math.log(2, 3))
print(np.log2(4)) //밑이 2라고 지정
print(np.log(4)) //맡이 자동으로 2라고 지정

-> 0.6309287535714574
   2.0
   1.3862943611198906
</pre>
</code>

<h3>역함수 관계</h3>

* y = x 대칭
<pre><code> 
x = np.arange(-1, 5, 0.01)
y1 = np.exp(x)

x1 = np.arange(0.000001, 5, 0.01)
y2 = np.log(x2)

y3 = x

plt.plot(x, y1, 'r-', x2, y2, 'b-', x, y3, 'k--') //x, y1은 지수함수, x2, y2는 로그함수, x, y3는 직선 

plt.ylim(-2, 6)
plt.axvline(x=0, color='k') //0을 중심으로 x축
plt.axhline(y=0, color='k') //0을 중심으로 y축

plt.xlabel('x')
plt.ylabel('y')
plt.show()
</pre>
</code>

<h3>함수 조작</h3>

* y=−loga(x) 와 y=−loga(1−x)
   - x=0.5 대칭
* Logistic Regression을 위한 함수
<pre><code> 
x = np.arange(-10, 10, 0.01)
y1 = -np.log(x)
y2 = -np.log(1-x)

plt.axvline(x=0, color='k')
plt.axhline(y=0, color='k')

plt.grid()
plt.plot(x, y1, '-b', x, y2, '-r')
plt.text(0.9, 2.0, 'y=-log(1-x)', fontsize=15)
plt.text(0.1, 3, 'y=-log(x)', fontsize=15)
plt.xlim(-0.3, 1.4)
plt.ylim(-0.5, 4)
plt.scatter(0.5, -np.log(0.5)) //대칭되는 지점 0.5를 점으로 표현
plt.show()
</pre>
</code>

<h3>극한</h3>

* 극한에 대해서는 어떠한 식을 코드로 표현할 수 있다 정도로만 이해하며 참고
* 극한에서 알아야 사실은 x가 어떤 값 a에 가까이 다가갈 때 a에 '한없이 가까이 간다'일 뿐, a에 도달하지 않는다는 점
* 이를 표현할 때, 엡실론(epsilon)이라는 아주 작은 값(ex, 0.0001) 등으로 표현
<pre><code> 
from sympy import *
init_printing()

x, y, z = xymbols('x y z')
a, b, c, t = symbols('a b c t')
</pre>
</code>

* lim x → 1 (x^3−1 / x−1) = 3
<pre><code> 
print("극한값:", limit(x**3-1) / (x-1), x, 1)

plot(((x**3-1) / (x-1)), xlim=(-5, 5), ylim=(-1, 10));
-> 3
</pre>
</code>

* lim x → ∞ (1+x / x)
<pre><code> 
print("극한값:", limit(1 + x) / x, x, oo) //oo = 무한대

plot((1+x) / x, xlim=(-10, 10), ylim=(-5, 5));
</pre>
</code>

* lim x → 1 ((√x+3)−2 / x−1) = 14
<pre><code> 
print("극한값:", limit((sqrt(x+3)-2) / (x-1), x, 1))

plot((sqrt(x+3)-2) / (x-1), xlim=(-5, 12), ylim=(-0.5, 1));
</pre>
</code>

<h3>삼각함수의 극한</h3>

* lim x → π/2+0 tanx=−∞ 
* lim x → π/2−0 tanx=∞ 
<pre><code> 
print("극한값:", limit(tan(x), x, pi/2, '+'))
print("극한값:", limit(tan(x), x, pi/2, '-'))

plot(tan(x), xlim=(-3.14), ylim=(-6, 6));
</pre>
</code>

* lim x → 0 (sinxx)=1
<pre><code> 
print("극한값:", limit(sin(x)/x, x, 0))

plot(sin(x)/x, ylim=(-2, 2));
</pre>
</code>

* lim x → 0 xsin(1/x)
<pre><code> 
print("극한값:", limit(x * sin(1/x), x, 0))

plot(x*sin(1/x), xlim=(-2, 2), ylim=(-1, 1.5));
</pre>
</code>

<h3>지수함수, 로그함수의 극한</h3>

* lim x → ∞ ((2^x) − 2^(−x) / 2^x + 2^−x)
<pre><code> 
print("극한값:", limit((2**x - 2**(-x)) / (2**x + 2**(-x)), x, oo))

plot((2**x - 2**(-x)) / (2**x + 2**(-x), xlim=(-10, 10), ylim==(-3, 3));

-> 극한값: 1
</pre>
</code>

* lim x → ∞ (log2(x+1) − log2(x))=0
<pre><code> 
print("극한값:", limit( log(x+1, 2) - log(x, 2), x, oo))

plot( log(x, 2), log(x+1, 2), xlim=(-4, 6), ylim(-4, 4));

-> 극한값: 0 // 두 개의 로그함수가 0으로 극한하고 있음을 볼 수 있음
</pre>
</code>

<h3>자연로그(e)의 밑</h3>

* lim x → ∞ (1 + 1/x)^x = e 
* lim x → ∞ (1 + (2/x)^x) = e^2 
* lim x → 0 ((e^x)−1) / x = 1 
* lim x → 0 ln(1+x) / x = 1
<pre><code> 
print("(1):", limit(1+1/x)**x, x, oo)
print("(2):", limit((1+2x/2)**x, x, oo)
print("(3):", limit((exp(x) - 1) / x), x, 0))
print("(4):", limit(ln(1+x) / x, x, 0))

plot(ln(1+x) / x, xlim=(-4, 6), ylim=(-2, 8));

-> (1): E
   (2): exp(2)
   (3): 1
   (4): 1
</pre>
</code>

<h3>미분</h3>

* 어떤 한 순간의 변화량을 표시한 것

<h3>미분과 기울기</h3>

* 어떤 함수를 나타내는 그래프에서 한 점의 미분값(미분계수)를 구하는 것은 해당 점에서의 접선을 의미
* 기울기는 방향성을 가짐
   - 이용할 미분 식 (수치 미분)
      + df(x)/dx = lim x → ∞ f(x+h)−f(x−h) / 2h 
* [ 주의 ] h는 아주 작은 수를 뜻하는데, 예를 들어 10e−50 정도의 수를 하면 파이썬은 이를 0.0으로 인식
* 따라서, 딥러닝에서 아주 작은 수를 정할 때 1e−4 정도로 설정해도 무방
<pre><code>
def numerical_differential(f, x):
   h = 1e-4
   return(f(x+h)-f(x-h)) /  (2*h)
</pre>
</code>

<h3>함수 위의 점  (a,b) 에서의 접선의 방정식</h3>

* 예제 : 점 (1, 7) 에서의 기울기
<pre><code> 
def my_fun(x):
   return 2*x**2 + 3*x +2
</pre>
</code>
<pre><code> 
def linear_func(a, b, c, x):
   return c*(x-a) +b
</pre></code>
<pre><code> 
c = numericla_differential(my_func, 1)

x = np.arange(-5, 5, 0.01)
y = linear_func(1, my_func(1), c, x)

plt.xlabel('x')
plt.ylabel('y')
plt.scatter(1, my_func(1))
plt.plot(x, my_func(x), x, y, 'r-')
plt.title('f(x) & linear function')
plt.show()
</pre></code>

<h3>미분 공식</h3>

* d / dx * (c) = 0 (c는상수) 
* d / dx * [ cf(x) ] = cf′(x) 
* d / dx * [f(x) + g(x)] = f′(x) + g′(x) 
* d / dx * [f(x) − g(x)] = f′(x) − g′(x) 
* d / dx * [f(x) * g(x)] = f(x)g′(x) + f′(x)g(x) (곱셈공식) 
* d / dx * [f(x) / g(x)] = g(x)f′(x) − f(x)g′(x) / [ g(x) ]^2 
* d / dx * [ xn ] = nx^n − 1

<h3>편미분</h3>

* 변수가 1개짜리인 위의 수치미분과 달리, 변수가 2개 이상일 때의 미분법을 편미분이라 함
* 다변수 함수에서 특정 변수를 제외한 나머지 변수는 상수로 처리하여 미분을 하는 것
* 각 변수에 대해 미분 표시를  σ 를 통해 나타남
* ex) f(x0, x1)=x0^2 + x1^2
<pre><code> 

</pre>
</code>
<pre><code> 

</pre></code>

* 예제1: x0에 대한 편미분,  σf / σx0
<pre><code> </pre></code>

* 예제2: x1에 대한 편미분,  σf / σx1
<pre><code> </pre></code>

<h3>기울기(gradient)</h3>

* 방향성을 가짐
<pre><code> </pre></code>
<pre><code> </pre></code>

<h3>기울기의 의미를 그래프로 확인</h3>

* 기울기가 가장 낮은 장소(가운데)로부터 거리가 멀어질수록 기울기가 커짐
* 기울기가 커진다는 것은 영향을 많이 받는다는 의미
* 기울기가 작다는 것은 영향을 적게 받는다는 의미
<pre><code> </pre></code>








<h3></h3>
<h3></h3>
<h3></h3>
<h3></h3>
<h3></h3>
<h3></h3>
<h3></h3>


<pre><code> </pre></code>
<pre><code> </pre></code>

<pre><code> </pre></code>

<pre><code> </pre></code>

<pre><code> </pre></code>

<pre><code> </pre></code>



