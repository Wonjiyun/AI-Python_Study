# AI-Python_Study
<h2>3월 29, 30일 python 노마드</h2>

<h3>variable</h3>

* my_age: Sanke Case 
* myAge: Camel Case 
* 첫 시작 숫자, 특수문자 공백 X

<h3>variable data type </h3>

* 숫자
* "문자"
* boolean: True/False, 0/1, off/on

<h3>function</h3>

* print()
* start with def + function

<pre>
<code>
def say_hello():
  print("hello how are you?")

say_hello()

-> Hello how are you?
</code>
</pre>

* function()에서 ()-> parameter: 문자열 X, variable 형식, function 안에서 쓸 수 있는 variable
* 우리가 직접 데이터를 function에 넣고 function은 이 데이터를 받아 사용함
* parameter는 함수로 전달하는 데이터를 저장하기 위한 palceholder일 뿐임

<pre>
<code>
def say_hello(user_name):
  print("Hello", username, "how are you?")

say_hello("nico")
say_hello("lynn")
say_hello("lewis")

-> Hello nico how are you?
   Hello lynn how are you?
   Hello lewis how are you?
</code>
</pre>

<h3>multi parameters</h3>

<pre>
<code>
def say_hello(user_name, user_age):
  print("Hello", user_name)
  print("you are", user_age, "years old")

say_hello("nico", 12)

-> Hello nico
   you are 12 years old 
</code>
</pre> 

<h2>4월 3일 python 노마드</h2>
<h3>Recap</h3>

<pre>
<code>
def tax_calculator():
  print(150000000000 * 0.35) // 계좌 잔액 * 세금 비율 

tax_calculator()
</code>
</pre>

* 함수를 좀 더 사용자화 하며, 함수에 우리의 데이터를 주고 함수에 매번 다른 값을 보낼 수 있게 할 때 parameter가 필요하다.
* 그래서 함수가 데이터를 받을 수 있게 하려면 해당 데이터를 위한 공간을 만들어줘야 한다. 그 공간은 함수 선언문의 괄호 안이다.
* 괄호 안 placeholder 명은 자유로 지정할 수 있다.

<pre>
<code>
def tax_calculator(money):
  print(money * 0.35) // 계좌 잔액 * 세금 비율 

tax_calculator(150000000000)
</code>
</pre>

<h3>Default Parameters</h3>

<pre>
<code>
def say_hello(user_name):
  print("hello", user_name)

say_hello("nico")
say_hello() // say_hello는 1개의 argument를 여기서 받아야 하는데 아무런 argument도 보내지 않았기 때문에 에러가 일어남

-> hello nico
   error
</code>
</pre>

<pre>
<code>
def say_hello(user_name="anonymous"): // 밑에서 argument를 받지 않았을 때 에러 대신 user_name을 'anonymous'로 설정해달라는 것
  print("hello", user_name)

say_hello("nico")
say_hello() 

-> hello nico
   anonymous
</code>
</pre>

* 각기 다른 수학 연산자를 사용하는 함수를 이용하여 계산기 만들기
<pre>
<code>
def plus(a, b)
  print(a + b)

def minus(c, d)
  print(c - d)

def multiple(e, f)
  print(e * f)

def devide(g, h)
  print(g / h)

def powerof(i, j)
print(i**j)
</code>
</pre>

<h3>Return Values</h3>

<pre>
<code>
def tax_calculator(money):
  print(money * 0.35) // 계좌 잔액 * 세금 비율 

tax_calculator(150000000000) 
</code>
</pre>

* tax_calculator 함수의 결과를 받아서 나중에 내 코드에 쓰고 싶을 때
  = 함수로부터 값을 받아내기

<pre>
<code>
def tax_calculator(money):
  return money * 0.35 //return이란 함수 바깥으로 값을 보낸다는 의미

def pay_tax(tax): //이 함수는 내야 하는 세금을 받아서 
  print("thank you for paying", tax) //이와 같이 출력해준 후 tax 변수를 출력
to pay = tax_calculator(150000000000) 
pay_tax(to_pay)
</code>
</pre>

<h3>Return Recap</h3>

* 문자열 안에 변수를 넣는 기능
<pre>
<code>
my_name = "nico"
my_age = 12
my_color_eyes = "brown"

print(f"Hello I'm {my_name}, I have {my_age} years in the earth, {my_color_eyes} is my eye color") //f=format
</code>
</pre>
<pre>
<code>
def make_juice(fruit):
  return f"{fruit} +!"

def add_ice(juice):
  return f"{juice}+@"

def add_sugar(iced_juice):
  return f"{iced_juice}+%"

juice = make_juice("*")
cold_juice = add_ice(juice)
perfect_juice = add_sugar(cold_juice)

print(perfect_juice)
</code>
</pre>

<h2>4월 3일 python 노마드</h2>

<h3>IF</h3>
<pre><code>
if condition: 
  "write the code to run" //condition을 만족하면 print할 문장 

ex)
a = 10

if 10 == 10:
  print("True!")

-> True!
</code></pre>

<h3>Else & Elif</h3>
<pre><code>
ex)
password_correct = True

if password_correct:
  print("Here is your money")
else: //if의 조건이 false일 때
print("Wrong password")

-> Here is your money
</code></pre>
<pre><code>
ex)
password_correct = False

if password_correct:
  print("Here is your money")
else: //if의 조건이 false일 때
print("Wrong password")

-> Wrong password
</code></pre>

<pre><code>
ex)
winner = 10

if winner > 10:
  print("Winner is greater than 10")
elif winner < 10: //elif는 또 다른 대안과 조건을 넣을 수 있도록 한다.
  pirnt("Winner is less than 10")
else: 
  print("Winner is 10")

-> Winner is 10
// python은 먼저 if의 조건문을 읽는다 다음 elif의 조건문을 읽는다.
</code></pre>
ex)
winner = 20

if winner > 10: //if 조건이 true라면 뒷부분의 조건은 동작하지 않는다.
  print("Winner is greater than 10")
elif winner < 10: //elif는 또 다른 대안과 조건을 넣을 수 있도록 한다.
  pirnt("Winner is less than 10")
else: 
  print("Winner is 10")

-> Winner is greater than 10

<h3>Recap</h3>
<h3>And & Or</h3>
<pre><code>
age = input("How old are you?") //input은 오직 하나의 argument만 받는다

print("user answer", age)
print(type(age))

-> How old are you? [입력 받기] 12
   user answer 12
   < class 'str' > //sting type
</code></pre>
string을 숫자로 변환
<pre><code>
age = int(input("How old are you?"))

if age < 18:
  print("You can't drink.")
else:
  print(Go ahead")

-> How old are you?
   16 [입력 받기]
   You can't drink
</code></pre>
<pre><code>
age = int(input("How old are you?"))

if age < 18:
  print("You can't drink.")
elif age > 18 and age < 35: // and일 경우 앞 뒤 조건 둘 다 True여야 True임 하나라도 False면 False임
  print("You drink beer")
elif age == 60 or age == 70: // or일 경우 앞 뒤 조건 둘 중 하나만 True여도 True임
  pring("Birthday party!")
else:
  print(Go ahead")

-> How old are you?
   60 [입력 받기]
   Birthday party!
</code></pre>
True and True == True
False and True == False
True and False == False
False and False == False

True or True == True
True or False == True
False or True == True
False or False == False

<h3></h3>
<h3></h3>
<h3></h3>
<pre><code>
</code></pre>
<pre><code>
</code><pre>
<pre><code>
</code><pre>
<pre><code>
</code><pre>
<pre><code>
</code><pre>