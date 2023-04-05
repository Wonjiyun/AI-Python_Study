# 2023.04.05 실습

# import math
# import numpy as np
# import matplotlib.pyplot as plt
# def linear_function(x):
#     a = 0.5
#     b = 2
#     return a*x + b
# x = np.arange(-5, 5, 0.1)
# y = linear_function(x)
# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Linear Function')
# plt.show()

# import math
# import numpy as np
# import matplotlib.pyplot as plt
# def quadratic_function(x):
#     a = 1
#     b = -1
#     c = -2
#     return a*x**2 + b*x + c
# x = np.arange(-5, 5, 0.1)
# y = quadratic_function(x)
# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Quadratic Function')
# plt.show()

# import math
# import numpy as np
# import matplotlib.pyplot as plt
# def cubic_function(x):
#     a = 4
#     b = 0
#     c = -1
#     d = -8
#     return a*x**3 + b*x**2 + c*x + d
# x = np.arange(-5, 5, 0.1)
# y = cubic_function(x)
# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Cubic Function')
# plt.show()

# import math
# import numpy as np
# import matplotlib.pyplot as plt
# def my_func(x):
#     a = 1
#     b = -3
#     c = 10
#     return a*x**2 + b*x + c
# x = np.arange(-10, 10)
# y = my_func(x)
# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.scatter(1.5, my_func(1.5))
# plt.text(1.5-1.5, my_func(1.5)+10, "min value of f(x)\n({}, {})".format(1.5, my_func(1.5)), fontsize=10)
# plt.title('my_func')
# # plt.show()

# min_val = min(y)
# print(min_val)

# import math
# import numpy as np
# import matplotlib.pyplot as plt
# def get_minimum(x1, x2, f):
#     x = np.arange(x1, x2, 0.01)
#     y = f(x)
#     plt.plot(x, y)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title('get_minimum')
#     plt.show()
#     return min(y)
# def my_func(x):
#     a = 1
#     b = -3
#     c = 10
#     return a*x**2 + b*x + c
# print(get_minimum(1, 4, my_func))


# import math
# import numpy as np
# import matplotlib.pyplot as plt
# def exponential_function(x):
#     a = 4
#     return a**x
# x = np.arange(-3, 2, 0.1)
# y = exponential_function(x)
# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.ylim(-1, 15)
# plt.xlim(-4, 3)
# plt.title('exponential function')
# plt.show()

import math
import numpy as np
import matplotlib.pyplot as plt
def exponential_function(x):
    a = 4
    return a**x
x = np.arange(-3, 2, 0.1)
y = exponential_function(x)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-1, 15)
plt.xlim(-4, 3)
plt.title('exponential function')
plt.show()