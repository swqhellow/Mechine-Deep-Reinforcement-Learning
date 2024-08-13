# abs()
print('The absolute value of -10 is {}'.format(abs(-10)))
# divmode()
res = divmod(7, 2)
print(f'7 divided by 2 equals {res[0]} leaving {res[1]}')
# sum()
print(f'the summary of the list {[1,2,3]} is {sum([1,2,3])}')
# round()
num = 6.6
print("{} half adjusts is {}".format(num, round(num)))
print(f"{num:.2f} half adjusts is {num:.0f}")
# pow()
print(f"2 to the power of 3 is {pow(2,3)}")
# min() and max()
ls = [1, 4, 2, 6, 9]
print(f'the max and min numbers in the\
 list:ls is {max(ls)} and {min(ls)}')
# hex()bin()bool()
print(f"""the hexadecimal of {15} is {hex(15)} 
              binary is {bin(15)}
              bool is {bool(num)}""")
# ord()
print(f"the ASCII of the char:{'a'} is {ord('a')}")
# chr()
print(f"the num:{65} is {chr(65)} in ASCII")
# list
print(
    f"list can turn a Iterative object into list \n such as {list(range(1,10))}")
# open
f1 = open(r'Learning\data\rnndata.txt', 'rb')
print(f"function:open can open a file and return a funchion\
      \nsuch as {f1.readline()}")
f1.close()
# range()
print(f"range(start,end,step) can return a range of num\
      \nsuch as range(1,10,2) is {list(range(1,10,2))}")
# enumerate()
print(f"enumerate() is used to return index and\
 value of a list,arry\
 such as\
     \nfor index,value in enumerate(range(1,3)):")
for i,num in enumerate(range(1,3)):
    print(f'{i}: index:{i}, value:{num}')
# set()
print(f"set() can return a list without having two same values\n\
      list:ls is[1,1,2,2,4,4,7,7],use the set() \n\
      result:{set([1,1,2,2,4,4,7,7])}")
# all()
print(f"if all element in the specified sequence are True, return True:\
 if there is a False, return False.example:")
print(f"all([1,2,4,6])={all([1,2,4,6])}")
print(f"all([1,2,0,6])={all([1,2,0,6])}")
# any()
print("if element in the specified sequence are False,\
 return True if there is any True")
print(f"any([1,2,0,6])={any([1,2,0,6])}")
print(f"any([0,0,0,0])={any([0,0,0,0])}")
# sort()
print(f"sort() can Sort the list,the default is ascending order\
 if you want to reverse order , you can add (reverse=True)")
ls = [1,4,9,2,7]
ls.sort()
print(f'list:{[1,4,9,2,7]} sorted is {ls},')
ls.sort(reverse=True)
print(f'reverse is {ls}')
print(f'''
notice the method sort() will not return a list ,
it will actually sort the elements of the list.
so you can get the same result using the code fellows：
ls = [1,24,7,2]
a = ls.sort()
print(a,ls)''') 
ls = [1,24,7,2]
a = ls
a.sort()
print(f'a and ls is the same{a,ls}')
print(f"""if you want to get the different result after
the assigment use a = ls.copy()
""") 
ls = [1,24,7,2]
a = ls.copy()
a.sort()
print(f'a and ls is different: {a,ls}')
# sorted()
a=[['a','b','B'],['c','e','A'],['c','f','D']]
print('sorted() can sort the char or string \
\n%s'%format(sorted(a)))
# len()
print("len() can return the length of an object \
\n the length of list:[1,2,4,5,3] is {}".format(len([1,2,4,5,3])))
import cv2
import numpy as np
import math

# 创建一个空白图像
image = np.zeros((500, 500, 3), dtype='uint8')

# 定义五角星的中心和半径
center = (250, 250)
radius = 100

# 计算五角星的顶点
points = []
for i in range(5):
    angle = i * 144 * math.pi / 180  # 144度是五角星每个顶点之间的角度
    x = int(center[0] + radius * math.cos(angle))
    y = int(center[1] - radius * math.sin(angle))
    points.append((x, y))

# 将顶点列表转换为 NumPy 数组
points = np.array(points, np.int32)
points = points.reshape((-1, 1, 2))

# 使用 cv2.fillPoly 函数绘制实心五角星
cv2.fillPoly(image, [points], color=(255, 255, 255))

# 显示图像
cv2.imshow('Solid Star', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
