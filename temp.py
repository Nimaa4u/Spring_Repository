# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
#ages = {'ME-205-02':18,'ME-205-04':19,'ME-205-01':23,'jim':24}

ages=[1,4,5,3,6,7,8,2,4,5,6,3,4,1,2,7,8,9,0]

data_dic={}

for i in ages:
    #print (i)
    #print (data_dic)   
    if i in data_dic:
        #print ('i', i)
        data_dic[i] = data_dic[i] + 1
        print ('i=',i,', item=',item,', data=', data_dic)
    else: 
        data_dic[i] =1
        print ('i=',i,', item=',item,', data=', data_dic)
#%%
def mode(mylist):
    count_dic={}
    
    for item in mylist:
        print('item',item)
        if item in count_dic:
            print ('count_dic',count_dic)
            count_dic[item] = count_dic[item] + 1
            print ('count_dic[item] ==>',count_dic[item])
        
        else: 
            count_dic[item] =1
            print('(1) count_dic[item] -->',count_dic[item])
    count_list = count_dic.values()
    print('COUNT_LIST',count_list)
    max_count = max (count_list)
    
    mode_list = []
    
    for item in count_dic: 
        if count_dic [item]==max_count:
            mode_list.append(item)
            
        
    return mode_list

#%%
def median(mylist):
    copylist = mylist[:]
    copylist.sort()

    print (copylist[len(copylist)//2])
    
    print (copylist[len(copylist)//2+1])

    if len(copylist) % 2 == 0:
        median = (copylist[len(copylist)//2-1] + copylist[len(copylist)//2])/2        
    else:
        median =copylist[len(copylist)//2]
    return median
#%%

mylist=list(range(0,100,13))

for item in mylist[0:]:
    print (item)

#%%
def mean_value(mylist):
    print (len(mylist))
    print(sum(mylist))
    mean_value = sum(mylist) / len(mylist)
    return mean_value

#%%


def getmax(mylist):
    
    max_value = mylist[0]
    
    for item in mylist[1:]:
        
        if max_value < item:
            max_value = item
            
    return max_value




#%%

def getmin(mylist):
    min_value = mylist[0]
    
    for i in range(1,len(mylist)):
        if min_value > mylist[i]:
            min_value = mylist[i]
    return min_value

            
#%%

def getmax(mylist):
    max_value=mylist[0]
    
    for i in range(1,len(mylist)):
        if max_value < mylist[i]:
            max_value = mylist[i]
    return max_value

#%%
def getrange(mylist):
    return max(mylist)-min(mylist)
#%%
mylist=[1,2,3,4,5,100,-100]
max(mylist)-min(mylist)

#%%
Name = "GHAZAL GHARAEE"

for i in range(len(name)):
    print(name[0:i])
    



#%%
import random 
import math 

def monte(points):
    
    inside_point = 0 
    
    for i in range(points):
        
        x = random.random()
        y = random.random()
        
        threshold = (x**2 + y**2) **0.5
        #print(i)
        
        if threshold <=1 :
            inside_point = inside_point +1 
                                   
            if i>0:
                pi = inside_point/i * 4
                print("Tot.=", i ,"In=", inside_point,"pi=",pi )
                       
    pi = inside_point/points * 4 
    return pi
#%%
import random 
random.random()

for sina in range (100):
    print(sina, random.random())
    
#%%

a= 50 
b= 30

print ("a = ", a)

if a > b:
    c= 10
else:
    c=20
    
print ("c = ", c) 
#%%
def wallis (pairs):
    acc = 1 
    num = 2
    
    for pairs in range (pairs):
        left_term = num / (num-1)
        right_term = num / (num +1)
        acc = acc * left_term * right_term 
        num = num +2 
        
    pi = acc * 2 
    return pi

#%%
acc = 0
for x in range (1,10):
    acc= acc + x
    print (x, acc)

#%% 
 
import math 
def arch(num_sides):
    inner_angle_B = 360 / num_sides
    half_angle_A = inner_angle_B / 2
    one_half_side = math.sin(math.radians(half_angle_A))
    sides = one_half_side *2 
    
    Poly = num_sides * sides 
    pi = Poly / 2 
    return pi
#%%
for NIMA in range(3,100,30):
    print (NIMA)
#%%
pi = 3.14
radious = 8.0
height = 16 

basearea = pi * radious ** 2

print("basearea", basearea)

cylindervolume = basearea * height 

cylindervolume

#%%

import math 
numsides = 8 
innerangleb = 360 / numsides 
halfanglea = innerangleb/2
onehalfsides=math.sin(math.radians(halfanglea))
sides = onehalfsides * 2
poly=numsides*sides
pi=poly/2
#%%

nima = [3,"cat",5, "nima"]

sina = [1,2,"babi",3,4,"father"]

gholam = [1,2,3,4]
#%%
mylist = [7,9,'a','ca',False]
#%%
def getrange(nimalist):
    return max(nimalist) - min(nimalist)

#%%
def getmax(glist):
    maxsofar=glist[0]
    for item in glist:
        print("ITEM = ",item)
        if item > maxsofar:
           maxsofar = item
           print ("MAXSOFAR", maxsofar)   
    return maxsofar
#%%
    
def getmin(minlist):
    minsofar=minlist[0]
    for item in minlist[1:]:
        if item < minsofar:
            minsofar = item
            print("MINSOFAR", minsofar)
    return minsofar
       
#%%
def mean(llist):
    mean = sum (llist)/len(llist)
    return mean 
#%%
    #list
def median(alist):
    copylist = alist[:]
    print ("copylist", copylist)
    copylist.sort()
    print ("sort", copylist)
    print (len(copylist))
    if len(copylist)%2 == 0:
        rightmid =len(copylist)//2
        leftmid =rightmid-1
        median = (copylist[leftmid]+copylist[rightmid])/2
        print (copylist[leftmid],copylist[rightmid])
    else: 
        print("else")
        mid = len (copylist)//2
        median = copylist[mid]
    return median 
#%%
#dictionary 

ages = {'david':45, 'brenda':46, 'nima':34, 'ghazal':20, 'Behnaz':29}    
for k in ages.keys():
    print(k)      
#%%
#

def mode(alist):
    countdict = {}
    print (countdict)
    for item in alist:
        if item in countdict:
            countdict[item] = countdict[item] + 1 
        else: 
            countdict[item] = 1
    countlist = countdict.values()
    print("countlist values",countlist)
    maxcount = max(countlist)
    
    modelist = []
    for item in countdict:
        if countdict[item] == maxcount:
            modelist.append(item)
    return modelist 
#%%
import random
import math

def carlo(time):
    
    incircle = 0 
    
    for i in range(time):
        x=random.random()
        y=random.random()
        #print (i)
        d = math.sqrt(x**2 + y**2)
        
        if d <= 1: 
            incircle = incircle + 1 
            
    pi = incircle / time * 4
        
    return pi
#%%
# page 55
acc =0 
for x in range(1,6):
    print (x)
    acc = acc + x 
print (acc)    
#%%

def leibniz(terms):
    acc = 0
    num = 4 
    den = 1 
        
    for aterm in range(terms):
        
        den = den +2 
        
    return acc
#%%
a = 5 
b = 3 
if a>b: 
    c=10 
else:
    c=20
#%%
# page 78

import random 
import math 
import turtle

def monte (numsim):
    wn = turtle.Screen()
    drawingt = turtle.Turtle()
    wn.setworldcoordinates(-2,-2,2,2)
    drawingt.up()
    drawingt.goto(-1,0)
    drawingt.down()
    drawingt.goto(1,0)
    
    drawingt.up()
    drawingt.goto(0,1)
    drawingt.down()
    drawingt.goto(0,-1)
    
    circle = 0 
    drawingt.up()
    
    for i in range(numsim):
        x = random.random()
        y = random.random()
        
        d = math.sqrt(x**2 + y**2)
        drawingt.goto(x,y)
        
        if d<= 1:
            circle = circle + 1 
            drawingt.color("blue")
        else:
            drawingt.color("red")
        drawingt.dot()
        
    pi = circle / numsim * 4 
    wn.exitonclick()

    return pi
#%% 
#page 87 
name = "Where Is My Love"
for i in range(len(name)):
    print(name[i])
#%%
name = "Ghazal Gharaee"
for i in range(len(name)):
    print(name[0:i+1])    
#%%    
for i in range(1000):
    print(i % 7, end= '')
#%%
def mode(alist):
    countdic = {}
    
    for item in alist:
        print("START OF LOOP ==>      ",item)
        if item in countdic: 
            countdic[item] = countdic[item] +1 
            print ("countdic[item]         ",countdic[item])
            print('countkey =========>>>>>>>>>>>', countdic.keys())
            print('countdic',countdic.values())
            print('countdic', countdic.items())
        else:
            countdic[item] = 1
    
    print("countdic ======>>>    ", countdic.items())    
    countlist = countdic.values()
    print("countdic value              ", countdic.values())
    print("countlist                 ",countlist)
    maxcount = max(countlist)
    
    modelist = [ ]
    if item in countdic:
        if countdic[item] == maxcount:
            modelist.append(item)
            
    return modelist
#%%    
import numpy as np
x = np.array([0,0,0])
c = np.array([175,90,160])
a = np.array([[2,1,4],[3,2,1],[1,3,3]])
b = np.array([630,550,600])

x3_only = int(np.min(b / a[:,2]))
lhs = np.array([np.sum(a[0]),np.sum(a[1]),np.sum(a[2])])
x.fill(int(np.min(b / lhs)))
obj_fun_val = x.dot(c)
print('All quantities the same:')
print('x:',x, 'Profit:',obj_fun_val)

#%%
import scipy.stats as sst
import numpy as np
import matplotlib.pyplot as plt

µ = 100
σ = 10
α = 0.05
x = sst.norm.ppf(1-α)

xv = np.linspace(µ-3*σ, µ+3*σ, 100, endpoint=True)     # 100 X-Axis points within ± 3σ of µ
print (xv)

yv = sst.norm.pdf(xv,µ,σ)   
print('yv=', yv)                           # 100 Y-Axis values

plt.plot(xv,yv,color='blue')			          # Show an X-Y chart

xvx = np.linspace(µ-3*σ, x, 100, endpoint=True)        # 100 X-Axis points from µ-3σ to x
yvx = sst.norm.pdf(xvx,µ,σ)                            # 100 Y-Axis values

plt.fill_between(xvx,0, yvx,color='red', alpha=0.25)   # Show a Filled X-Y Chart

txtx = (µ-3*σ + x)/2           
print(txtx)       
                 # X location for annotation text
txty = sst.norm.pdf(txtx,µ,σ)/2                        # Y location for annotation text
txt = 'P(X<=' + str(x) + ')'                           # annotation text                          
plt.annotate(txt,xy=(txtx,txty), fontsize=10)          # show annotation text 

plt.plot([µ-3.25*σ, µ+3.25*σ],[0,0],color='black', linewidth=0.75)  # Draw X-Axis line
plt.show()         
#%%
f =  open('D:\grades.txt')
n = 0
sum = 0.0
for line in f:
    line_items = line.split(',')
    student_name = line_items[0] + ' ' + line_items[1]
    major = line_items[2]
    grade = float(line_items[3])
    print(student_name, major, grade)     
    
#%%
f =  open('D:\grades.txt')
grade_by_major = {}
print('START FROM HERE')
print('grade_by_major=     ', grade_by_major)

n = 0
sum = 0.0
for line in f:
    print('============================================================')
    print ('LINE NUMBER = ', n )
    print('============================================================')
    
    line_items = line.split(',')
    print (line_items)
    major = line_items[2]
    grade = float(line_items[3])
    grade_by_major.setdefault(major,[0,0])
    print('---------------------------')
    print(grade_by_major)
   
    print('print= ', grade)
    grade_by_major[major][0] = grade_by_major[major][0] + grade
    grade_by_major[major][1] = grade_by_major[major][1] + 1
    
    print ('grade_by_major[major][0] = ', grade_by_major[major][0])
    print ('grade_by_major[major][1] = ', grade_by_major[major][1])
    print(grade_by_major)
    print('************************************************************')
    n=n+1
print (grade_by_major.items())    
for k, v in grade_by_major.items():
    print('%s\t%0.2f' % (k,v[0]/v[1]))
#%%
ages = {}

for i in range(10):
    i_value = i
    ages.setdefault(i_value,[0,0])
    for j in range(20):
        nima = 'nima'
        ages[i_value][0]=i
        ages[i_value][1]=j
print (ages)
#%%


f =  open('D:\On_Time_On_Time_Performance_1998_1.csv')
grade_by_major = {}
print('START FROM HERE')
print('grade_by_major=     ', grade_by_major)

n = 0
sum = 0.0
for line in f:
    if n == 0:
       n=n+1 
       exit
    else:
        
        print('============================================================')
        print ('LINE NUMBER = ', n )
        print('============================================================')
        
        line_items = line.split(',')
        print ('print line', line_items)
        print('major')
        major = line_items[0]
        print(major)
        print(line_items[33])
        grade = float(line_items[33])
        grade_by_major.setdefault(major,[0,0,0])
        print('---------------------------')
        print(grade_by_major)
       
        print('print= ', grade)
        grade_by_major[major][0] = grade_by_major[major][0] + grade
        grade_by_major[major][1] = grade_by_major[major][1] + 1
        
        print ('grade_by_major[major][0] = ', grade_by_major[major][0])
        print ('grade_by_major[major][1] = ', grade_by_major[major][1])
        print(grade_by_major)
        print('************************************************************')
        n=n+1
    
print (grade_by_major.items())    
for k, v in grade_by_major.items():
    print('%s\t%0.2f' % (k,v[0]/v[1]))
#%%
import time
    
f =  open('D:\On_Time_On_Time_Performance_1998_1.csv')
grade_by_major = {}
print('START FROM HERE')
print('grade_by_major=     ', grade_by_major)

n = 0
j = 0
sum = 0.0

for line in f:
    
    if n == 0:
        line_items = line.split(',')
        print ('N= ', n)
        n=n+1
        for j in range(50):                      
            print('J= ',j , 'line_itemsn= ', line_items[j])
    elif n > 0:
        line_items = line.split(',')
        major = line_items[0]
        print('YEAR = ', major)
        print('line_items = ', line_items[33])
        if line_items[33] == '': 
            grade = 0
        else:
            grade = float(line_items[33])
        
        grade_by_major.setdefault(major,[0,0,0])
        print('---------------------------')
        print(grade_by_major)
        
        print('print= ', grade)
        grade_by_major[major][0] = grade_by_major[major][0] + grade
        grade_by_major[major][1] = grade_by_major[major][1] + 1
        print ('grade_by_major[major][0] = ', grade_by_major[major][0])
        print ('grade_by_major[major][1] = ', grade_by_major[major][1])
        print(grade_by_major)
        print('************************************************************')
        n=n+1
    
print (grade_by_major.items())    
for k, v in grade_by_major.items():
    print('%s\t%0.2f' % (k,v[0]/v[1]))

#%%    
import pandas as pd
grades = pd.read_csv('D:\student_grades_header.txt')
print(grades)
grades.groupby('major').mean()

#%%
import pandas as pd
import scipy  
import scikits.bootstrap as bootstrap 

grades = pd.read_csv('D:\On_Time_On_Time_Performance_1998_1.csv')
#print(grades)
grades.groupby(['Carrier','Year','Month'])['DepDelay'].mean()
#grades.groupby(['Carrier','Year','Month'])['DepDelay'].size().unstack()
#grades.groupby(['Year'])['DepDelay'].mean()
#grades.groupby(['Carrier','Year','Month'])['DepDelay'].std()
#grades.groupby('DayofMonth').mean()

#%%
    
f =  open('D:\On_Time_On_Time_Performance_1998_1.csv')
grade_by_major = {}

n = 0
j = 0
sum = 0.0

for line in f:
    
    if n == 0:
        line_items = line.split(',')
        n=n+1    
    elif n > 0:
        line_items = line.split(',')
        major = line_items[8]

        if line_items[33] == '': 
            grade = 0
        else:
            grade = float(line_items[33])
        
        grade_by_major.setdefault(major,[0,0,0])
        
        grade_by_major[major][0] = grade_by_major[major][0] + grade
        grade_by_major[major][1] = grade_by_major[major][1] + 1

        n=n+1


print (grade_by_major.items())    
for k, v in grade_by_major.items():
    print('%s\t%0.2f' % (k,v[0]/v[1]))

#%%
from scipy.stats import t
from numpy import average, std
from math import sqrt

if __name__ == '__main__':
    # data we want to evaluate: average height of 30 one year old male and
    # female toddlers. Interestingly, at this age height is not bimodal yet
    data = [63.5, 81.3, 88.9, 63.5, 76.2, 67.3, 66.0, 64.8, 74.9, 81.3, 76.2,
            72.4, 76.2, 81.3, 71.1, 80.0, 73.7, 74.9, 76.2, 86.4, 73.7, 81.3,
            68.6, 71.1, 83.8, 71.1, 68.6, 81.3, 73.7, 74.9]
    mean = average(data)
    # evaluate sample variance by setting delta degrees of freedom (ddof) to
    # 1. The degree used in calculations is N - ddof
    stddev = std(data, ddof=1)
    # Get the endpoints of the range that contains 95% of the distribution
    t_bounds = t.interval(0.95, len(data) - 1)
    # sum mean to the confidence interval
    ci = [mean + critval * stddev / sqrt(len(data)) for critval in t_bounds]
    print ("Mean: %f" % mean)
    print ("Confidence Interval 95%%: %f, %f" % (ci[0], ci[1]))   
    
    
#%%    
##%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
        'pre_score': [4, 24, 31, 2, 3],
        'mid_score': [25, 94, 57, 62, 70],
        'post_score': [5, 43, 23, 23, 51]}
df = pd.DataFrame(raw_data, columns = ['first_name', 'pre_score', 'mid_score', 'post_score'])
df

# Setting the positions and width for the bars
pos = list(range(len(df['pre_score']))) 
width = 0.25 

# Plotting the bars
fig, ax = plt.subplots(figsize=(10,5))

# Create a bar with pre_score data,
# in position pos,
plt.bar(pos, 
        #using df['pre_score'] data,
        df['pre_score'], 
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#EE3224', 
        # with label the first value in first_name
        label=df['first_name'][0]) 

# Create a bar with mid_score data,
# in position pos + some width buffer,
plt.bar([p + width for p in pos], 
        #using df['mid_score'] data,
        df['mid_score'],
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#F78F1E', 
        # with label the second value in first_name
        label=df['first_name'][1]) 

# Create a bar with post_score data,
# in position pos + some width buffer,
plt.bar([p + width*2 for p in pos], 
        #using df['post_score'] data,
        df['post_score'], 
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#FFC222', 
        # with label the third value in first_name
        label=df['first_name'][2]) 

# Set the y axis label
ax.set_ylabel('Score')

# Set the chart's title
ax.set_title('Test Subject Scores')

# Set the position of the x ticks
ax.set_xticks([p + 1.5 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(df['first_name'])

# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*4)
plt.ylim([0, max(df['pre_score'] + df['mid_score'] + df['post_score'])] )

# Adding the legend and showing the plot
plt.legend(['Pre Score', 'Mid Score', 'Post Score'], loc='upper left')
plt.grid()
plt.show()



#%%
    import pandas as pd

grades = pd.read_csv('D:\driving.txt')
print(grades)
#grades.groupby(['Test number'])['SALevel 3'].mean()
#grades.groupby(['Carrier','Year','Month'])['DepDelay'].size().unstack()
#grades.groupby(['Year'])['DepDelay'].mean()
#grades.groupby(['Carrier','Year','Month'])['DepDelay'].std()
#grades.groupby('DayofMonth').mean()

    
#%%
    import pandas as pd
grades = pd.read_csv('D:\student_grades_header.txt')
print(grades)
grades.groupby('major').mean()
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy  
import scikits.bootstrap as bootstrap 

grades = pd.read_csv('D:\driving.csv')
#print(grades)
print(grades.groupby(['Group type', 'Time', 'Test Type'])['SA Level 3'].mean())
#grades.groupby(['Carrier','Year','Month'])['DepDelay'].size().unstack()
#grades.groupby(['Year'])['DepDelay'].mean()
#grades.groupby(['Carrier','Year','Month'])['DepDelay'].std()
#grades.groupby('DayofMonth').mean()

#===========================================================================================
ag = grades.groupby('Time').mean()

#ag.unstack().plot(kind='bar', subplots=True, layout=(2,2))

#==========================================================================================
my_plot = grades.groupby(['Group type', 'Time', 'Test Type'])['SA Level 3'].mean().plot(kind='bar')

#grades.head()
#grades.describe()


#%%
# print bar chart by python 
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
y_pos = np.arange(len(objects))
performance = [10,8,6,4,2,1]
 
plt.barh(y_pos, performance, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.ylabel('test')
plt.xlabel('Usage')
plt.title('Programming language usage')
plt.show()
#%%

import numpy as np
import matplotlib.pyplot as plt

N = 4
men_means = (20, 35, 30, 35)
men_std = (2, 3, 4, 1)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots(1,1)
rects1 = ax.bar(ind, men_means, width, color='r', yerr=men_std)

women_means = (25, 32, 34, 20)
women_std = (3, 5, 2, 3)
rects2 = ax.bar(ind + width, women_means, width, color='y', yerr=women_std)

# add some text for labels, title and axes ticks
ax.set_ylabel('Scores')
ax.set_xlabel('Experimental                   Placeb')



ax.set_title('Scores by group and gender')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Simulator', 'On-Road', 'Simulator', 'On-road'))

ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

n= 6

m1 = (0.10,0.12,0.10,0.11,0.14,0.10)
m2=(0.21,0.21,0.20,0.22,0.20,0.21)
m3=(0.29,0.27,0.28,0.24,0.23,0.23)
m4=(0.41,0.39,0.35,0.37,0.41,0.40)
x=[1,2,3,4,5,6]

fig, ax = plt.subplots()

index = np.arange(n)
bar_width = 0.2

opacity = 0.4
error_config = {'ecolor': '0.3'}
r1 = ax.bar(index, m1, bar_width,
                 alpha=opacity,
                 color='b',

                 error_kw=error_config)

r2 = ax.bar(index + bar_width, m2, bar_width,
                 alpha=opacity,
                 color='r',

                 error_kw=error_config)

r3 = ax.bar(index + bar_width+ bar_width, m3, bar_width,
                 alpha=opacity,
                 color='y',
                 error_kw=error_config)
r4 = ax.bar(index + bar_width+ bar_width+ bar_width, m4, bar_width,
                 alpha=opacity,
                 color='c',
                 error_kw=error_config)                 
plt.xlabel('D')
plt.ylabel('Anz')
plt.title('Th')

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

ax1.bar(x,m1, 0.2) #% thickness=0.2
ax2.bar(x,m2, 0.2)
ax3.bar(x,m3, 0.2)
ax4.bar(x,m4, 0.2)

plt.tight_layout()
plt.show()

#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.rand(6, 4),
                 index=['one', 'two', 'three', 'four', 'five', 'six'],
                 columns=pd.Index(['A', 'B', 'C', 'D'], 
                 name='Genus')).round(2)


df.plot(kind='bar',figsize=(10,4))

ax = plt.gca()
pos = []
for bar in ax.patches:
    pos.append(bar.get_x()+bar.get_width()/2.)


ax.set_xticks(pos,minor=True)
lab = []
for i in range(len(pos)):
    l = df.columns.values[i//len(df.index.values)]
    lab.append(l)

ax.set_xticklabels(lab,minor=True)
ax.tick_params(axis='x', which='major', pad=15, size=0)
plt.setp(ax.get_xticklabels(), rotation=0)

plt.show()


#%%

import matplotlib.pyplot as plt
import plotly.plotly as py
# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

fig = plt.figure()

ax1 = fig.add_subplot(221)
ax1.plot([1,2,3,4,5], [10,5,10,5,10], 'r-')

ax2 = fig.add_subplot(222)
ax2.plot([1,2,3,4], [1,4,9,16], 'k-')

ax3 = fig.add_subplot(223)
ax3.plot([1,2,3,4], [1,10,100,1000], 'b-')

ax4 = fig.add_subplot(224)
ax4.plot

#%%
import matplotlib.pyplot as plt

x = range(10)
y = range(10)

fig, ax = plt.subplots(nrows=2, ncols=2)

for row in ax:
    for col in row:
        col.plot(x, y)

plt.show()

#%%

fig, ax = plt.subplots(nrows=2, ncols=2)

plt.subplot(2, 2, 1)
plt.plot(x, y)

plt.subplot(2, 2, 2)
plt.plot(x, y)

plt.subplot(2, 2, 3)
plt.plot(x, y)

plt.subplot(2, 2, 4)
plt.plot(x, y)

plt.show()

#%%
import matplotlib.pyplot as plt

fig = plt.figure()
axes = fig.subplots(nrows=2, ncols=2)

plt.show()
#%%
import matplotlib.pyplot as plt
import numpy as np
x = np.arange (0,10.0, 0.1)
fig, ax = plt.subplots (2,2)
ax[0,0].plot(x, x)
ax[0,1].plot(x, x**2)
ax[1,0].plot(x, np.sqrt(x))
ax[1,1].plot(x, 1/x)
plt.show ()

#%%
# libraries and data
# libraries
import numpy as np
import matplotlib.pyplot as plt
 
# set width of bar
barWidth = 0.25
 
# set height of bar
bars1 = [12, 30, 1, 8, 22]
bars2 = [28, 6, 16, 5, 10]
bars3 = [29, 3, 24, 25, 17]
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
# Make the plot
plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='var1')
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='var2')
plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='var3')
 
# Add xticks on the middle of the group bars
plt.xlabel('group', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['A', 'B', 'C', 'D', 'E'])
 
# Create legend & Show graphic
plt.legend()
plt.show()

#%%
