import math
import numpy as np

m1=0
m2=0.5
m3=1

m4=0
m5=0.5
m6=1

de1=.21
de2=1.21
de3=.21

de4=.21
de5=.21
de6=.21

p1=0
p2=10000
p3=0
p4=0
p5=-1000
p6=0
p7=0
p8=-1000
p9=0

q1=0
q2=10000
q3=0
q4=0
q5=0
q6=0
q7=0
q8=-1000
q9=0

r1=0
r2=0
r3=0
r4=0
r5=0
r6=0
r7=0
r8=0
r9=1000

size = 10

x = np.empty((size))
y = np.empty((size))

mf1 = np.empty((size))
mf2 = np.empty((size))
mf3 = np.empty((size))
mf4 = np.empty((size))
mf5 = np.empty((size))
mf6 = np.empty((size))

z = np.empty((size,size))

for i in range(0,size):
    for j in range(0,size):

        x[i]=(i+1)/10
        y[j]=(j+1)/10

        mf1[i]=math.exp((-(x[i]-m1)**2)/(2*de1**2))
        mf2[i]=math.exp((-(x[i]-m2)**2)/(2*de2**2))
        mf3[i]=math.exp((-(x[i]-m3)**2)/(2*de3**2))
    
        mf4[j]=math.exp((-(y[j]-m4)**2)/(2*de4**2))
        mf5[j]=math.exp((-(y[j]-m5)**2)/(2*de5**2))
        mf6[j]=math.exp((-(y[j]-m6)**2)/(2*de6**2))


        print(mf1[i], mf2[i], mf3[i], mf4[j], mf5[j], mf6[j])


        inf1=mf1[i]*mf4[j]
        inf2=mf1[i]*mf5[j]
        inf3=mf1[i]*mf6[j]
        inf4=mf2[i]*mf4[j]
        inf5=mf2[i]*mf5[j]
        inf6=mf2[i]*mf6[j]
        inf7=mf3[i]*mf4[j]
        inf8=mf3[i]*mf5[j]
        inf9=mf3[i]*mf6[j]

        reg1=inf1*((p1*x[i])+(q1*y[j])+r1)
        reg2=inf2*((p2*x[i])+(q2*y[j])+r2)
        reg3=inf3*((p3*x[i])+(q3*y[j])+r3)
        reg4=inf4*((p4*x[i])+(q4*y[j])+r4)
        reg5=inf5*((p5*x[i])+(q5*y[j])+r5)
        reg6=inf6*((p6*x[i])+(q6*y[j])+r6)
        reg7=inf7*((p7*x[i])+(q7*y[j])+r7)
        reg8=inf8*((p8*x[i])+(q8*y[j])+r8)
        reg9=inf9*((p9*x[i])+(q9*y[j])+r9)

        b=inf1+inf2+inf3+inf4+inf5+inf6+inf7+inf8+inf9
        a=reg1+reg2+reg3+reg4+reg5+reg6+reg7+reg8+reg9

        z[i][j]=a/b

print(z)


# figure(1)
# surf(x,y,z)

# figure(2)
# plot(x,mf1,x,mf2,x,mf3)

# figure(3)
# plot(y,mf4,y,mf5,y,mf6)
