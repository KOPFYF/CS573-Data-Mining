import math
x = 0.0000001
y = 0.0000001
H = -(3+x+y)/7*math.log2((3+x+y)/7)-(4-x-y)/7*math.log2((4-x-y)/7)

H10 = 1
H11 = -y/2*math.log2(y/2)-(2-y)/2*math.log2((2-y)/2)
H12 = -(x+2)/3*math.log2((x+2)/3)-(1-x)/3*math.log2((1-x)/3)

H20 = -(y+1)/2*math.log2((y+1)/2)-(1-y)/2*math.log2((1-y)/2)
H21 = -x/3*math.log2(x/3)-(3-x)/3*math.log2((3-x)/3)
H22 = 0

IG1 = H - 2/7*H10 - 2/7*H11 - 3/7*H12

IG2 = H - 2/7*H20 - 3/7*H21 - 2/7*H22

print(IG1,IG2)

a2 = 3/10
b2 = 1/4
A = a2/b2*(1-b2)/(1-a2)
print(9/7)