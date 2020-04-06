import math
import numpy as npy
from collections import Counter
"""Minimos Cuadros"""
def min_square(array, val_cap):
    n = (len(array)-1)
    ny = []
    result = []
    for val in range(n,val_cap-1):
        for i in range(len(array)+1):
            ny.append(i)
        res = vars_init(ny,array)
        array.append(res)
        result = array
        ny = []
    return result
def vars_init(x, y):
    xs = 0
    ys = 0
    x_for_y = []
    sum_x_for_y = 0
    squares_x = []
    sum_squares_x = 0 
    for j,i in zip(x,y):
        j=j+1
        xs = xs + j
        ys = ys + i 
        x_for_y.append(j*i)
        squares_x.append(math.pow(j,2))
    for i in x_for_y:
        sum_x_for_y = sum_x_for_y+i
    for i in squares_x:
        sum_squares_x = sum_squares_x+i
    b = formula_b((len(x)-1),sum_x_for_y,xs,ys,sum_squares_x)
    a = formula_a(b,ys,xs,(len(x)-1))
    y_n = fomula_y(a,b,len(x))
    return y_n

def formula_b(n, sum_x_for_y, sum_x, sum_y, sum_sqrts_x):
    return ((n*sum_x_for_y)-(sum_x*sum_y))/((n*sum_sqrts_x)-math.pow(sum_x,2))

def formula_a(b, sum_y, sum_x, n):
    return (sum_y-(b*sum_x))/n

def fomula_y(a, b, sgt_n):
    return (a+(b*sgt_n))

"""Mean"""
def mean(narray):
    return sum(narray)/(len(narray))

"""Median"""
def median(narray):
    n = len(narray)
    if (n % 2) == 0:
        index = int(n/2)
        result = (narray[index-1] + narray[index])/2
    else:
        index = int((n-1)/2)
        result =narray[index]
    return result

"""Typical deviation"""
def typical_deviation(mean,narray):
    pow = 0
    for i in narray:
        pow = pow + i**2
    return ((pow/len(narray))-(mean**2))

"""Typical deviation table"""
def typical_deviation_table(mularray):
    mularray[0] = list(map(float,mularray[0]))
    mularray[1] = list(map(float,mularray[1]))
    val = 0
    size = len(mularray[0])
    sum_ni = sum(mularray[1])
    mularray[1].append(sum_ni)
    n_i = []
    ni_x_fi =[]
    qrtx_x_fi = []
    for i in mularray[1]:
        val = val + i
        if (mularray[1][len(mularray[1])-1]) == i:
            break
        n_i.append(val)
    mularray.append(n_i)
    for i in range(size):
        ni_x_fi.append((mularray[0][i]*mularray[1][i]))
        if (size-1) == i:
            sum_ni_fi =sum(ni_x_fi)
            ni_x_fi.append(sum_ni_fi)
    mularray.append(ni_x_fi)
    for i in range(size):
        qrtx_x_fi.append(((mularray[0][i]**2)*mularray[1][i]))
        if (size-1) == i:
            sum_pow_fi = sum(qrtx_x_fi)
            qrtx_x_fi.append(sum_pow_fi)
    mularray.append(qrtx_x_fi)
    mean = sum_ni_fi/sum_ni
    typical_deviation = (sum_pow_fi/sum_ni)-(mean**2)
    return mularray,(mean),float("{0:.2f}".format(typical_deviation))





"""Northwest corner""" 
def northwest_corner(cost,supply,demand):
    print(cost)
    print(supply)
    print(demand)
    assert sum(supply) == sum(demand)
    total = []
    for i in len(cost):
        for j in len(cost[i]):
            print("ss")
                                
    return total
"""typical_deviation_table_interval"""

def typical_deviation_table_interval(intervals,f_i):
    x_i = [];
    for i in intervals:
        x_i.append(sum(i)/2)
    return typical_deviation_table([x_i,f_i])


def northwest_corner(cost,supply,demand):
    c,r,total_cost=0,0,0
    column = (len(cost)-1)
    row = (len(cost[0])-1)
    value = npy.zeros_like(cost)
    while True:
        if demand[c]!=0:
            if demand[c]<supply[r]:
                value[r][c]=demand[c]
                supply[r]-=demand[c]
                demand[c]=0
                if c==column:
                    c = 0
                else:
                    c += 1
            else:
                if(demand[c]==supply[r]):
                    value[r][c]=supply[r]
                    demand[c]=0
                    supply[r]=0
                    if(c==column):
                        c=0
                    else:
                        c += 1
                    if r==row:
                        r=0
                    else:
                        r += 1
                else:
                    value[r][c]=supply[r]
                    demand[c]-=supply[r]
                    if r==row:
                        r=0
                    else:
                        r += 1
        else:
            if c==column:
                c=0
            else:
                c += 1
        if demand[column]==0:
            break
    for i in range(row+1):
        for j in range(column+1):
            total_cost += cost[i][j]*value[i][j]
    return total_cost

def method_vogle(cost,supply,demand):
    value = npy.zeros_like(demand)
    col_penalty = npy.zeros_like(supply)
    row_penalty = npy.zeros_like(demand)
    val = npy.min(supply,axis=0)
"""def northwest_corner(cost,supply,demand):

    #Compliminto obligatorio paa seguir  
    assert sum(demand) == sum(supply)

    C = npy.copy(cost)
    S = npy.copy(supply)
    D = npy.copy(demand)

    n,m = C.shape

    has_degenerated_init_solution = False
    has_unique_solution = True
    has_degenerated_mid_solution = False
    
    X_start = npy.full((n,m),npy.nan)
    fill_X = npy.ones((n,m),dtype=bool)
    indexs = [(i, j) for i in range(n) for j in range(m)]

    #LLenado de los indicies 
    def fill_indexs(i,j):
        fill_X[i,j] = False
        indexs_i = [
            (i,jj) for jj in range(m) if fill_X[i, jj]
        ]
        indexs_j = [
            (ii, j) for ii in range(n) if fill_X[ii, j]
        ]
        allowed_indexs = indexs_i + indexs_j
        if allowed_indexs:
            return allowed_indexs[0]
        else:
            return None

        xs = sorted(zip(indexs,C.flatten()),key=lambda a, b: (a[0],a[1]))

        for (i, j), _ in xs:
            contained = min([S[i],D[j]])

            if contained == 0:
                continue
            elif not npy.isnan(X_start[i, j]):
                continue
            else:
                X_start[i, j] = contained
                if S[i] == contained and D[j] == contained:
                    fill_zeros_indexs = fill_indexs(i,j)
                    if fill_zeros_indexs:
                        X_start[fill_zeros_indexs] = 0
                        fill_X[fill_zeros_indexs] = False
                        has_degenerated_init_solution = True
                S[i] -= contained
                D[j] -= contained
            if D[j] == 0:
                fill_X[:,j] = False
            if S[i] == 0:
                fill_X[i,:] = False 

    while True:
        U = npy.array([npy.nan]*n)
        V = npy.array([npy.nan]*m)
        SS = npy.full((n, m), npy.nan)

        _x , _y = npy.where(npy.isnan(X_start))
        basis = list(zip(_x,_y))
        f = basis[0][0]
        U[f] = 0

        while any(npy.isnan(U)) or any(npy.isnan(V)):
            for i, j in basis:
                if npy.isnan(U[i]) and not npy.isnan(V[j]):
                    U[i] = C[i, j] - V[j]
                elif not npy.isnan(U[i]) and npy.isnan(V[j]):
                    V[j] = C[i, j] - U[i]
                else:
                    continue
        for i in range(n):
            for j in  range(m):
                if npy.isnan(X_start[i, j]):
                    SS[i, j] = C[i, j] - U[i] - V[j]

        S = npy.nanmin(SS) 
        print(SS)
        if S > 0:
            break
        elif S == 0:
            has_unique_solution = False
            break

        i, j = npy.argwhere(SS == S)[0]
        start = (i, j)

        T = npy.zeros((n, m))

        for i in range(0, n):
            for j in range(0, m):
                if not npy.isnan(X_start[i, j]):
                    T[i, j] = 1
        T[start] = 1
        while True:
            _xs, _ys = npy.nonzero(T)
            xcount, ycount = Counter(_xs),Counter(_ys)
            for x, count in xcount.items():
                if count <= 1:
                    T[x, :] = 0
            for y, count in ycount.items():
                if count <= 1:
                    T[:, y] = 0
            if all(x > 1 for x in xcount.values()) and all(y > 1 for y in ycount.values()):
                break

        diff = lambda x1,y1,x2,y2: (abs(x1-x2) + abs(y1-y2)) if((x1==x2 or y1==y2) and not (x1==x2 and y1==y2)) else npy.inf
        stripe = set(tuple(p) for p in npy.argwhere(T > 0))
        size = len(stripe)
        path = [start]
        while len(path) < size:
            last = path[-1]
            if last in stripe:
                stripe.remove(last)
                #print(last)
            next1 = min(stripe,lambda x, y: diff(last), (x, y))
            path.append(next1)
        neg = path[:]
        pos = path[::2]
        print(*neg)
        #print(*pos)
        q = min(X_start[list(zip(*neg))])
        if q == 0:
            has_degenerated_mid_solution = True
        X_start[start] = 0
        X_start[list(zip(*neg))] -= q
        X_start[list(zip(*pos))] += q

        for ne in neg:
            if X_start[ne] == 0:
                X_start[ne] = npy.nan
                break
    
    X_final = npy.copy(X_start)
    for i in range(0, n):
        for j in range(0, m):
            if npy.isnan(X_final[i, j]):
                X_final[i, j] = 0

    return X,npy.sum(X_start*C)
            
        
    #print(test[(len(test)-1).__str__()]['0'])
    #print(test['0'][(len(test['0'])-1).__str__()])
    #for i in range(5):
    #   for j in range(5):
    #      print(test[i.__str__()+""+j.__str__()])

    
"""