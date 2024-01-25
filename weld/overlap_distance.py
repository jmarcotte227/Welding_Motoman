import sympy as sym
from sympy.plotting import plot

from sympy.plotting.plot import MatplotlibBackend, Plot


def get_sympy_subplots(plot:Plot):
    backend = MatplotlibBackend(plot)

    backend.process_series()
    backend.fig.tight_layout()
    return backend.plt

def overlap_distance(h_1, w_1, h_2, w_2):
    x = sym.Symbol('x')
    d = sym.Symbol('d')
    x2 = sym.Symbol('x2')

    a_1 = -4*h_1/(w_1**2)
    c_1 = h_1
    bead_a = a_1*x**2+c_1

    a_2 = -4*h_2/(w_2**2)
    c_2 = h_2
    bead_b = a_2*(x-d)**2+c_2

    # find intersection
    d_prime = sym.solveset(bead_a-bead_b, x).args[0]

    #find x1 val
    x1 = sym.solveset(bead_b, x).args[0]
    D = max(sym.solveset(bead_a, x).args)
    
    #find y vals
    y1 = bead_a.subs(x, x1)
    y2 = bead_b.subs(x, x2)

    # find tangent point
    bead_b_der = sym.diff(bead_b,x).subs(x,x2)
    k = (y2-y1)/(x2-x1)
    x2_gen = sym.solveset(bead_b_der-k, x2).args[0].args[1] #magic numbers to select correct solution
    y2 = y2.subs(x2, x2_gen)

    S_AED = sym.integrate(bead_b,(x,x1, d_prime))+sym.integrate(bead_a, (x,d_prime, D))
    S_ABCF = (y1+y2)/2*(x2-x1)
    S_ABD = sym.integrate(bead_a, (x,x1,D))
    S_ACF = sym.integrate(bead_b, (x, x1,x2_gen))

    S_BEC = S_ABCF+S_AED-S_ABD-S_ACF
    
    S_BEC = S_BEC.subs(x2,x2_gen)
    
    # finding zeros
    f = S_BEC-S_AED
    equality_points = sym.solveset(f,d).args[0].args[0]
    return equality_points, bead_a, bead_b, x1, x2_gen, y1, y2, d

if __name__=="__main__":
    h1 = 1.48609461041326
    w1 = 4.77244089559745
    h2 = 1.53421587430785
    w2 = 4.95234504814499

    dist, a, b, x_1, x_2, y_1, y_2, d = overlap_distance(h1, w1, h2, w2)

    b = b.subs(d, dist)
    x_1 = x_1.subs(d, dist)
    x_2 = x_2.subs(d, dist)
    y_1 = y_1.subs(d, dist)
    y_2 = y_2.subs(d, dist)

    slope = (y_2-y_1)/(x_2-x_1)

    x = sym.symbols('x')
    p1 = plot(a, show=False)
    p2 = plot(b, show=False)
    p1.append(p2[0])

    plt = get_sympy_subplots(p1)
    plt.plot([x_1, x_2], [y_1, y_2])
    plt.xlim([-4, 7])
    plt.ylim([0, 2])
    plt.axis('off')

    plt.show()


    # print(a)
    # print(b)
    # print(x_1)
    # print(x_2)
    # print(y_1)
    # print(y_2)
