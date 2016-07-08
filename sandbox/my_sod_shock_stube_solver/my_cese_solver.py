"""
Sod Shock Tube Problem -- CESE solver
Editor: You-Hao Chang

Journal of Computational Physics 119, 295-324 (1995)
The Method of Space-Time Conservation Element and Solution
Element -- A New Approach for Solving the Navier-Stokes and 
Eluer Equations

Sod tube condition: 
    x range: 101 points btw -0.505 ~ 0.495
    two region: left region, -0.505 ~ -0.005 (51 points)
                right region, 0.005 ~  0.495 (50 points)
"""

import matplotlib
matplotlib.use('TKAgg')

import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# initialization
it = 100            # number of iterations
nx = it + 1         # number of points (x-axis)
dt = 0.4 * 10**(-2)
dx = 0.1 * 10**(-1)
ga = 1.4            #gamma = Cp/Cv

rhol = 1.0
pl   = 1.0
ul   = 0.0
rhor = 0.125
pr   = 0.1
ur   = 0.0

mtx_q  = [[0. for dummy_col in range(nx)] for dummy_row in range(3)]
mtx_qn = [[0. for dummy_col in range(nx)] for dummy_row in range(3)]
mtx_qx = [[0. for dummy_col in range(nx)] for dummy_row in range(3)]
mtx_qt = [[0. for dummy_col in range(nx)] for dummy_row in range(3)]
mtx_s  = [[0. for dummy_col in range(nx)] for dummy_row in range(3)]
vxl = [0. for idx in range(3)]
vxr = [0. for idx in range(3)]

# output lists
xx  = [0. for idx in range(nx)] # x-axis
status_rho  = [rhor if idx > nx / 2 else rhol for idx in range(nx)] # rho
status_vel  = [0. for idx in range(nx)] # v
status_p  = [pr if idx > nx / 2 else pl for idx in range(nx)] # p

ia = 1

hdt = dt / 2.0
tt   = hdt / float(it)
qdt = dt / 4.0

hdx = dx / 2.0
qdx = dx / 4.0

dtx = dt / dx

a1 = ga - 1.0
a2 = 3.0 - ga
a3 = a2 / 2.0
a4 = 1.5 * a1

itp = it + 1

for j in xrange(0, nx):
    # Eq (4.1), u1, u2 and u3
    if j <= 50:
        mtx_q[0][j] = rhol
        mtx_q[1][j] = rhol * ul
        mtx_q[2][j] = pl / a1 + 0.5 * rhol * ul**2
    else:
        mtx_q[0][j] = rhor
        mtx_q[1][j] = rhor * ur
        mtx_q[2][j] = pr / a1 + 0.5 * rhor * ur**2

# setting the x-axis
t2 = dx * float(itp)
xx[0] = -0.5 * t2
for j in xrange(0, nx - 1):
    xx[j+1] = xx[j] + dx

fig = plt.figure()
frame_seq = []
m = nx

# start to evaluate the solution iteratively
for i in xrange(0, it):

    # evaluate the current status of gas
    for j in xrange(0, m):
        w2 = mtx_q[1][j] / mtx_q[0][j]  # u2/u1
        w3 = mtx_q[2][j] / mtx_q[0][j]  # u3/u1

        # Reference: Yung-Yu's notes -> Sec.3.1, Eq. (3.14)
        f21 = -a3 * w2**2               # -0.5*(3.0-gamma)*(u2/u1)^2
        f22 = a2 * w2                   # (3.0-gamma)*(u2/u1)
        f31 = a1 * w2**3 - ga * w2 * w3 # (gamma-1.0)*(u2/u1)^2-gamma*u2*u3/(u1)^2
        f32 = ga * w3 - a4 * w2**2      # gamma*u3/u1 - 1.5*(gamma-1.0)*(u2/u1)^2
        f33 = ga * w2                   # gamma*u2/u1

        # Eq.(4.17), (u_mt)nj = -(f_mx)nj = -(fm,k*u_kx)nj, (u_mt)nj -> qt, (u_kx)nj -> qx
        mtx_qt[0][j] = -mtx_qx[1][j]
        mtx_qt[1][j] = -(f21 * mtx_qx[0][j] + f22 * mtx_qx[1][j] + a1  * mtx_qx[2][j])
        mtx_qt[2][j] = -(f31 * mtx_qx[0][j] + f32 * mtx_qx[1][j] + f33 * mtx_qx[2][j])

        # Eq.(4.25), (u_m)nj -> q, (u_mt)nj -> qt, (u_kx)nj -> qx
        # for s0: u_0x -> mtx_qx[0][j], f_0  -> mtx_q[1][j], f_0t -> f0,k*u_kt 
        #     s1: u_1x -> mtx_qx[1][j], f_1  -> a1*mtx_q[2][j]+f21*mtx_q[0][j]+f22*mtx_q[1][j]
        #                           f_1t -> f21*mtx_qx[0][j]+f22*mtx_qx[1][j]+a1*mtx_qx[2][j]
        #     s2: u_2x -> mtx_qx[2][j], f_2  -> f31*mtx_q[0][j]+f32*mtx_q[1][j]+f33*mtx_q[2][j]
        #                           f_2t -> f31*mtx_qx[0][j]+f32*mtx_qx[1][j]+f33*mtx_qx[2][j]
        mtx_s[0][j] = qdx * mtx_qx[0][j] + dtx * (mtx_q[1][j] + qdt * mtx_qt[1][j])
        
        mtx_s[1][j] = qdx * mtx_qx[1][j] + dtx * (f21 * (mtx_q[0][j] + qdt * mtx_qt[0][j]) + \
                                          f22 * (mtx_q[1][j] + qdt * mtx_qt[1][j]) + \
                                          a1  * (mtx_q[2][j] + qdt * mtx_qt[2][j])) 
        mtx_s[2][j] = qdx * mtx_qx[2][j] + dtx * (f31 * (mtx_q[0][j] + qdt * mtx_qt[0][j]) + \
                                          f32 * (mtx_q[1][j] + qdt * mtx_qt[1][j]) + \
                                          f33 * (mtx_q[2][j] + qdt * mtx_qt[2][j]))

    # evaluate the status of gas after time stamp moves forward by 1 hdt
    mm = m - 1
    for j in xrange(0, mm): # j -> 1 hdx
        for k in xrange(0, 3):
            # Eq.(4.24), index of 'qn' will be the same for the following calculation
            mtx_qn[k][j+1] = 0.5 * (mtx_q[k][j] + mtx_q[k][j+1] + mtx_s[k][j] - mtx_s[k][j+1])
            # Eq.(4.27) and Eq.(4.36)  
            vxl[k] = (mtx_qn[k][j+1] - mtx_q[k][j] - hdt * mtx_qt[k][j]) / hdx     # l -> -
            vxr[k] = (mtx_q[k][j+1] + hdt * mtx_qt[k][j+1] - mtx_qn[k][j+1]) / hdx # r -> +
            # Eq.(4.38) and Eq.(4.39)
            mtx_qx[k][j+1] = (vxl[k] * (math.fabs(vxr[k]))**ia + vxr[k] * (math.fabs(vxl[k]))**ia)\
                             / ((math.fabs(vxl[k]))**ia + (math.fabs(vxr[k]))**ia + 10**(-60))

    for j in xrange(1, m):
        for k in xrange(0, 3):
            mtx_q[k][j] = mtx_qn[k][j]

    if i % 2 != 0:
        for j in xrange(1, m):
            for k in xrange(0, 3):
                mtx_q[k][j-1] = mtx_q[k][j]
                mtx_qx[k][j-1] = mtx_qx[k][j]

        for j in xrange(0, itp):
            status_rho[j] = mtx_q[0][j]  # rho
            status_vel[j] = mtx_q[1][j] / mtx_q[0][j]  # v
            status_p[j] = a1 * (mtx_q[2][j] - 0.5 * mtx_q[0][j] * status_vel[j]**2) # p
    

        plt.subplot(311)
        plot_rho = plt.scatter(xx, status_rho, color="r")
        plt.xlabel("x")
        plt.ylabel("density")
        plt.xlim(-0.55, 0.55)
        plt.ylim(-0.1, 1.1)
        plt.subplot(312)
        plot_vel = plt.scatter(xx, status_vel, color="g")
        plt.xlim(-0.55, 0.55)
        plt.ylim(-0.1, 1.1)
        plt.xlabel("x")
        plt.ylabel("velocity")
        plt.subplot(313)
        plot_p = plt.scatter(xx, status_p, color="b")
        plt.xlim(-0.55, 0.55)
        plt.ylim(-0.1, 1.1)
        plt.xlabel("x")
        plt.ylabel("pressure")
        frame_seq.append((plot_rho, plot_vel, plot_p))

    file = open("%03d" % (i + 1) + ".dat", 'w')
    for j in xrange(0, m):
        file.write(str(xx[j]) + " " + str(status_rho[j]) + " " + str(status_vel[j]) + " " + str(status_p[j]) + "\n")
        #print '{0:7f} {1:7f} {2:7f} {3:7f}'.format(xx[j], status_rho[j], status_vel[j], status_p[j])

    file.close()

# animation
ani = animation.ArtistAnimation(fig, frame_seq, interval=25, repeat_delay=300, blit=True)
ani.save('mySodTube.mp4', fps=10);

plt.show()

