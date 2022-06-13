import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.linalg import eigh_tridiagonal

# Input parameter
xmin = 0
xmax = 1
dx = 0.0033
dt = 1e-7
tmax = 0.01

Nx = int((xmax - xmin) / dx)  # Spatial grid size
Nt = int((tmax - 0) / dt)  # Time grid size

x = np.linspace(xmin, xmax, Nx)  # Spatial grid
t = np.linspace(0, tmax, Nt)  # Time grid

# Psi Initial & Potential
# init=1
# psi01 = -(np.sin(x)+np.cos(x))
# V = -1/np.sqrt(2+x**2)

# Psi initial & Potential (Another example)
init = 2
psi01 = np.sqrt(2) * np.sin(np.pi * x)
V = -1e4 * np.exp(-(x - 1 / 2) ** 2 / (2 * (1 / 20) ** 2))

T = 100 / tmax  # Periode
W = (2 * np.pi) / T  # Omega
Et = (np.sin((np.pi * (t / tmax)) ** 2)) * (np.cos(W * t))  # Electric Field

# Normalization
fnorm1 = np.sqrt(np.sum(np.absolute((psi01) ** 2))) * dx
# Psi_normalization
psi01 = psi01 / fnorm1
psi02 = psi01  # for diferennt method

psi = np.zeros([Nt, Nx])  # Matrik psi
psi[0] = psi01  # Input psi01 in matrix
alpha = dt / dx ** 2  # dt/dx**2


# Method 1 Finite Difference (Gaussian)
# Compute matrix psi
def compute_psi(psi):
    for t in range(0, Nt - 1):
        for i in range(1, Nx - 1):
            # H_interaction=E[t].i
            psi[t + 1][i] = psi[t][i] + 1j / 2 * alpha * (psi[t][i + 1] - 2 * psi[t][i] + psi[t][i - 1]) - 1j * dt * \
                            psi[t][i] * (V[i] + Et[t] * i)
        # Normalization
        normal = np.sqrt(np.sum(np.absolute((psi[t + 1]) ** 2))) * dx
        for i in range(1, Nx - 1):
            psi[t + 1][i] = psi[t + 1][i] / normal
    return psi


psi_m1 = compute_psi(psi.astype(complex))

# Methode Eigenstate Evolution (Filter)
# define tridiagonal matrix
d = (1 * dt / 2) + V[1:-1] + Et[1] * x[1:-1]  # Diagonal
e = -1 / (2 * dx ** 2) * np.ones(len(d) - 1)
# Compute eigenstates and eigen-energies using spicy library
w, v = eigh_tridiagonal(d, e)  # w= eigen value , v= eiegen function
# Compute matrix psi
E_js = w[0:70]
psi_js = np.pad(v.T[0:70], [(0, 0), (1, 1)], mode='constant')
cs = np.dot(psi_js, psi02)


def psi_m2(t):
    return psi_js.T @ (cs * np.exp(-1j * E_js * t))


# Plot Grafik
rumus = r'$ \psi_0 = -(sin(x) +cos(x))$' + '   ' + r'$V= \frac{-1 }{\sqrt{2+x^2}}$' + '   ' + r'$\varepsilon=sin(2\pi \frac{t}{tmax}).cos(\omega t)$' + '   ' + r'$H_i=\varepsilon .x$'
rumus2 = r'$ \sqrt{\pi} sin(\pi x)' + '   ' + r'$V= \frac{-1 }{\sqrt{2+x^2}}$' + '   ' + r'$\varepsilon=sin(2\pi \frac{t}{tmax}).cos(\omega t)$' + '   ' + r'$H_i=\varepsilon .x$'

# Plot Eigenstate
plt.figure()
ax11 = plt.subplot(313)

plt.plot(x[1:-1], v.T[50], 'y', label='50')
ax11.set_ylabel(r'$ \varphi (x)$')
plt.legend(loc='upper left')
plt.xlabel('x/L')
ax12 = plt.subplot(312)
plt.plot(x[1:-1], v.T[5], 'm', label='5')
ax12.set_ylabel(r'$ \varphi (x)$')
plt.legend(loc='upper left')
ax13 = plt.subplot(311)
plt.title('Eigen State \n' + rumus, fontsize=9)
plt.plot(x[1:-1], v.T[1], 'c', label='1')
ax13.set_ylabel(r'$ \varphi (x)$')
plt.legend(loc='upper left')

# Plot graph for t grid Nt-95000, Nt=75000, Nt-25000
plt.figure()
ax1 = plt.subplot(313)
ax1.set_ylabel(r'$| \psi (x)|^2$   t=' + format((Nt - 95000) * dt))
plt.plot(x, np.absolute(psi_m1[5000]) ** 2, 'r-', label='Finite Difference')
plt.plot(x, np.absolute(psi_m2(5000 * dt)) ** 2, 'b--', label='Eigenstate Evolution')
plt.legend(loc='upper left')
plt.xlabel('x/L')
ax2 = plt.subplot(312)
ax2.set_ylabel(r'$| \psi (x)|^2$   t=' + format((Nt - 75000) * dt))
plt.plot(x, np.absolute(psi_m1[25000]) ** 2, 'r-', label='Finite Difference')
plt.plot(x, np.absolute(psi_m2(25000 * dt)) ** 2, 'b--', label='Eigenstate Evolution')
plt.legend(loc='upper left')
ax3 = plt.subplot(311)
plt.title('Time Dependent Schrodinger Equation (TDSE) \n' + rumus, fontsize=9)
ax3.set_ylabel(r'$| \psi (x)|^2$   t=' + format((Nt - 25000) * dt))
plt.plot(x, np.absolute(psi_m1[75000]) ** 2, 'r-', label='Finite Difference')
plt.plot(x, np.absolute(psi_m2(75000 * dt)) ** 2, 'b--', label='Eigenstate Evolution')
plt.legend(loc='upper left')


# Plot animation for psi every t along x
def animate(i):
    ln1.set_data(x, np.absolute(psi_m1[100 * i]) ** 2)
    ln2.set_data(x, np.absolute(psi_m2(100 * i * dt)) ** 2)
    time_text.set_text(r'$(10^4)t=$' + '{:.1f}'.format(100 * i * dt * 1e4))


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ln1, = plt.plot([], [], 'r-', lw=2, markersize=8, label='Finite Difference')
ln2, = plt.plot([], [], 'b--', lw=3, markersize=8, label='Eigenstate Evlotion')
time_text = ax.text(0.8, 850, '', fontsize=11,
                    bbox=dict(facecolor='white', edgecolor='black'))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1000)
ax.set_ylabel(r'$|\psi(x)|^2$', fontsize=20)
ax.set_xlabel(r'$x/L$', fontsize=20)
ax.legend(loc='upper left')
if init == 1:
    ax.set_title('Time Dependent Schrodinger Equation (TDSE) \n' + rumus, fontsize=9)
elif init == 2:
    ax.set_title(' Time Dependent Schrodinger Equation (TDSE) \n' + rumus2, fontsize=9)
else:
    ax.set_title('(TSDE) \n', fontsize=9)
plt.tight_layout()
ani = animation.FuncAnimation(fig, animate, frames=1000, interval=50)
ani.save('pen1.gif', writer='pillow', fps=50, dpi=100)

# Print Hasil
print('----TDSE UAS MEKANIKA KUANTUM----')
print('Parameter input   xmin= ', xmin, ' xmax=', xmax, ' dx= ', dx, ' dt= ', dt, ' tmax=', tmax)
print('Spatial grid size=', Nx, 'Time gird size=', Nt, )
print('Periode= ', T, 'Omega= ', W)
print('dx/dt**2= ', alpha)
# Menampilkan Grafik
plt.show()
