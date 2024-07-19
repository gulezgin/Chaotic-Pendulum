import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Çift sarkaç parametreleri
m1 = 3.0
m2 = 1.0
L1 = 1.0
L2 = 0.5
g = 9.81

# Diferansiyel denklemler
def deriv(t, y):
    theta1, omega1, theta2, omega2 = y
    delta = theta2 - theta1

    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) * np.cos(delta)
    den2 = (L2 / L1) * den1

    dydt = np.zeros_like(y)
    dydt[0] = omega1
    dydt[1] = (m2 * L1 * omega1 * omega1 * np.sin(delta) * np.cos(delta) +
               m2 * g * np.sin(theta2) * np.cos(delta) +
               m2 * L2 * omega2 * omega2 * np.sin(delta) -
               (m1 + m2) * g * np.sin(theta1)) / den1
    dydt[2] = omega2
    dydt[3] = (-m2 * L2 * omega2 * omega2 * np.sin(delta) * np.cos(delta) +
               (m1 + m2) * (L1 * omega1 * omega1 * np.sin(delta) -
               g * np.sin(theta2) + g * np.sin(theta1) * np.cos(delta))) / den2

    return dydt

# Başlangıç koşulları
theta1_0 = np.pi / 2
theta2_0 = np.pi / 2
omega1_0 = 0.0
omega2_0 = 0.0
y0 = [theta1_0, omega1_0, theta2_0, omega2_0]

# Zaman aralığı
t_span = [0, 20]
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Başlangıçta statik olarak sarkacı tutmak için
initial_theta1 = np.pi / 2
initial_theta2 = np.pi / 2
holding = True

# Sonuçları görselleştirmek ve animasyon yapmak için hazırlık
sol = solve_ivp(deriv, t_span, y0, t_eval=t_eval)
theta1 = sol.y[0]
theta2 = sol.y[2]

x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

fig, ax = plt.subplots()
ax.set_xlim(-L1 - L2 - 0.5, L1 + L2 + 0.5)
ax.set_ylim(-L1 - L2 - 0.5, L1 + L2 + 0.5)

line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], 'r-', lw=1)
coords = []

def init():
    line.set_data([], [])
    trace.set_data([], [])
    return line, trace

def update(frame):
    global holding
    if holding:
        return line, trace
    thisx = [0, x1[frame], x2[frame]]
    thisy = [0, y1[frame], y2[frame]]
    line.set_data(thisx, thisy)
    coords.append((x2[frame], y2[frame]))
    trace.set_data(*zip(*coords))
    return line, trace

def on_click(event):
    global initial_theta1, initial_theta2, holding, y0, sol, theta1, theta2, x1, y1, x2, y2, coords
    if event.inaxes is not None:
        holding = False
        initial_theta1 = np.arctan2(event.xdata, -event.ydata)
        initial_theta2 = np.arctan2(event.xdata, -event.ydata)
        y0 = [initial_theta1, 0, initial_theta2, 0]
        sol = solve_ivp(deriv, t_span, y0, t_eval=t_eval)
        theta1 = sol.y[0]
        theta2 = sol.y[2]
        x1 = L1 * np.sin(theta1)
        y1 = -L1 * np.cos(theta1)
        x2 = x1 + L2 * np.sin(theta2)
        y2 = y1 - L2 * np.cos(theta2)
        coords.clear()
        ani.event_source.start()

fig.canvas.mpl_connect('button_press_event', on_click)

ani = FuncAnimation(fig, update, frames=range(len(t_eval)),
                    init_func=init, blit=True, interval=20)

plt.title('Çift Bağlantılı Kaotik Sarkaç Animasyonu')
plt.show()

# Koordinatları kaydetmek için gerekli kod
import csv

with open('sarkac_koordinatlariSimulation.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["X", "Y"])
    writer.writerows(coords)
