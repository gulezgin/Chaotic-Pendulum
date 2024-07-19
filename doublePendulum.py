import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Çift sarkaç parametreleri
m1 = 2.0  # Üst kütlenin kütlesi
m2 = 1.0  # Alt kütlenin kütlesi
L1 = 1.0  # Üst sarkacın uzunluğu
L2 = 0.5  # Alt sarkacın uzunluğu
g = 9.81  # Yerçekimi ivmesi


# Diferansiyel denklemler
def deriv(t, y):
    theta1, z1, theta2, z2 = y
    delta = theta2 - theta1

    M = m1 + m2
    mu = 1 + (m2 / m1)

    dydt = np.zeros_like(y)
    dydt[0] = z1
    dydt[1] = (m2 * g * np.sin(theta2) * np.cos(delta) - m2 * np.sin(delta) * (
                L1 * z1 ** 2 * np.cos(delta) + L2 * z2 ** 2) - (M * g * np.sin(theta1))) / (
                          L1 * (mu - m2 * np.cos(delta) ** 2))
    dydt[2] = z2
    dydt[3] = ((M * L1 * z1 ** 2 * np.sin(delta) - g * M * np.sin(theta2) + g * m1 * np.sin(theta1) * np.cos(
        delta) + m2 * L2 * z2 ** 2 * np.sin(delta) * np.cos(delta)) / (L2 * (mu - m2 * np.cos(delta) ** 2)))

    return dydt


# Başlangıç koşulları
y0 = [np.pi / 2, 0, np.pi / 2, 0]  # [theta1, z1, theta2, z2]

# Zaman aralığı
t_span = [0, 20]
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Diferansiyel denklemleri çöz
sol = solve_ivp(deriv, t_span, y0, t_eval=t_eval)

# Sonuçları görselleştir
theta1 = sol.y[0]
theta2 = sol.y[2]

x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

plt.figure()
plt.plot(x1, y1, label='Üst Sarkaç')
plt.plot(x2, y2, label='Alt Sarkaç')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Çift Sarkaç Simülasyonu')
plt.grid()
plt.show()
