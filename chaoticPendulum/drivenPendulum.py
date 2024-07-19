import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Sürücülü sarkaç parametreleri
m = 2.0  # Sarkaç kütlesi
L = 1.0  # Sarkaç uzunluğu
g = 9.81  # Yerçekimi ivmesi
b = 0.2  # Sönüm katsayısı
A = 1.2  # Sürücü kuvvet genliği
omega = 2/3  # Sürücü kuvvet frekansı

# Diferansiyel denklemler
def deriv(t, y):
    theta, omega_theta = y
    dydt = [
        omega_theta,
        -b * omega_theta - (g / L) * np.sin(theta) + A * np.cos(omega * t)
    ]
    return dydt

# Başlangıç koşulları
theta0 = 0.2  # Başlangıç açısı (radyan)
omega_theta0 = 0.0  # Başlangıç açısal hızı
y0 = [theta0, omega_theta0]

# Zaman aralığı
t_span = [0, 100]
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Diferansiyel denklemleri çöz
sol = solve_ivp(deriv, t_span, y0, t_eval=t_eval)

# Sonuçları görselleştir
theta = sol.y[0]

x = L * np.sin(theta)
y = -L * np.cos(theta)

plt.figure()
plt.plot(t_eval, theta, label='Açı (theta)')
plt.xlabel('Zaman')
plt.ylabel('Açı (theta)')
plt.title('Sürücülü Sarkaç Simülasyonu')
plt.legend()
plt.grid()
plt.show()

# Faz uzayını (theta - omega_theta) çiz
plt.figure()
plt.plot(theta, sol.y[1], label='Faz Uzayı')
plt.xlabel('Açı (theta)')
plt.ylabel('Açısal Hız (omega_theta)')
plt.title('Sürücülü Sarkaç Faz Uzayı')
plt.legend()
plt.grid()
plt.show()
