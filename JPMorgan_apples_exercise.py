import numpy as np
import csv
import pandas
import matplotlib.pyplot as plt

with open('apples_exercise.csv',newline='') as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    data = list(data)
data_arr = np.array(data)
data_num = np.delete(np.delete(data_arr, 0, axis=1), 0, axis=0).astype(float)
for i in range(data_num.shape[0]):
    # Turn into GBP
    data_num[i,0] *= (1/data_num[i,2])

# Random Walk process for German Apples
German_apples = data_num[:,0]
X0_G = German_apples[0]
T_G = 2250
dt_G = 1
N_G = int(T_G/dt_G)

X_G = np.zeros(N_G)
X_G[0]=X0_G

# Idiosyncratic noise
increments = np.diff(German_apples)
mu_hat = np.mean(increments)
sigma_hat = np.std(increments, ddof=1)

for t in range(1, T_G):
    dW = np.random.normal(0, 1)  # standard normal
    X_G[t] = X_G[t-1] + mu_hat*dt_G + sigma_hat*np.sqrt(dt_G)*dW

plt.plot(np.linspace(0, T_G, N_G), X_G)
plt.title("Random Walk Process Simulation")
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.show()

# Parameters for OU process for UK Apples
UK_apples = data_num[:,1]
X0_UK = UK_apples[0]
T_UK = 2250
dt_UK = 1
N_UK = int(T_UK/dt_UK)

X_UK = np.zeros(N_UK)
X_UK[0] = X0_UK

# Mean Reversion Speed Parameter Calculation

# AR(1) Model
# x_t = \alpha + \phi x_{t-1} + \varepsilon_t, e_t \sim (0,\sigma_\varepsilon^2)
UK_apples_lag = UK_apples[:-1]
Y_UK = UK_apples[1:]
ones = np.ones_like(UK_apples_lag)

# OLS for [alpha, phi]
A = np.vstack([ones, UK_apples_lag]).T
coef, residuals_sum_sq, rank, s = np.linalg.lstsq(A, Y_UK, rcond=None)
alpha_hat, phi_hat = coef[0], coef[1]

theta = alpha_hat / (1.0-phi_hat)

# Mean-Reversion Speed Calculation
kappa = -np.log(phi_hat) / dt_UK

# residuals and residual sample variance
residuals = Y_UK - (alpha_hat + phi_hat * UK_apples_lag)
sigma_eps2 = np.std(residuals, ddof=1)

# OU volatility
sigma_ou = np.sqrt( 2*kappa * sigma_eps2 / (1 - np.exp(-2*kappa*dt_UK)))

for t in range(1, N_UK):
    dW = np.sqrt(dt_UK) * np.random.normal(0, 1)
    X_UK[t] = X_UK[t-1] + kappa * (theta - X_UK[t-1]) * dt_UK + sigma_ou * dW

UK_apples_OU_params = {
        "phi": phi_hat,
        "alpha": alpha_hat,
        "residuals": residuals,
        "sigma_eps2": sigma_eps2,
        "theta": theta,
        "kappa": kappa,
        "sigma_ou": sigma_ou,
        "n_obs": N_UK,
        "dt": dt_UK,
        "dW": dW
    }

plt.plot(np.linspace(0, T_UK, N_UK), X_UK)
plt.title("Ornstein-Uhlenbeck Process Simulation")
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.show()

# Predictor for UK Apples based on German Apple Pricing
# Simple Polynomial Regression

UK_apples_reshaped = data_num[:, 1].reshape(-1, 1)
German_apples = data_num[:, 0]

def create_poly(X, degree):
    poly_features = [X ** i for i in range(degree + 1)]
    poly_features = [f if f.ndim == 2 else f.reshape(-1,1) for f in poly_features]
    return np.concatenate(poly_features, axis=1)

def fit_linear(X, y):
    theta = np.linalg.solve(X.T @ X, X.T @ y)
    return theta

def predict(X, theta):
    return X @ theta

degrees = [1, 2, 3, 5, 10] 
plt.figure(figsize=(10,6))
plt.scatter(UK_apples_reshaped, German_apples, color='blue', label='Observed')

# Dense x for smooth curve
X_plot = np.linspace(UK_apples.min(), UK_apples.max(), 500).reshape(-1,1)

for d in degrees:
    X_poly = create_poly(UK_apples, d)
    theta = fit_linear(X_poly, German_apples)
    
    X_plot_poly = create_poly(X_plot, d)
    y_plot = predict(X_plot_poly, theta)
    
    plt.plot(X_plot, y_plot, label=f'Degree {d}')

plt.xlabel("UK Apple Prices")
plt.ylabel("German Apple Prices")
plt.title("Polynomial Regression: UK → German Apples")
plt.legend()
plt.show()
