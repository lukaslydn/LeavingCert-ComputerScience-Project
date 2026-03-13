import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Fake historical data
# -----------------------------
history = np.array([
    18, 19, 20, 21, 21, 22, 22, 23, 24, 24,
    25, 25, 26, 26, 27, 27, 26, 25, 24, 24,
    23, 22, 22, 21, 21, 20, 20, 19, 19, 18
])

# -----------------------------
# 2. Estimate parameters
# -----------------------------
mu = history.mean()
sigma = history.std(ddof=1)
phi = np.corrcoef(history[1:], history[:-1])[0, 1]

# -----------------------------
# 3. Monte Carlo settings
# -----------------------------
N = 1000   # number of simulations
H = 14      # forecast horizon (days)

rng = np.random.default_rng()
paths = np.zeros((N, H))

# -----------------------------
# 4. Simulate AR(1) paths
# -----------------------------
for i in range(N):
    paths[i, 0] = history[-1]
    for t in range(1, H):
        noise = sigma * rng.normal()
        paths[i, t] = phi * paths[i, t-1] + (1 - phi) * mu + noise

# -----------------------------
# 5. DSI definition (rolling mean)
# -----------------------------
def DSI(temp_path, window=5):
    return np.mean(temp_path[-window:])

dsi_values = np.array([DSI(paths[i]) for i in range(N)])

# -----------------------------
# 6. Probability above threshold
# -----------------------------
threshold = 27.0
probability = np.mean(dsi_values >= threshold)

print(f"Mean temperature (mu): {mu:.2f}")
print(f"Std deviation (sigma): {sigma:.2f}")
print(f"Persistence (phi): {phi:.2f}")
print(f"Probability DSI ≥ {threshold}: {probability:.3f}")

# -----------------------------
# 7. Plot sample paths
# -----------------------------
plt.figure()
for i in range(50):
    plt.plot(paths[i])
plt.xlabel("Days into future")
plt.ylabel("Temperature")
plt.title("Sample Monte Carlo Temperature Paths")
plt.show()

# -----------------------------
# 8. Plot DSI distribution
# -----------------------------
plt.figure()
plt.hist(dsi_values, bins=40, density=True)
plt.axvline(threshold)
plt.xlabel("DSI (5-day rolling mean temperature)")
plt.ylabel("Density")
plt.title("Distribution of DSI Outcomes")
plt.show()
