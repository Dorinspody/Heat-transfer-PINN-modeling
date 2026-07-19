
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict
import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
import os


OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) #where th plots save

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



L = 100.0         # length of object
T_max = 40.0      #  time
ALPHA = 2.0       # thermal diffusivity
U_HOT = 100.0     # temperature at the hot end (x = L)
U_COLD = 0.0      # initial temp

def initial_condition(x):
    return np.full_like(x, U_COLD)

def analytical_u(x, t, n_terms=200):    # Analytical solution of 1D transient heat conduction using a Fourier sine-series expansion

    u_steady = U_COLD + (U_HOT - U_COLD) * x / L
    v = np.zeros_like(x, dtype=float)
    dU = U_HOT - U_COLD
    for n in range(1, n_terms + 1):
        b_n = 2 * dU * ((-1) ** n) / (n * np.pi)
        v += b_n * np.sin(n * np.pi * x / L) * np.exp(-ALPHA * (n * np.pi / L) ** 2 * t)
    return u_steady + v



# NORMALIZATION (critical for stable training)
alpha_hat = ALPHA * T_max / L ** 2

def to_hat_xt(x, t):
    return x / L, t / T_max

def to_hat_u(u):
    return (u - U_COLD) / (U_HOT - U_COLD)

def from_hat_u(u_hat):
    return U_COLD + u_hat * (U_HOT - U_COLD)



# DATA GENERATION collocation  IC  BC points, sampled directly

def sample_collocation(n, seed=None):
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    unit = sampler.random(n)   # already in [0,1]^2 -> use directly as (x_hat, t_hat)
    return torch.tensor(unit, dtype=torch.float32, requires_grad=True, device=device)

def sample_ic(n, seed=None):
    sampler = qmc.LatinHypercube(d=1, seed=seed)
    x_hat = sampler.random(n)          # in [0,1]
    t_hat = np.zeros_like(x_hat)
    x_ic = torch.tensor(np.hstack([x_hat, t_hat]), dtype=torch.float32, device=device)
    u_hat_ic = np.zeros_like(x_hat)    # normalized cold IC is always 0
    u_ic = torch.tensor(u_hat_ic, dtype=torch.float32, device=device)
    return x_ic, u_ic

def sample_bc(n, seed=None):
    sampler = qmc.LatinHypercube(d=1, seed=seed)
    t_hat = sampler.random(n)          # in [0,1]
    zeros = np.zeros_like(t_hat)
    ones = np.ones_like(t_hat)

    x_cold = np.hstack([zeros, t_hat])
    x_hot = np.hstack([ones, t_hat])
    x_bc = np.vstack([x_cold, x_hot])
    u_bc = np.vstack([np.zeros_like(t_hat), np.ones_like(t_hat)])

    return (torch.tensor(x_bc, dtype=torch.float32, device=device),
            torch.tensor(u_bc, dtype=torch.float32, device=device))

# the number of data in each category
N_COLLOC = 2000
N_IC = 200
N_BC = 150
RESAMPLE_EVERY = 200

x_colloc = sample_collocation(N_COLLOC, seed=0)
x_ic, u_ic = sample_ic(N_IC, seed=1)
x_bc, u_bc = sample_bc(N_BC, seed=2)



# PINN Model

class PINN(nn.Module):
    def __init__(self, layers_size: List[int], device, alpha: float = None):
        super().__init__()
        self.activation = nn.Tanh()
        self.loss_mse = nn.MSELoss(reduction='mean')
        self.alpha = (nn.Parameter(torch.tensor(1.0, device=device))
                      if alpha is None else alpha)
        self.device = device
        self.layers = nn.ModuleList([
            nn.Linear(layers_size[i], layers_size[i + 1])
            for i in range(len(layers_size) - 1)
        ]).to(device)
        self.loss_history, self.epoch_history, self.alpha_history = [], [], []
        self.init_layers()

    def init_layers(self):
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight.data)
            nn.init.zeros_(layer.bias.data)

    def forward(self, x: torch.Tensor):
        inp = x
        for layer in self.layers[:-1]:
            inp = self.activation(layer(inp))
        return self.layers[-1](inp)

    def loss_data(self, x, u):
        return self.loss_mse(self.forward(x), u)

    def loss_pde(self, x_colloc: torch.Tensor):
        x = x_colloc[:, [0]]
        t = x_colloc[:, [1]]
        u = self.forward(torch.cat([x, t], dim=1))

        ones = torch.ones_like(u)
        u_t = autograd.grad(u, t, ones, create_graph=True)[0]
        u_x = autograd.grad(u, x, ones, create_graph=True)[0]
        u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

        f = u_t - self.alpha * u_xx
        return f.square().mean()

    def total_loss(self, x_ic, u_ic, x_bc, u_bc, x_colloc):
        l_ic = self.loss_data(x_ic, u_ic)
        l_bc = self.loss_data(x_bc, u_bc)
        l_pde = self.loss_pde(x_colloc)
        loss = l_pde + 10.0 * l_ic + 10.0 * l_bc
        return loss, l_pde, l_ic, l_bc

    def start_train(self, training_data: Dict[str, torch.Tensor], max_iter: int,
                     resample_every: int = None, resample_fn=None):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        for i in range(max_iter):
            if resample_every and resample_fn and i > 0 and i % resample_every == 0:
                training_data.update(resample_fn(i))

            optimizer.zero_grad()
            loss, l_pde, l_ic, l_bc = self.total_loss(**training_data)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                self.loss_history.append(loss.item())
                self.epoch_history.append(i)
                log = f"Epoch: {i:5d} | Loss: {loss.item():.4e} | PDE: {l_pde.item():.4e} | IC: {l_ic.item():.4e} | BC: {l_bc.item():.4e}"
                if isinstance(self.alpha, torch.Tensor):
                    log += f" | alpha: {self.alpha.item():.4f}"
                    self.alpha_history.append(self.alpha.item())
                print(log)
        return training_data



# training

model = PINN(layers_size=[2, 32, 32, 32, 1], device=device, alpha=alpha_hat)

training_data = dict(x_ic=x_ic, u_ic=u_ic, x_bc=x_bc, u_bc=u_bc, x_colloc=x_colloc)

def resample_fn(epoch):
    return {"x_colloc": sample_collocation(N_COLLOC, seed=epoch)}

training_data = model.start_train(training_data, max_iter=3000,
                                   resample_every=RESAMPLE_EVERY, resample_fn=resample_fn)

x_colloc_final = sample_collocation(3000, seed=999)
lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=800,
                     tolerance_grad=1e-11, tolerance_change=1e-13,
                     history_size=50, line_search_fn="strong_wolfe")

it = {"i": 0}
def closure():
    lbfgs.zero_grad()
    loss, l_pde, l_ic, l_bc = model.total_loss(x_ic, u_ic, x_bc, u_bc, x_colloc_final)
    loss.backward()
    it["i"] += 1
    if it["i"] % 200 == 0:
        print(f"[LBFGS] Iter {it['i']:5d} | Loss: {loss.item():.4e} | "
              f"PDE: {l_pde.item():.4e} | IC: {l_ic.item():.4e} | BC: {l_bc.item():.4e}")
    return loss

lbfgs.step(closure)
print("Training complete.")

# validation with PINN vs analytical Fourier-series solution

model.eval()

n_x, n_t = 200, 200
x_grid = np.linspace(0, L, n_x)
t_grid = np.linspace(0, T_max, n_t)
Xg, Tg = np.meshgrid(x_grid, t_grid)   # rows = t, cols = x

Xg_hat, Tg_hat = Xg / L, Tg / T_max
xt_hat_flat = torch.tensor(
    np.stack([Xg_hat.flatten(), Tg_hat.flatten()], axis=1), dtype=torch.float32, device=device
)
with torch.no_grad():
    u_hat_pred_grid = model(xt_hat_flat).cpu().numpy().reshape(n_t, n_x)
u_pred_grid = from_hat_u(u_hat_pred_grid)   # back to physical temperature units

u_true_grid = analytical_u(Xg, Tg)
err_grid = np.abs(u_pred_grid - u_true_grid)

l2_err = np.sqrt(np.mean(err_grid ** 2))
max_err = np.max(err_grid)
rel_l2 = np.linalg.norm(err_grid) / np.linalg.norm(u_true_grid)
print(f"\n[VALIDATION] Max abs error: {max_err:.4f} | L2 error: {l2_err:.4f} "
      f"| Relative L2 error: {rel_l2:.4%}")

# plotting
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

im0 = axes[0].pcolormesh(t_grid, x_grid, u_pred_grid.T, cmap='jet', shading='auto')
axes[0].set_title('PINN Prediction')
axes[0].set_xlabel('t'); axes[0].set_ylabel('x')
fig.colorbar(im0, ax=axes[0], label='u(x,t)')

im1 = axes[1].pcolormesh(t_grid, x_grid, u_true_grid.T, cmap='jet', shading='auto')
axes[1].set_title('Analytical Solution')
axes[1].set_xlabel('t'); axes[1].set_ylabel('x')
fig.colorbar(im1, ax=axes[1], label='u(x,t)')

im2 = axes[2].pcolormesh(t_grid, x_grid, err_grid.T, cmap='viridis', shading='auto')
axes[2].set_title(f'Absolute Error (max={max_err:.3f})')
axes[2].set_xlabel('t'); axes[2].set_ylabel('x')
fig.colorbar(im2, ax=axes[2], label='|error|')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pinn_1d_validation_heatmaps.png'), dpi=150)
print("Saved pinn_1d_validation_heatmaps.png")


snapshot_times = [0.0, 2.0, 5.0, 10.0, 20.0, 40.0]
fig, ax = plt.subplots(figsize=(7, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(snapshot_times)))
for c, t_val in zip(colors, snapshot_times):
    x_line = np.linspace(0, L, 300)
    t_line = np.full_like(x_line, t_val)
    xt_hat = torch.tensor(np.stack([x_line / L, t_line / T_max], axis=1),
                           dtype=torch.float32, device=device)
    with torch.no_grad():
        u_p = from_hat_u(model(xt_hat).cpu().numpy().flatten())
    u_a = analytical_u(x_line, t_line)
    ax.plot(x_line, u_a, color=c, linestyle='-', linewidth=2)
    ax.plot(x_line, u_p, color=c, linestyle='--', linewidth=1.5,
             label=f"t={t_val:g}")
ax.plot([], [], 'k-', label='Analytical (solid)')
ax.plot([], [], 'k--', label='PINN (dashed)')
ax.set_xlabel('x'); ax.set_ylabel('u(x,t)')
ax.set_title('PINN vs Analytical: Temperature Profiles Over Time')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pinn_1d_validation_profiles.png'), dpi=150)
print("Saved pinn_1d_validation_profiles.png")

# heatmap
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.pcolormesh(t_grid, x_grid, u_pred_grid.T, cmap='jet', shading='auto')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('u(x, t)')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_title('PINN Prediction of the 1D Heat Equation')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pinn_1d_heatmap.png'), dpi=150)
print("Saved pinn_1d_heatmap.png")

# Contour plot
fig, ax = plt.subplots(figsize=(6, 5))
cf = ax.contourf(t_grid, x_grid, u_pred_grid.T, levels=20, cmap='jet')
cs = ax.contour(t_grid, x_grid, u_pred_grid.T, levels=8, colors='k', linewidths=0.5)
ax.clabel(cs, inline=True, fontsize=7)
fig.colorbar(cf, ax=ax, label='u(x,t)')
ax.set_xlabel('t'); ax.set_ylabel('x')
ax.set_title('PINN Prediction (Contour): 1D Heat Equation')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pinn_1d_contour.png'), dpi=150)
print("Saved pinn_1d_contour.png")

#  Loss history
plt.figure(figsize=(6, 4))
plt.plot(model.epoch_history, model.loss_history)
plt.yscale('log')
plt.xlabel('Epoch'); plt.ylabel('Loss (log scale)')
plt.title('Training Loss History')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pinn_1d_loss.png'), dpi=150)
print("Saved pinn_1d_loss.png")
