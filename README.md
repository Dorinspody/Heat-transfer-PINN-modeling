1D Heat Equation PINN

This project implements a Physics-Informed Neural Network (PINN) to solve the one-dimensional transient heat equation. Rather than relying on labeled data, the network is trained by minimizing residuals derived from the governing PDE, the boundary conditions, and the initial condition. Collocation points are generated via Latin Hypercube Sampling and periodically resampled to improve coverage of the domain during training. The model's predictions are validated against the closed-form analytical solution, and results are visualized as heatmaps, contour plots, and temperature profiles over time. All output figures are automatically saved to the directory containing the script.The model achieved a training loss of 2.1375 × 10⁻³ and a validation loss of 2.8817 × 10⁻⁵. the model obtained an MSE of 0.5489 and an R² score of 0.998513, demonstrating excellent agreement with the analytical solution.
Governing Equation:

Loss function: L = LPDE ​+ LBC​ + LIC
​
PDE: u_t = alpha * u_xx​

Initial condition: u(x, 0) = 0

Boundary conditions: u(0, t) = 0, u(L, t) = 100.0
