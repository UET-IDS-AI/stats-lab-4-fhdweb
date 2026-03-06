"""
AI Stats Lab
Random Variables and Distributions
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad


# =========================================================
# QUESTION 1 — CDF Probabilities
# =========================================================

def cdf_probabilities():

    # Analytical probabilities
    analytic_gt5 = math.exp(-5)
    analytic_lt5 = 1 - math.exp(-5)
    analytic_interval = math.exp(-3) - math.exp(-7)

    # Monte Carlo simulation
    np.random.seed(42)
    samples = np.random.exponential(scale=1, size=100000)

    simulated_gt5 = np.mean(samples > 5)

    return (
        analytic_gt5,
        analytic_lt5,
        analytic_interval,
        simulated_gt5
    )


# =========================================================
# QUESTION 2 — PDF Validation and Plot
# =========================================================

def pdf_validation_plot():

    # Define the candidate PDF
    def f(x):
        return 2 * x * math.exp(-(x**2))

    # Compute the integral from 0 to infinity
    integral_value, _ = quad(f, 0, np.inf)

    # Check non-negativity (on sample points)
    x_vals = np.linspace(0, 3, 100)
    y_vals = 2 * x_vals * np.exp(-(x_vals**2))
    non_negative = np.all(y_vals >= 0)

    # Determine if valid PDF
    is_valid_pdf = non_negative and abs(integral_value - 1) < 1e-3

    # Plot the PDF
    plt.figure()
    plt.plot(x_vals, y_vals)
    plt.title("PDF: f(x) = 2x e^{-x^2}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.close()

    return (
        integral_value,
        is_valid_pdf
    )


# =========================================================
# QUESTION 3 — Exponential Distribution
# =========================================================

def exponential_probabilities():

    # Analytical probabilities
    analytic_gt5 = math.exp(-5)
    analytic_interval = math.exp(-1) - math.exp(-3)

    # Simulation
    np.random.seed(42)
    samples = np.random.exponential(scale=1, size=100000)

    simulated_gt5 = np.mean(samples > 5)
    simulated_interval = np.mean((samples > 1) & (samples < 3))

    return (
        analytic_gt5,
        analytic_interval,
        simulated_gt5,
        simulated_interval
    )


# =========================================================
# QUESTION 4 — Gaussian Distribution
# =========================================================

def gaussian_probabilities():

    mu = 10
    sigma = 2

    # Analytical probabilities
    analytic_le12 = norm.cdf(12, loc=mu, scale=sigma)
    analytic_interval = norm.cdf(12, loc=mu, scale=sigma) - norm.cdf(8, loc=mu, scale=sigma)

    # Simulation
    np.random.seed(42)
    samples = np.random.normal(mu, sigma, 100000)

    simulated_le12 = np.mean(samples <= 12)
    simulated_interval = np.mean((samples > 8) & (samples < 12))

    return (
        analytic_le12,
        analytic_interval,
        simulated_le12,
        simulated_interval
    )
