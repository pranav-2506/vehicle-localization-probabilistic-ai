# Autonomous Vehicle Localization & Probabilistic Modeling

This project implements probabilistic state estimation and decision modeling techniques for autonomous vehicle localization in a simulated 2D racetrack environment.

## Overview

The system localizes a vehicle using noisy sensor measurements and evaluates competitive racing outcomes using probabilistic graphical models.

Core components include:

- Particle Filter (Monte Carlo Localization)
- Kalman Filter (Linear Gaussian State Estimation)
- Bayesian Network for modeling overtaking and crash probabilities
- Robustness analysis under heavy-tailed noise (Gaussian, Laplace, Cauchy)

## 1. Particle Filter

- Uniform particle initialization across map bounds
- Transition sampling based on vehicle motion model
- Likelihood weighting using sensor measurements
- Resampling using weighted sampling with replacement
- Convergence visualization and trajectory estimation

The particle filter estimates the vehicle’s position and orientation in real time using range sensor data.

## 2. Kalman Filter

- State vector: position and velocity
- Linear motion model
- GPS measurement updates
- Collision-aware velocity correction
- Evaluation under varying measurement noise distributions

The Kalman filter provides efficient Gaussian state estimation and is compared against particle filtering performance.

## 3. Bayesian Network Modeling

A probabilistic graphical model was constructed to analyze overtaking success, crash likelihood, and race outcomes based on:

- Relative speed
- Overtake timing
- Collision events
- Winning probability

Inference was performed using exact and sampling-based methods.

## 4. Heavy Tailed Noise Robustness

The system was stress-tested using:

- Gaussian noise
- Laplace noise
- Cauchy noise

Performance degradation under heavy-tailed distributions was analyzed to evaluate estimator robustness.

## Technologies Used

- Python
- NumPy
- Probabilistic inference algorithms (enumeration, elimination, sampling)
- FilterPy (Kalman filter implementation)
- Custom simulation environment
