"""
Exact Python replica of PSOMbest_Main_Master.m
(Twumasi et al. 2024 reproduction driver, DFFG-style, 1548-line MATLAB original)

Library mapping (MATLAB -> Python):
  rng(seed,'twister')      -> numpy.random.RandomState(seed)   [MT19937 family; streams are
                              statistically equivalent but NOT bit-identical to MATLAB]
  lhsdesign (toolbox-free) -> local_lhsdesign()                [identical maximin LHS]
  kmeans (toolbox-free)    -> local_kmeans()                   [identical k-means++ Lloyd]
  writecell/xlsread        -> pandas.ExcelWriter(openpyxl)
  figure/saveas/semilogy   -> matplotlib (Agg)
  .mat -v7.3               -> .npz (scipy unavailable on this machine)
  diary                    -> tee'd log file
GPU: RTX 4050 present but no torch/cupy/CUDA stack installed -> CPU only.
Run:  python psombest_main_master_python.py --mode full|sample|debug [--workers N]

CORRECTED VERSION (2026-11-04). Changes made after comparing line by line with PSOMbest_Main_Master.m:
  1. local_lhsdesign: randperm in MATLAB counts from 1, rs.permutation counts from 0. Added "+ 1" so the design
     lies in (0, 1) again (before: about -0.006 to 0.992, i.e. some starting points fell below the lower bound).
  2. run_pso_awd: the hook computed the new inertia but returned the old one, so inertia stayed at 1.0 and the delay
     term was always zero. Now, as in MATLAB, inertia, c1 and c2 are updated at the END of each iteration and used
     from the next one (iteration 1 uses w=1, c1=1.5, c2=2.0). The evolutionary state is taken at the start of the
     iteration, exactly as in the MATLAB file.
  3. local_kmeans: the "previous labels" start as all in cluster 1 (index 0 here), as in MATLAB (ones).
  4. run_real_coded_ga: elite count uses MATLAB rounding (half away from zero) instead of Python's round-half-even.
Everything else is unchanged. Random streams still differ from MATLAB (MT19937 here), so compare statistically.
"""

import argparse
import math
import os
import re
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np
from openpyxl import Workbook

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

REALMAX = float(np.finfo(np.float64).max)

STUDY_NAME = 'PSOMbest_Python_Reproduction'
MASTER_FILE = os.path.abspath(__file__)
RESULTS_ROOT = os.path.join(os.path.dirname(MASTER_FILE), STUDY_NAME + '_Results')
FIGURES_DIR = os.path.join(RESULTS_ROOT, 'Figures')
COMPOSITES_DIR = os.path.join(RESULTS_ROOT, 'Composite_Plots')
EXCEL_FILENAME = os.path.join(RESULTS_ROOT, STUDY_NAME + '_Statistics.xlsx')
NPZ_FILENAME = os.path.join(RESULTS_ROOT, STUDY_NAME + '_RawResults.npz')
LOG_FILENAME = os.path.join(RESULTS_ROOT, STUDY_NAME + '_Log.txt')


# %% ===== LOCAL BENCHMARK / MATH FUNCTIONS (mirror of MATLAB locals) ==========

def expand_bound(bound, dimension):
    expanded = np.full(dimension, float(bound)) if np.isscalar(bound) \
        else np.reshape(np.asarray(bound, float), -1).copy()
    if expanded.size != dimension:
        raise ValueError('Bound vector length must equal the problem dimension.')
    return expanded


def safe_relative_difference(value, reference):
    absolute_difference = abs(value - reference)
    if reference == 0:
        return 0.0 if value == 0 else absolute_difference
    return absolute_difference / abs(reference)


def evaluate_objective(objective_function, position):
    value = objective_function(np.asarray(position, float).reshape(1, -1))
    value = float(np.real(np.ravel(value)[0]))
    if not np.isfinite(value):
        value = REALMAX
    return value


def psombest_objective(x, base_name, noise_sigma, rs):
    dimension = x.shape[1]
    if base_name == 'F1':
        weights = np.arange(dimension, 0, -1, dtype=float)
        y = np.sum((x ** 2) * weights, axis=1)
    elif base_name == 'F2':
        indices = np.arange(1, dimension + 1, dtype=float)
        weighted_sum = np.sum(0.5 * indices * x, axis=1)
        y = np.sum(x ** 2, axis=1) + weighted_sum ** 2 + weighted_sum ** 4
    elif base_name == 'F3':
        y = (x[:, 0] - 1) ** 2 + np.sum(
            np.arange(2, dimension + 1, dtype=float)
            * (2 * x[:, 1:] ** 2 - x[:, :-1]) ** 2, axis=1)
    elif base_name in ('F4', 'F18'):
        powers = np.arange(2, dimension + 2, dtype=float)[None, :]
        y = np.sum(np.abs(x) ** powers, axis=1)
    elif base_name == 'F5':
        x1, x2 = x[:, 0], x[:, 1]
        y = x1 ** 2 + 2 * x2 ** 2 - 0.3 * np.cos(3 * np.pi * x1) \
            - 0.4 * np.cos(4 * np.pi * x2) + 0.7
    elif base_name == 'F6':
        x1, x2 = x[:, 0], x[:, 1]
        y = 0.26 * (x1 ** 2 + x2 ** 2) - 0.48 * x1 * x2
    elif base_name == 'F7':
        r2 = x[:, 0] ** 2 + x[:, 1] ** 2
        y = -(1 + np.cos(12 * np.sqrt(r2))) / (0.5 * r2 + 2)
    elif base_name == 'F8':
        y = 10 * dimension + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1)
    elif base_name == 'F9':
        y = -20 * np.exp(-0.2 * np.sqrt(np.mean(x ** 2, axis=1))) \
            - np.exp(np.mean(np.cos(2 * np.pi * x), axis=1)) + 20 + math.e
    elif base_name == 'F10':
        w = 1 + (x - 1) / 4
        first = np.sin(np.pi * w[:, 0]) ** 2
        middle = np.sum((w[:, :-1] - 1) ** 2
                        * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1) ** 2), axis=1)
        final = (w[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:, -1]) ** 2)
        y = first + middle + final
    elif base_name == 'F11':
        indices = np.arange(1, dimension + 1, dtype=float)
        y = 1 + np.sum(x ** 2, axis=1) / 4000 \
            - np.prod(np.cos(x / np.sqrt(indices)), axis=1)
    elif base_name == 'F12':
        y = 0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x, axis=1)
    elif base_name == 'F13':
        nx = x + noise_sigma * rs.standard_normal(x.shape)
        y = np.sum(nx ** 2, axis=1)
    elif base_name == 'F14':
        nx = x + noise_sigma * rs.standard_normal(x.shape)
        y = 10 * dimension + np.sum(nx ** 2 - 10 * np.cos(2 * np.pi * nx), axis=1)
    elif base_name == 'F15':
        nx = x + noise_sigma * rs.standard_normal(x.shape)
        y = -20 * np.exp(-0.2 * np.sqrt(np.mean(nx ** 2, axis=1))) \
            - np.exp(np.mean(np.cos(2 * np.pi * nx), axis=1)) + 20 + math.e
    elif base_name == 'F16':
        nx = x + noise_sigma * rs.standard_normal(x.shape)
        indices = np.arange(1, dimension + 1, dtype=float)
        y = 1 + np.sum(nx ** 2, axis=1) / 4000 \
            - np.prod(np.cos(nx / np.sqrt(indices)), axis=1)
    elif base_name == 'F17':
        y = np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (x[:, :-1] - 1) ** 2, axis=1)
    else:
        raise ValueError('Unknown benchmark function: %s' % base_name)
    y = np.real(np.asarray(y, float)).ravel()
    y[~np.isfinite(y)] = REALMAX
    return y


def psombest_benchmark_info(function_name):
    n = function_name.upper()
    if n == 'F1':
        return -65.536, 65.536, 30, 0, 0, 'Rotated hyper-ellipsoid'
    if n == 'F2':
        return -5, 10, 30, 0, 0, 'Zakharov'
    if n == 'F3':
        return -10, 10, 30, 0, 0, 'Dixon-Price'
    if n == 'F4':
        return -1, 1, 30, 0, 0, 'Sum of different powers'
    if n == 'F5':
        return -100, 100, 2, 0, 0, 'Bohachevsky 1'
    if n == 'F6':
        return -10, 10, 2, 0, 0, 'Matyas'
    if n == 'F7':
        return -5.12, 5.12, 2, -1, -1, 'Drop-Wave'
    if n == 'F8':
        return -5.12, 5.12, 2, 0, 0, 'Rastrigin'
    if n == 'F9':
        return -32.768, 32.768, 100, 0, 0, 'Ackley'
    if n == 'F10':
        return -10, 10, 100, 0, 0, 'Levy'
    if n == 'F11':
        return -600, 600, 100, 0, 0, 'Griewank'
    if n == 'F12':
        return -5, 5, 100, -39.1661657037714 * 100, -3916.16599, 'Styblinski-Tang'
    if n == 'F13':
        return -100, 400, 30, 0, 0, 'Noisy Sphere'
    if n == 'F14':
        return -5.12, 5.12, 30, 0, 0, 'Noisy Rastrigin'
    if n == 'F15':
        return -32, 32, 30, 0, 0, 'Noisy Ackley'
    if n == 'F16':
        return -600, 600, 30, 0, 0, 'Noisy Griewank'
    if n == 'F17_2D':
        return -5, 10, 2, 0, 0, 'Rosenbrock (2D)'
    if n == 'F17_30D':
        return -5, 10, 30, 0, 0, 'Rosenbrock (30D)'
    if n == 'F17_100D':
        return -5, 10, 100, 0, 0, 'Rosenbrock (100D)'
    if n == 'F18_2D':
        return -1, 1, 2, 0, 0, 'Powell sum (2D)'
    if n == 'F18_30D':
        return -1, 1, 30, 0, 0, 'Powell sum (30D)'
    if n == 'F18_100D':
        return -1, 1, 100, 0, 0, 'Powell sum (100D)'
    raise ValueError('Unknown benchmark label: %s' % function_name)


# %% ===== SELECTION / SAMPLING / CLUSTERING HELPERS ==========================

def rank_selection_probabilities(costs, pressure):
    population_size = len(costs)
    sorted_index = np.argsort(costs, kind='stable')
    ranks = np.zeros(population_size)
    ranks[sorted_index] = np.arange(population_size, dtype=float)
    denominator = max(1, population_size - 1)
    weights = np.exp(-pressure * ranks / denominator)
    return weights / np.sum(weights)


def roulette_select(probabilities, rs):
    cumulative_probability = np.cumsum(np.asarray(probabilities).ravel())
    hits = np.where(rs.random_sample() <= cumulative_probability)[0]
    return int(hits[0]) if hits.size else len(probabilities) - 1


def local_lhsdesign(number_of_points, dimension, number_of_trials, rs):
    best_minimum_distance = -np.inf
    design = np.zeros((number_of_points, dimension))
    for _ in range(number_of_trials):
        candidate = np.zeros((number_of_points, dimension))
        for coordinate in range(dimension):
            candidate[:, coordinate] = \
                (rs.permutation(number_of_points) + 1 - rs.random_sample(number_of_points)) \
                / number_of_points
        squared_norm = np.sum(candidate ** 2, axis=1)
        squared_distance = squared_norm[:, None] + squared_norm[None, :] \
            - 2 * (candidate @ candidate.T)
        np.fill_diagonal(squared_distance, np.inf)
        minimum_distance = math.sqrt(max(0.0, float(np.min(squared_distance))))
        if minimum_distance > best_minimum_distance:
            best_minimum_distance = minimum_distance
            design = candidate
    return design


def local_kmeans(data, number_of_clusters, maximum_iterations, tolerance, rs):
    number_of_points, dimension = data.shape
    number_of_clusters = int(min(max(1, round(number_of_clusters)), number_of_points))

    centers = np.zeros((number_of_clusters, dimension))
    centers[0] = data[rs.randint(number_of_points)]
    closest_distance = np.sum((data - centers[0]) ** 2, axis=1)

    for cluster_number in range(1, number_of_clusters):
        total_distance = float(np.sum(closest_distance))
        if (not np.isfinite(total_distance)) or total_distance <= np.finfo(float).eps:
            selected_index = rs.randint(number_of_points)
        else:
            threshold = rs.random_sample() * total_distance
            cumulative_distance = np.cumsum(closest_distance)
            hits = np.where(cumulative_distance >= threshold)[0]
            selected_index = int(hits[0]) if hits.size else number_of_points - 1
        centers[cluster_number] = data[selected_index]
        new_distance = np.sum((data - centers[cluster_number]) ** 2, axis=1)
        closest_distance = np.minimum(closest_distance, new_distance)

    cluster_index = np.zeros(number_of_points, dtype=int)   # MATLAB starts with ones (cluster 1)
    squared_distance = np.zeros((number_of_points, number_of_clusters))
    for _ in range(maximum_iterations):
        for cluster_number in range(number_of_clusters):
            difference = data - centers[cluster_number]
            squared_distance[:, cluster_number] = np.sum(difference ** 2, axis=1)
        distance_to_center = squared_distance.min(axis=1)
        new_cluster_index = squared_distance.argmin(axis=1)

        for cluster_number in range(number_of_clusters):
            if not np.any(new_cluster_index == cluster_number):
                counts = np.bincount(new_cluster_index, minlength=number_of_clusters)
                eligible = counts[new_cluster_index] > 1
                eligible_distance = distance_to_center.copy()
                eligible_distance[~eligible] = -np.inf
                farthest_index = int(np.argmax(eligible_distance))
                if not np.isfinite(eligible_distance[farthest_index]):
                    farthest_index = rs.randint(number_of_points)
                new_cluster_index[farthest_index] = cluster_number
                distance_to_center[farthest_index] = 0

        new_centers = centers.copy()
        for cluster_number in range(number_of_clusters):
            members = data[new_cluster_index == cluster_number]
            if members.shape[0] > 0:
                new_centers[cluster_number] = np.mean(members, axis=0)

        center_change = float(np.max(np.sqrt(np.sum((new_centers - centers) ** 2, axis=1))))
        labels_unchanged = np.array_equal(new_cluster_index, cluster_index)
        centers = new_centers
        cluster_index = new_cluster_index
        if labels_unchanged or center_change <= tolerance:
            break
    return cluster_index


# %% ===== ALGORITHMS =========================================================

def run_psombest(population_size, maximum_iterations, lower_bound, upper_bound,
                 dimension, objective_function, parameters, rs):
    lower = expand_bound(lower_bound, dimension)
    upper = expand_bound(upper_bound, dimension)
    span = upper - lower

    unit_design = local_lhsdesign(population_size, dimension,
                                  parameters['lhs_trials'], rs)
    positions = lower + unit_design * span
    velocities = np.zeros((population_size, dimension))

    costs = np.full(population_size, np.inf)
    personal_best_positions = positions.copy()
    personal_best_costs = np.full(population_size, np.inf)
    best_score = np.inf
    best_position = np.zeros(dimension)

    for particle_index in range(population_size):
        costs_i = evaluate_objective(objective_function, positions[particle_index])
        costs[particle_index] = costs_i
        personal_best_costs[particle_index] = costs_i
        if costs_i < best_score:
            best_score = costs_i
            best_position = positions[particle_index].copy()

    velocity_max = parameters['vmax_factor'] * span
    velocity_min = -velocity_max
    inertia_weight = parameters['w']
    convergence_curve = np.full(maximum_iterations, np.nan)

    for iteration in range(maximum_iterations):
        cluster_index = local_kmeans(positions, parameters['number_of_clusters'],
                                     parameters['kmeans_max_iter'],
                                     parameters['kmeans_tolerance'], rs)
        cluster_leaders = np.zeros((parameters['number_of_clusters'], dimension))
        for cluster_number in range(parameters['number_of_clusters']):
            members = np.where(cluster_index == cluster_number)[0]
            if members.size == 0:
                cluster_leaders[cluster_number] = best_position
            elif parameters['scalar_mbest']:
                leader_scalar = positions[members[0], 0]
                cluster_leaders[cluster_number] = leader_scalar
            else:
                leader_particle = members[np.argmin(costs[members])]
                cluster_leaders[cluster_number] = positions[leader_particle]

        for particle_index in range(population_size):
            local_leader = cluster_leaders[cluster_index[particle_index]]
            candidate_velocity = \
                inertia_weight * velocities[particle_index] + \
                parameters['c1'] * rs.random_sample(dimension) * \
                    (personal_best_positions[particle_index] - positions[particle_index]) + \
                parameters['c2'] * rs.random_sample(dimension) * \
                    (best_position - positions[particle_index]) + \
                parameters['c3'] * rs.random_sample(dimension) * \
                    (local_leader - positions[particle_index])
            candidate_velocity = np.maximum(candidate_velocity, velocity_min)
            candidate_velocity = np.minimum(candidate_velocity, velocity_max)
            candidate_position = positions[particle_index] + candidate_velocity

            outside = (candidate_position < lower) | (candidate_position > upper)
            if parameters['boundary_mirror']:
                candidate_velocity[outside] = -candidate_velocity[outside]
            candidate_position = np.maximum(candidate_position, lower)
            candidate_position = np.minimum(candidate_position, upper)

            positions[particle_index] = candidate_position
            if parameters['store_velocity']:
                velocities[particle_index] = candidate_velocity
            else:
                velocities[particle_index] = 0.0

            cost_i = evaluate_objective(objective_function, candidate_position)
            costs[particle_index] = cost_i
            if cost_i < personal_best_costs[particle_index]:
                personal_best_costs[particle_index] = cost_i
                personal_best_positions[particle_index] = candidate_position.copy()
                if cost_i < best_score:
                    best_score = cost_i
                    best_position = candidate_position.copy()

        convergence_curve[iteration] = best_score
        inertia_weight *= parameters['wdamp']
    return best_score, best_position, convergence_curve


def _generic_pso_loop(positions, velocities, costs, personal_best_positions,
                      personal_best_costs, best_score, best_position,
                      lower, upper, span, parameters, objective_function,
                      maximum_iterations, rs, extra_velocity_terms=None,
                      adaptive_hook=None):
    velocity_max = parameters['vmax_factor'] * span
    velocity_min = -velocity_max
    inertia_weight = parameters['w']
    c1, c2 = parameters['c1'], parameters['c2']
    dimension = positions.shape[1]
    convergence_curve = np.full(maximum_iterations, np.nan)

    for iteration in range(maximum_iterations):
        if adaptive_hook is not None:
            inertia_weight, c1, c2, extra_state = \
                adaptive_hook(iteration, inertia_weight, c1, c2, best_score)
        else:
            extra_state = None

        previous_updated_velocity = np.zeros(dimension)
        for particle_index in range(positions.shape[0]):
            candidate_velocity = inertia_weight * velocities[particle_index]
            if extra_velocity_terms is not None:
                candidate_velocity = candidate_velocity + extra_velocity_terms(
                    particle_index, previous_updated_velocity, inertia_weight)
            candidate_velocity = candidate_velocity + \
                c1 * rs.random_sample(dimension) * \
                    (personal_best_positions[particle_index] - positions[particle_index]) + \
                c2 * rs.random_sample(dimension) * \
                    (best_position - positions[particle_index])

            candidate_velocity = np.maximum(candidate_velocity, velocity_min)
            candidate_velocity = np.minimum(candidate_velocity, velocity_max)
            candidate_position = positions[particle_index] + candidate_velocity

            outside = (candidate_position < lower) | (candidate_position > upper)
            if parameters['boundary_mirror']:
                candidate_velocity[outside] = -candidate_velocity[outside]
            candidate_position = np.maximum(candidate_position, lower)
            candidate_position = np.minimum(candidate_position, upper)

            positions[particle_index] = candidate_position
            velocities[particle_index] = candidate_velocity
            previous_updated_velocity = candidate_velocity
            cost_i = evaluate_objective(objective_function, candidate_position)

            if cost_i < personal_best_costs[particle_index]:
                personal_best_costs[particle_index] = cost_i
                personal_best_positions[particle_index] = candidate_position.copy()
                if cost_i < best_score:
                    best_score = cost_i
                    best_position = candidate_position.copy()

        convergence_curve[iteration] = best_score
        if adaptive_hook is None:
            inertia_weight *= parameters['wdamp']
    return best_score, best_position, convergence_curve


def run_classical_pso(population_size, maximum_iterations, lower_bound, upper_bound,
                      dimension, objective_function, parameters, rs):
    lower = expand_bound(lower_bound, dimension)
    upper = expand_bound(upper_bound, dimension)
    span = upper - lower
    positions = lower + rs.random_sample((population_size, dimension)) * span
    velocities = np.zeros((population_size, dimension))
    personal_best_positions = positions.copy()
    personal_best_costs = np.full(population_size, np.inf)
    best_score, best_position = np.inf, np.zeros(dimension)
    for particle_index in range(population_size):
        cost_i = evaluate_objective(objective_function, positions[particle_index])
        personal_best_costs[particle_index] = cost_i
        if cost_i < best_score:
            best_score = cost_i
            best_position = positions[particle_index].copy()
    return _generic_pso_loop(positions, velocities, None, personal_best_positions,
                             personal_best_costs, best_score, best_position,
                             lower, upper, span, parameters, objective_function,
                             maximum_iterations, rs)


def run_pso_awd(population_size, maximum_iterations, lower_bound, upper_bound,
                dimension, objective_function, parameters, rs):
    lower = expand_bound(lower_bound, dimension)
    upper = expand_bound(upper_bound, dimension)
    span = upper - lower
    positions = lower + rs.random_sample((population_size, dimension)) * span
    velocities = np.zeros((population_size, dimension))
    personal_best_positions = positions.copy()
    personal_best_costs = np.full(population_size, np.inf)
    best_score, best_position = np.inf, np.zeros(dimension)
    for particle_index in range(population_size):
        cost_i = evaluate_objective(objective_function, positions[particle_index])
        personal_best_costs[particle_index] = cost_i
        if cost_i < best_score:
            best_score = cost_i
            best_position = positions[particle_index].copy()
    initial_worst_score = max([0.0] + list(personal_best_costs))

    pending = {'values': None}

    def hook(iteration, inertia_weight, c1, c2, best_score_now):
        # Values used in THIS iteration: the initial ones for iteration 0, otherwise those computed at the end of
        # the previous iteration (MATLAB updates c1, c2 and inertia after the particle loop).
        if pending['values'] is not None:
            inertia_weight, c1, c2 = pending['values']
        if initial_worst_score == 0:
            evolutionary_state = np.inf if best_score_now < 0 else 0.0
        else:
            evolutionary_state = (initial_worst_score - best_score_now) / initial_worst_score
        remaining_ratio = (maximum_iterations - (iteration + 1)) / maximum_iterations
        c1_next = (parameters['c1_initial'] - parameters['c1_final']) * remaining_ratio \
            + parameters['c1_final']
        c2_next = (parameters['c2_initial'] - parameters['c2_final']) * remaining_ratio \
            + parameters['c2_final']
        with np.errstate(over='ignore'):
            w_next = 1 - parameters['adaptive_a'] / \
                (1 + np.exp(parameters['adaptive_b'] * evolutionary_state))
        pending['values'] = (float(w_next), c1_next, c2_next)
        return inertia_weight, c1, c2, evolutionary_state

    def delay_term(particle_index, previous_updated_velocity, inertia_weight):
        if particle_index == 0:
            return np.zeros(previous_updated_velocity.shape)
        return (1 - inertia_weight) * previous_updated_velocity

    return _generic_pso_loop(positions, velocities, None, personal_best_positions,
                             personal_best_costs, best_score, best_position,
                             lower, upper, span, parameters, objective_function,
                             maximum_iterations, rs,
                             extra_velocity_terms=delay_term, adaptive_hook=hook)


def run_psoea(population_size, maximum_iterations, lower_bound, upper_bound,
              dimension, objective_function, parameters, rs):
    lower = expand_bound(lower_bound, dimension)
    upper = expand_bound(upper_bound, dimension)
    span = upper - lower
    original_positions = lower + rs.random_sample((population_size, dimension)) * span
    original_costs = np.full(population_size, np.inf)
    best_score, best_position = np.inf, np.zeros(dimension)
    for particle_index in range(population_size):
        original_costs[particle_index] = evaluate_objective(
            objective_function, original_positions[particle_index])
        if original_costs[particle_index] < best_score:
            best_score = original_costs[particle_index]
            best_position = original_positions[particle_index].copy()

    parent_probability = rank_selection_probabilities(original_costs, 1.0)
    positions = np.zeros((population_size, dimension))
    costs = np.inf * np.ones(population_size)
    for particle_index in range(population_size):
        first_parent = roulette_select(parent_probability, rs)
        second_parent = roulette_select(parent_probability, rs)
        while second_parent == first_parent and population_size > 1:
            second_parent = roulette_select(parent_probability, rs)
        crossover_mask = rs.random_sample(dimension) < 0.5
        child = original_positions[first_parent].copy()
        child[~crossover_mask] = original_positions[second_parent][~crossover_mask]
        positions[particle_index] = child
        costs[particle_index] = evaluate_objective(objective_function, child)
        if costs[particle_index] < best_score:
            best_score = costs[particle_index]
            best_position = child.copy()

    velocities = np.zeros((population_size, dimension))
    return _generic_pso_loop(positions, velocities, costs, positions.copy(),
                             costs.copy(), best_score, best_position,
                             lower, upper, span, parameters, objective_function,
                             maximum_iterations, rs)


def run_real_coded_ga(population_size, maximum_iterations, lower_bound, upper_bound,
                      dimension, objective_function, parameters, rs):
    lower = expand_bound(lower_bound, dimension)
    upper = expand_bound(upper_bound, dimension)
    span = upper - lower
    positions = lower + rs.random_sample((population_size, dimension)) * span
    costs = np.full(population_size, np.inf)
    for individual_index in range(population_size):
        costs[individual_index] = evaluate_objective(
            objective_function, positions[individual_index])

    best_index = int(np.argmin(costs))
    best_score = float(costs[best_index])
    best_position = positions[best_index].copy()
    convergence_curve = np.full(maximum_iterations, np.nan)
    elite_count = max(2, int(math.floor(parameters['elite_fraction'] * population_size + 0.5)))
    elite_count = min(elite_count, population_size - 1)

    for iteration in range(maximum_iterations):
        sorted_index = np.argsort(costs, kind='stable')
        sorted_costs = costs[sorted_index]
        sorted_positions = positions[sorted_index]
        elites = sorted_positions[:elite_count].copy()
        elite_costs = sorted_costs[:elite_count].copy()

        selection_probability = rank_selection_probabilities(
            costs, parameters['selection_pressure'])
        children_count = population_size - elite_count
        children = np.zeros((children_count, dimension))
        child_costs = np.full(children_count, np.inf)

        for child_index in range(children_count):
            first_parent = roulette_select(selection_probability, rs)
            second_parent = roulette_select(selection_probability, rs)
            while second_parent == first_parent and population_size > 1:
                second_parent = roulette_select(selection_probability, rs)

            if rs.random_sample() <= parameters['crossover_probability']:
                alpha = rs.random_sample(dimension)
                child = alpha * positions[first_parent] \
                    + (1 - alpha) * positions[second_parent]
            else:
                child = positions[first_parent].copy()

            mutation_mask = rs.random_sample(dimension) < parameters['mutation_probability']
            if np.any(mutation_mask):
                cooling = 1 - 0.9 * iteration / max(1, maximum_iterations - 1)
                child[mutation_mask] += parameters['mutation_scale'] * cooling \
                    * span[mutation_mask] \
                    * rs.standard_normal(int(np.sum(mutation_mask)))

            child = np.maximum(child, lower)
            child = np.minimum(child, upper)
            children[child_index] = child
            child_costs[child_index] = evaluate_objective(objective_function, child)

        positions = np.vstack((elites, children))
        costs = np.concatenate((elite_costs, child_costs))
        generation_best = float(np.min(costs))
        generation_index = int(np.argmin(costs))
        if generation_best < best_score:
            best_score = generation_best
            best_position = positions[generation_index].copy()
        convergence_curve[iteration] = best_score
    return best_score, best_position, convergence_curve


# %% ===== PAPER REFERENCE TABLES (Tables 4-12, transcribed) ==================

PAPER_LABELS = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10',
                'F11', 'F12', 'F13', 'F14', 'F15', 'F16',
                'F17_2D', 'F17_30D', 'F17_100D', 'F18_2D', 'F18_30D', 'F18_100D']
PAPER_ALGORITHMS = ['PSO', 'PSO-AWDV', 'PSOEA', 'GA', 'PSOMbest']

PAPER_BEST = np.array([
    [4.0218e-98, 6.06252, 5.1859e-100, 0.00180, 6.603e-236],
    [7.50504e-10, 1.98923, 1.90473e-9, 1.28608, 4.11752e-27],
    [0.66667, 1.63586, 0.66667, 0.66671, 0.18241],
    [1.9856e-171, 2.6034e-18, 3.1923e-171, 1.63753e-17, 0],
    [0, 0, 0, 0, 0],
    [0, 4.75199e-68, 0, 3.2249e-239, 0],
    [-1, -0.99999, -1, -1, -1],
    [0, 0, 0, 0, 0],
    [1.15463e-14, 0.00712, 7.99361e-15, 0.00059, 4.44089e-15],
    [2.175409076, 8.27561, 0.54385, 2.86289, 1.49976e-32],
    [3.96683e-13, 10.34546028, 3.6926e-13, 26.97701, 0],
    [-3421.831, -3001.808, -3407.695, -3455.488, -3916.166],
    [0.0012633, 1.54593591, 0.00157898, 0.01615, 0.000593],
    [21.289979, 30.1634412, 20.44469693, 2.559060, 0.112073],
    [0.0308622, 1.50390775, 0.035410354, 2.665462, 0.020417],
    [7.55207e-5, 0.55838062, 0.00012165, 1.72972, 2.70531e-5],
    [0, 6.22777e-24, 0, 8.56965e-19, 0],
    [0.593594803, 32.90125955, 0.11054394, 1.45339704, 0],
    [126.9879929, 3393.706633, 71.6360264, 604.1686149, 0],
    [0, 5.19977e-18, 0, 9.88369e-11, 0],
    [18.96820774, 30.37178684, 8.15080702, 26.5396838, 0],
    [81.50860794, 908.6016601, 91.2217011, 95.98033087, 0]])

PAPER_MAE = np.array([
    [4.296e-92, 76.98422, 3.89785e-92, 13.91806, 2.93273e-8],
    [6.66621e-8, 3.75342, 6.79923e-8, 3.07840, 2.31505e-6],
    [0.66667, 7.93845, 0.66667, 0.85748, 0.71922],
    [5.1162e-145, 5.49909e-12, 3.9635e-150, 3.50266e-11, 0],
    [0, 0, 0, 0, 0],
    [0, 6.56244e-44, 0, 2.03137e-18, 0],
    [0.02, 0.02014, 0.02, 0.02893, 0.02255],
    [0, 0, 0, 0, 0],
    [2.08544e-14, 0.02165, 2.00018e-14, 0.00087, 0.00727],
    [10.71004, 18.72230, 11.07623, 3.83042, 3.70392e-9],
    [0.12136, 19.94082, 0.09148, 36.62218, 5.29789e-11],
    [660.20023, 1114.70001, 642.65524, 697.45748, 10.24644],
    [0.00121559, 6.130958766, 0.001373176, 3.336978416, 0.003408802],
    [19.0844968, 19.10981999, 17.26557704, 3.084261342, 0.252170126],
    [0.428796032, 1.009873388, 0.038415831, 1.553110144, 0.019118491],
    [0.005105199, 0.417872321, 0.005398832, 1.380169938, 0.07289616],
    [0, 8.59134e-11, 0, 0.000108437, 0],
    [13.04416821, 47.88386943, 12.8030527, 0.970510359, 5.8795e-6],
    [63.53987079, 2551.497442, 65.67604382, 366.677693, 32.30026951],
    [0, 2.77658e-12, 0, 0.000847392, 0],
    [11.40941039, 25.36109884, 7.876864475, 10.92232852, 0.489048421],
    [42.37149273, 453.8217618, 44.09560408, 542.0518502, 39.11061438]])

PAPER_SD = np.array([
    [2.45705e-91, 85.49406, 1.88719e-91, 30.9640, 2.07375e-7],
    [1.15117e-7, 1.14183, 1.43314e-7, 1.45745, 1.18901e-5],
    [2.73792e-16, 5.73753, 2.48253e-16, 0.70548, 0.90817],
    [3.6008e-144, 2.18341e-11, 2.5084e-149, 1.34234e-10, 0],
    [0, 0, 0, 0, 0],
    [0, 4.6403e-43, 0, 1.06667e-17, 0],
    [0, 0.00012, 0, 0.02474, 0.01262],
    [0, 0, 0, 0, 0],
    [6.80507e-15, 0.01172, 4.91493e-15, 0.00012, 0.05135],
    [3.37357, 5.40355, 4.11081, 0.44650, 1.79588e-8],
    [0.18319, 4.63812, 0.11219, 5.94389, 2.15226e-10],
    [53.81486, 74.99359, 55.46988, 80.36825, 46.62574],
    [0.001280966, 13.12464091, 0.002050239, 10.93430651, 0.023208555],
    [19.0844968, 19.10981999, 17.26557704, 3.084261342, 0.252170126],
    [1.462144248, 0.654539306, 0.198708018, 0.718214117, 0.011711702],
    [0.375443752, 0.167354619, 0.012615807, 0.887211018, 0.013269013],
    [0, 9.519e-10, 0, 0.000836554, 0],
    [24.57905489, 62.50468094, 23.70867439, 0.520100848, 6.57347e-5],
    [np.nan, 2983.610568, 60.31844995, 296.9643339, 49.41550314],
    [0, 2.63792e-11, 0, 0.004507696, 0],
    [20.23738586, 30.91413765, 3.324209896, 0.696991482, 2.986028171],
    [29.5871517, 214.3659764, 23.84112572, 5472.462358, 0.745742797]])

PAPER_WORST = np.array([
    [1.73539e-90, 378.5818023, 1.33379e-90, 151.7558672, 1.46636e-6],
    [5.57142e-7, 6.108257572, 9.66256e-7, 6.788905131, 8.09531e-5],
    [0.666666667, 24.76677769, 0.666666667, 4.413501022, 6.68136002],
    [2.5463e-143, 1.12244e-10, 1.7744e-148, 8.94607e-10, 0],
    [0, 0, 0, 0, 0],
    [0, 3.28119e-42, 0, 7.05859e-17, 0],
    [-1, -0.9993919, -1, -0.9362453, -0.9362453],
    [0, 0, 0, 0, 0],
    [3.9968e-14, 0.058225798, 3.9968e-14, 0.001114711, 0.363106893],
    [18.63381944, 38.17312859, 19.54244632, 5.111238465, 9.07663e-8],
    [0.793765033, 30.48733926, 0.468569005, 50.76553979, 1.41017e-9],
    [-3195.64389, -2624.39726, -3153.23374, -3067.8354, -3680.43650],
    [0.099155026, 49.17726443, 0.010186423, 44.63780751, 0.005507762],
    [85.92131535, 69.87803908, 69.66203281, 14.63966182, 2.299103273],
    [3.724887065, 4.039111779, 0.937067222, 5.063922483, 0.076904616],
    [0.07289616, 1.290203231, 0.051755699, 5.394507826, 0.046925553],
    [0, 4.25887e-9, 0, 0.003385736, 0],
    [74.82296694, 314.860825, 76.90164891, 3.637892802, 0.000293975],
    [229.168419, 13570.16771, 313.5026471, 1685.645411, 137.0706187],
    [0, 1.18448e-10, 0, 0.017839829, 0],
    [76.71101223, 146.907062, 25.53414442, 29.35884125, 8.150807023],
    [195.5733344, 1637.580727, 148.9283369, 24586.70819, 98.98273584]])


def paper_algorithm_reference(function_name, algorithm_name):
    fi = PAPER_LABELS.index(function_name.upper()) \
        if function_name.upper() in PAPER_LABELS else -1
    ai = PAPER_ALGORITHMS.index(algorithm_name) \
        if algorithm_name in PAPER_ALGORITHMS else -1
    available = fi >= 0 and ai >= 0
    reference = {'best': np.nan, 'mae': np.nan, 'sd': np.nan, 'worst': np.nan}
    if available:
        reference = {'best': PAPER_BEST[fi, ai], 'mae': PAPER_MAE[fi, ai],
                     'sd': PAPER_SD[fi, ai], 'worst': PAPER_WORST[fi, ai]}
    return available, reference


# %% ===== WORKER (one run of one algorithm on one case) ======================

ALGO_KEYS = {'PSO': 'call_PSO', 'PSO-AWDV': 'call_PSO_AWDV', 'PSOEA': 'call_PSOEA',
             'GA': 'call_GA', 'PSOMbest': 'call_PSOMbest_reproduction'}


def run_single_run(payload):
    (function_name, algorithm_name, run_seed, number_of_particles,
     maximum_iterations, noise_sigma, params_all) = payload

    rs = np.random.RandomState(run_seed)
    lower_bound, upper_bound, dimension, true_optimum, paper_optimum, descriptive_name = \
        psombest_benchmark_info(function_name)
    base_name = re.match(r'^F\d+', function_name.upper()).group(0)

    def objective_function(x):
        return psombest_objective(x, base_name, noise_sigma, rs)

    if algorithm_name == 'PSO':
        solver, params = run_classical_pso, params_all['param_pso']
    elif algorithm_name == 'PSO-AWDV':
        solver, params = run_pso_awd, params_all['param_awd']
    elif algorithm_name == 'PSOEA':
        solver, params = run_psoea, params_all['param_psoea']
    elif algorithm_name == 'GA':
        solver, params = run_real_coded_ga, params_all['param_ga']
    else:
        solver, params = run_psombest, params_all['param_reproduction']

    start = time.perf_counter()
    best_score, best_position, convergence_curve = solver(
        number_of_particles, maximum_iterations, lower_bound, upper_bound,
        dimension, objective_function, params, rs)
    elapsed_time = time.perf_counter() - start

    if (not np.isscalar(best_score)) or (not np.isfinite(best_score)):
        raise ValueError('%s returned an invalid best score on %s.'
                         % (algorithm_name, function_name))
    if best_position.size != dimension:
        raise ValueError('%s returned a best position with the wrong dimension.'
                         % algorithm_name)
    if convergence_curve.size != maximum_iterations:
        raise ValueError('%s returned a convergence curve with the wrong length.'
                         % algorithm_name)
    return {'function_name': function_name, 'algorithm_name': algorithm_name,
            'run_seed': run_seed, 'best_score': float(best_score),
            'elapsed_time': elapsed_time,
            'convergence_curve': convergence_curve.astype(float),
            'best_position': best_position.astype(float).ravel(),
            'dimension': dimension, 'true_optimum': true_optimum,
            'paper_optimum': paper_optimum, 'descriptive_name': descriptive_name}


# %% ===== LOGGER =============================================================

class Logger:
    def __init__(self, path):
        self.file = open(path, 'w', encoding='utf-8')

    def __call__(self, text=''):
        print(text)
        self.file.write(text + '\n')
        self.file.flush()

    def close(self):
        self.file.close()


# %% ===== MAIN DRIVER (mirror of MATLAB sections 1-9) ========================

COLOR_MAP = [(0.9290, 0.6940, 0.1250), (0.8500, 0.3250, 0.0980),
             (0.0000, 0.4470, 0.7410), (0.0000, 0.0000, 0.0000),
             (0.4660, 0.6740, 0.1880), (0.4940, 0.1840, 0.5560)]
LINE_STYLES = ['-', '-', '-', '-', '-', '--']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='full', choices=['debug', 'sample', 'full'])
    parser.add_argument('--workers', type=int, default=min(4, cpu_count()))
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(COMPOSITES_DIR, exist_ok=True)
    for stale in (EXCEL_FILENAME, NPZ_FILENAME, LOG_FILENAME):
        if os.path.exists(stale):
            os.remove(stale)
    log = Logger(LOG_FILENAME)

    log('=' * 70)
    log('PSOMbest reproduction started at %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
    log('Master file: %s' % MASTER_FILE)
    log('Results: %s' % RESULTS_ROOT)
    log('=' * 70)
    log()

    # ---- 1. CENTRAL CONFIGURATION ------------------------------------------
    experiment_mode = args.mode
    configurations = [('PSO', 'call_PSO'), ('PSO-AWDV', 'call_PSO_AWDV'),
                      ('PSOEA', 'call_PSOEA'), ('GA', 'call_GA'),
                      ('PSOMbest', 'call_PSOMbest_reproduction')]

    if experiment_mode == 'debug':
        benchmark_names = ['F8']
        number_of_particles, maximum_iterations, number_of_runs = 30, 100, 3
    elif experiment_mode == 'sample':
        benchmark_names = ['F1', 'F3', 'F8', 'F10', 'F12', 'F13', 'F16',
                           'F17_30D', 'F18_30D']
        number_of_particles, maximum_iterations, number_of_runs = 60, 300, 10
    else:
        benchmark_names = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
                           'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16',
                           'F17_2D', 'F17_30D', 'F17_100D',
                           'F18_2D', 'F18_30D', 'F18_100D']
        number_of_particles, maximum_iterations, number_of_runs = 150, 1000, 50

    master_seed = 0
    noise_sigma = 0.01
    save_individual_figures = True
    generate_composite_plots = True

    # ---- 2. ALGORITHM PARAMETERS -------------------------------------------
    param_reproduction = {'w': 1.0, 'wdamp': 0.99, 'c1': 0.5, 'c2': 1.5, 'c3': 1.0,
                          'number_of_clusters': 5, 'vmax_factor': 1.0,
                          'kmeans_max_iter': 100, 'kmeans_tolerance': 1e-10,
                          'lhs_trials': 5, 'scalar_mbest': True,
                          'store_velocity': False, 'boundary_mirror': False}
    param_corrected = dict(param_reproduction, scalar_mbest=False, store_velocity=True)
    param_table3 = dict(param_corrected, c1=1.5, c2=2.0)
    param_pso = {'w': 1.0, 'wdamp': 0.99, 'c1': 1.5, 'c2': 2.0,
                 'vmax_factor': 0.1, 'boundary_mirror': True}
    param_awd = dict(param_pso, c1_initial=2.5, c1_final=0.5,
                     c2_initial=0.5, c2_final=2.5,
                     adaptive_a=0.9, adaptive_b=0.5)
    param_psoea = dict(param_pso, crossover_probability=1.0,
                       mutation_probability=0.02, mutation_scale=0.10,
                       elite_fraction=0.10, stagnation_limit=20)
    param_ga = {'crossover_probability': 1.0, 'mutation_probability': 0.02,
                'mutation_scale': 0.10, 'elite_fraction': 0.05,
                'selection_pressure': 1.0}
    params_all = {'param_reproduction': param_reproduction,
                  'param_corrected': param_corrected, 'param_table3': param_table3,
                  'param_pso': param_pso, 'param_awd': param_awd,
                  'param_psoea': param_psoea, 'param_ga': param_ga}

    number_of_benchmarks = len(benchmark_names)
    number_of_algorithms = len(configurations)
    algorithm_names = [c[0] for c in configurations]

    all_best_scores = np.full((number_of_benchmarks, number_of_algorithms,
                               number_of_runs), np.nan)
    all_run_times = np.full_like(all_best_scores, np.nan)
    all_convergence = [[None] * number_of_runs
                       for _ in range(number_of_benchmarks * number_of_algorithms)]
    all_best_positions = [[None] * number_of_runs
                          for _ in range(number_of_benchmarks * number_of_algorithms)]

    problem_names, problem_dimensions = [], []
    true_optima, paper_optima = [], []

    log('Experiment mode: %s' % experiment_mode)
    log('Algorithms: %d | Cases: %d | Runs: %d | Particles: %d | Iterations: %d'
        % (number_of_algorithms, number_of_benchmarks, number_of_runs,
           number_of_particles, maximum_iterations))
    log('Workers: %d (CPU processes; runs are seed-isolated so scheduling '
        'cannot change results)' % args.workers)
    log('GPU: none available in Python stack (CPU execution)')
    log()

    # ---- 4. MAIN EXPERIMENT LOOP -------------------------------------------
    payloads = []
    for function_index, function_name in enumerate(benchmark_names):
        lb, ub, dim, topt, popt, desc = psombest_benchmark_info(function_name)
        problem_names.append(desc)
        problem_dimensions.append(dim)
        true_optima.append(topt)
        paper_optima.append(popt)
        for algorithm_index, (algorithm_name, _) in enumerate(configurations):
            for run_index in range(number_of_runs):
                payloads.append((function_name, algorithm_name,
                                 master_seed + run_index + 1,
                                 number_of_particles, maximum_iterations,
                                 noise_sigma, params_all))

    started_at = time.time()
    with Pool(args.workers) as pool:
        for payload, result in zip(payloads, pool.imap(run_single_run, payloads)):
            function_index = benchmark_names.index(result['function_name'])
            algorithm_index = algorithm_names.index(result['algorithm_name'])
            run_index = result['run_seed'] - master_seed - 1
            flat = function_index * number_of_algorithms + algorithm_index
            all_best_scores[function_index, algorithm_index, run_index] = \
                result['best_score']
            all_run_times[function_index, algorithm_index, run_index] = \
                result['elapsed_time']
            all_convergence[flat][run_index] = result['convergence_curve']
            all_best_positions[flat][run_index] = result['best_position']
            if function_index != getattr(main, '_last_case', -1):
                log('=' * 78)
                log('Case %d/%d: %s - %s (%d dimensions)'
                    % (function_index + 1, number_of_benchmarks,
                       result['function_name'], result['descriptive_name'],
                       result['dimension']))
                log('-' * 78)
                log('  Running %s' % result['algorithm_name'])
                main._last_case = function_index
            elif algorithm_index != getattr(main, '_last_algo', -1):
                log('  Running %s' % result['algorithm_name'])
                main._last_algo = algorithm_index
            log('    Run %2d/%2d | best = %.12g | %.3f s'
                % (run_index + 1, number_of_runs, result['best_score'],
                   result['elapsed_time']))
    main._last_case = -1
    main._last_algo = -1
    log('Total experiment wall time: %.1f s' % (time.time() - started_at))

    # ---- 4a/5/6. STATISTICS -------------------------------------------------
    all_mean = np.nanmean(all_best_scores, axis=2)
    all_sd = np.nanstd(all_best_scores, axis=2, ddof=0)
    all_mae = np.nanmean(np.abs(all_best_scores
                                - np.array(paper_optima)[:, None, None]), axis=2)
    all_best = np.nanmin(all_best_scores, axis=2)
    all_worst = np.nanmax(all_best_scores, axis=2)
    all_mean_time = np.nanmean(all_run_times, axis=2)
    all_rank = np.full((number_of_benchmarks, number_of_algorithms), np.nan)

    for function_index in range(number_of_benchmarks):
        rank_matrix = np.column_stack((all_mean[function_index],
                                       all_sd[function_index],
                                       all_best[function_index]))
        order = sorted(range(number_of_algorithms),
                       key=lambda a: (rank_matrix[a, 0], rank_matrix[a, 1],
                                      rank_matrix[a, 2]))
        for rank_position, algo_index in enumerate(order):
            all_rank[function_index, algo_index] = rank_position + 1

    # ---- 4b. PER-FUNCTION CONVERGENCE FIGURES -------------------------------
    if save_individual_figures:
        for function_index, function_name in enumerate(benchmark_names):
            fig, ax = plt.subplots(figsize=(8, 5))
            for algorithm_index in range(number_of_algorithms):
                flat = function_index * number_of_algorithms + algorithm_index
                curve_matrix = np.vstack(all_convergence[flat])
                mean_curve = curve_matrix.mean(axis=0)
                color = COLOR_MAP[(algorithm_index - 1) % len(COLOR_MAP)]
                style = LINE_STYLES[(algorithm_index - 1) % len(LINE_STYLES)]
                if np.all(mean_curve > 0) and np.all(np.isfinite(mean_curve)):
                    ax.semilogy(np.arange(1, maximum_iterations + 1), mean_curve,
                                linewidth=1.8, color=color, linestyle=style,
                                label=algorithm_names[algorithm_index])
                else:
                    ax.plot(np.arange(1, maximum_iterations + 1), mean_curve,
                            linewidth=1.8, color=color, linestyle=style,
                            label=algorithm_names[algorithm_index])
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Mean best objective value')
            ax.set_title('%s - %s' % (function_name, problem_names[function_index]),
                         fontweight='normal')
            ax.grid(True)
            ax.legend(loc='best')
            fig.savefig(os.path.join(FIGURES_DIR, function_name + '.png'),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

    # ---- 4c/5/6. EXCEL WORKBOOK (openpyxl direct; pandas is policy-blocked) --
    def _cell(v):
        if isinstance(v, (float, np.floating)) and not np.isfinite(v):
            return None
        return v

    workbook = Workbook()
    workbook.remove(workbook.active)

    for function_index, function_name in enumerate(benchmark_names):
        sheet = workbook.create_sheet(function_name[:31])
        sheet.append(['%s - %s' % (function_name, problem_names[function_index])])
        sheet.append(['Dimension', int(problem_dimensions[function_index])])
        sheet.append(['True mathematical optimum', true_optima[function_index]])
        sheet.append(['Paper benchmark value', paper_optima[function_index]])
        sheet.append([])
        sheet.append(['Raw final best scores'])
        sheet.append(['Algorithm'] + ['Run %d' % (r + 1) for r in range(number_of_runs)])
        for algorithm_index in range(number_of_algorithms):
            sheet.append([algorithm_names[algorithm_index]]
                         + [_cell(all_best_scores[function_index, algorithm_index, r])
                            for r in range(number_of_runs)])
        sheet.append([])
        sheet.append(['Summary statistics'])
        sheet.append(['Metric'] + algorithm_names)
        for metric_name, table in [('Mean', all_mean), ('Population SD', all_sd),
                                   ('MAE', all_mae), ('Best', all_best),
                                   ('Worst', all_worst),
                                   ('Mean time (s)', all_mean_time),
                                   ('Rank', all_rank)]:
            sheet.append([metric_name] + [_cell(table[function_index, a])
                                          for a in range(number_of_algorithms)])

    journal_sheet = workbook.create_sheet('Journal_Summary')
    journal_sheet.append(['Case', 'Function', 'Dimension', 'Algorithm', 'Mean',
                          'Population SD', 'MAE', 'Best', 'Worst',
                          'Mean time (s)', 'Rank'])
    for function_index in range(number_of_benchmarks):
        for algorithm_index in range(number_of_algorithms):
            journal_sheet.append([
                benchmark_names[function_index], problem_names[function_index],
                int(problem_dimensions[function_index]),
                algorithm_names[algorithm_index],
                _cell(all_mean[function_index, algorithm_index]),
                _cell(all_sd[function_index, algorithm_index]),
                _cell(all_mae[function_index, algorithm_index]),
                _cell(all_best[function_index, algorithm_index]),
                _cell(all_worst[function_index, algorithm_index]),
                _cell(all_mean_time[function_index, algorithm_index]),
                _cell(all_rank[function_index, algorithm_index])])

    match_sheet = workbook.create_sheet('Paper_Match')
    match_sheet.append(['Case', 'Algorithm', 'Paper Best', 'Reproduced Best',
                        'Absolute Best Difference', 'Relative Best Difference',
                        'Paper MAE', 'Reproduced MAE', 'Absolute MAE Difference',
                        'Relative MAE Difference', 'Paper SD', 'Reproduced SD',
                        'Absolute SD Difference', 'Relative SD Difference',
                        'Paper Worst', 'Reproduced Worst',
                        'Absolute Worst Difference', 'Relative Worst Difference',
                        'Reference available'])
    for function_index, function_name in enumerate(benchmark_names):
        for algorithm_index, algorithm_name in enumerate(algorithm_names):
            available, ref = paper_algorithm_reference(function_name, algorithm_name)
            row = [function_name, algorithm_name] + [None] * 16 + [bool(available)]
            if available:
                reproduced = [all_best[function_index, algorithm_index],
                              all_mae[function_index, algorithm_index],
                              all_sd[function_index, algorithm_index],
                              all_worst[function_index, algorithm_index]]
                published = [ref['best'], ref['mae'], ref['sd'], ref['worst']]
                values = []
                for rep, pub in zip(reproduced, published):
                    values.extend([_cell(pub), _cell(rep), _cell(abs(rep - pub)),
                                   _cell(safe_relative_difference(rep, pub))])
                row[2:18] = values
            match_sheet.append(row)

    published_sheet = workbook.create_sheet('Published_Optimum')
    reproduced_sheet = workbook.create_sheet('Reproduced_Optimum')
    optimum_headers = ['No', 'Case', 'Function', 'Benchmark value'] + algorithm_names
    published_sheet.append(optimum_headers)
    reproduced_sheet.append(optimum_headers)
    for function_index, function_name in enumerate(benchmark_names):
        case_number = int(re.match(r'F(\d+)', function_name).group(1))
        common = [case_number, function_name, problem_names[function_index],
                  paper_optima[function_index]]
        published_row, reproduced_row = list(common), list(common)
        for algorithm_index, algorithm_name in enumerate(algorithm_names):
            available, ref = paper_algorithm_reference(function_name, algorithm_name)
            published_row.append(_cell(ref['best']) if available else None)
            reproduced_row.append(_cell(all_best[function_index, algorithm_index]))
        published_sheet.append(published_row)
        reproduced_sheet.append(reproduced_row)

    published_computation_time = [84.52, 91.86, 93.32, 67.74, 150.29]
    time_sheet = workbook.create_sheet('Computation_Time')
    time_sheet.append(['Algorithm', 'Published average time (s)',
                       'Reproduced average time (s)', 'Absolute difference (s)',
                       'Relative difference', 'Timing note'])
    for algorithm_index, algorithm_name in enumerate(algorithm_names):
        reproduced_time = float(np.mean(all_mean_time[:, algorithm_index]))
        reference_index = PAPER_ALGORITHMS.index(algorithm_name) \
            if algorithm_name in PAPER_ALGORITHMS else None
        published_time = published_computation_time[reference_index] \
            if reference_index is not None else None
        time_sheet.append([algorithm_name, published_time, _cell(reproduced_time),
                           abs(reproduced_time - published_time),
                           safe_relative_difference(reproduced_time, published_time),
                           'Hardware-dependent; compare cautiously'])

    workbook.save(EXCEL_FILENAME)

    # ---- 7. COMPOSITE CONVERGENCE PLOTS -------------------------------------
    if generate_composite_plots:
        group_size = 6
        number_of_groups = int(math.ceil(number_of_benchmarks / group_size))
        for group_index in range(number_of_groups):
            first_index = group_index * group_size
            last_index = min(first_index + group_size, number_of_benchmarks)
            fig, axes = plt.subplots(2, 3, figsize=(14, 7.6))
            for local_index, function_index in enumerate(range(first_index, last_index)):
                ax = axes[local_index // 3][local_index % 3]
                for algorithm_index in range(number_of_algorithms):
                    flat = function_index * number_of_algorithms + algorithm_index
                    mean_curve = np.vstack(all_convergence[flat]).mean(axis=0)
                    color = COLOR_MAP[(algorithm_index - 1) % len(COLOR_MAP)]
                    style = LINE_STYLES[(algorithm_index - 1) % len(LINE_STYLES)]
                    if np.all(mean_curve > 0) and np.all(np.isfinite(mean_curve)):
                        ax.semilogy(np.arange(1, maximum_iterations + 1), mean_curve,
                                    linewidth=1.2, color=color, linestyle=style,
                                    label=algorithm_names[algorithm_index])
                    else:
                        ax.plot(np.arange(1, maximum_iterations + 1), mean_curve,
                                linewidth=1.2, color=color, linestyle=style,
                                label=algorithm_names[algorithm_index])
                ax.set_title(benchmark_names[function_index], fontweight='normal')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Mean best')
                ax.grid(True)
            handles, labels = axes[0][0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=len(labels),
                       frameon=False)
            fig.tight_layout(rect=[0, 0.05, 1, 1])
            composite_name = 'Composite_%02d_%s_to_%s' % (
                group_index + 1, benchmark_names[first_index],
                benchmark_names[last_index - 1])
            fig.savefig(os.path.join(COMPOSITES_DIR, composite_name + '.png'),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

    # ---- 8. SAVE RAW RESULTS -------------------------------------------------
    curves_obj = np.empty(len(all_convergence), dtype=object)
    positions_obj = np.empty(len(all_best_positions), dtype=object)
    for block_index in range(len(all_convergence)):
        curves_obj[block_index] = np.vstack(all_convergence[block_index])
        positions_obj[block_index] = np.vstack(all_best_positions[block_index])
    np.savez_compressed(
        NPZ_FILENAME,
        study_name=STUDY_NAME, experiment_mode=experiment_mode,
        benchmark_names=np.array(benchmark_names),
        problem_names=np.array(problem_names),
        problem_dimensions=np.array(problem_dimensions),
        true_optima=np.array(true_optima), paper_optima=np.array(paper_optima),
        number_of_particles=number_of_particles,
        maximum_iterations=maximum_iterations, number_of_runs=number_of_runs,
        master_seed=master_seed, noise_sigma=noise_sigma,
        all_best_scores=all_best_scores, all_run_times=all_run_times,
        all_convergence=curves_obj,
        all_best_positions=positions_obj,
        all_mean=all_mean, all_sd=all_sd, all_mae=all_mae, all_best=all_best,
        all_worst=all_worst, all_mean_time=all_mean_time, all_rank=all_rank)

    # ---- 9. FINAL CONSOLE SUMMARY --------------------------------------------
    log()
    log('=' * 96)
    log('FINAL SUMMARY')
    log('=' * 96)
    header = '%-11s' % 'Case'
    for algorithm_name in algorithm_names:
        header += ' | %-24s' % algorithm_name
    log(header)
    log('-' * 96)
    for function_index, function_name in enumerate(benchmark_names):
        row = '%-11s' % function_name
        for algorithm_index in range(number_of_algorithms):
            row += ' | %-24.8g' % all_best[function_index, algorithm_index]
        log(row)
    log()
    log('Excel results: %s' % EXCEL_FILENAME)
    log('Raw results:   %s' % NPZ_FILENAME)
    log('Figures:       %s' % FIGURES_DIR)
    log('Completed at %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
    log.close()


if __name__ == '__main__':
    main()
