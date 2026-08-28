import logging
import time
import concurrent.futures
from functools import partial
from backtest import fetch_data, run_backtest
import multiprocessing

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

import math

def poisson_deviance(y: float, mu: float) -> float:
    mu = max(1e-4, mu)
    y = max(0.0, float(y))
    if y == 0:
        return 2 * mu
    return 2 * (y * math.log(y / mu) - (y - mu))

def evaluate_combination(weights, df_gw, df_players, df_teams, df_fixtures):
    """Worker function for multiprocessing."""
    mean_dev = run_backtest(weights=weights, df_gw=df_gw, df_players=df_players, df_teams=df_teams, df_fixtures=df_fixtures)
    return weights, mean_dev

def run_grid_search():
    logger.info("==========================================")
    logger.info("   FPL ENSEMBLE WEIGHTS GRID SEARCH")
    logger.info("==========================================")
    
    logger.info("Pre-fetching Vaastav historical data for the grid search...")
    try:
        df_gw, df_players, df_teams, df_fixtures = fetch_data()
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return

    logger.info("Data loaded. Generating combinations...\n")

    # Generate parameter combinations for Kalman filter
    # Format: (process_variance, measurement_variance)
    combinations = []
    for pv in [0.01, 0.05, 0.10, 0.15]:
        for mv in [0.1, 0.2, 0.3, 0.4, 0.5]:
            combinations.append((pv, mv))
    
    total_combinations = len(combinations)
    num_cores = multiprocessing.cpu_count()
    # Use process pool to evaluate combinations in parallel
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    logger.info(f"Firing up {num_cores} CPU cores to process {len(combinations)} combinations in parallel!\n")
    
    start_time = time.time()
    best_weights = None
    best_dev = float('inf')
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = {executor.submit(evaluate_combination, w, df_gw, df_players, df_teams, df_fixtures): w for w in combinations}
        for future in concurrent.futures.as_completed(futures):
            w, score = future.result()
            pv, mv = w
            logger.info(f"Tested Params (Process={pv:.2f}, Measure={mv:.2f}) -> Deviance: {score:.4f}")
            if score < best_dev:
                best_dev = score
                best_weights = w

    elapsed = time.time() - start_time
    logger.info("\n==========================================")
    logger.info(f"GRID SEARCH COMPLETE ({len(combinations)} combinations tested in {elapsed:.1f}s)")
    logger.info("==========================================")
    logger.info(f"🏆 BEST PARAMS: Process Variance={best_weights[0]:.2f} | Measurement Variance={best_weights[1]:.2f}")
    logger.info(f"📈 LOWEST POISSON DEVIANCE: {best_dev:.4f}")
    logger.info("==========================================")
    logger.info("To use these weights, update EnsembleForecaster in xp_model.py!")

if __name__ == "__main__":
    run_grid_search()
