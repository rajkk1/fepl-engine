import logging
import time
import concurrent.futures
from functools import partial
from backtest import fetch_data, run_backtest
import multiprocessing

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def evaluate_combination(weights, df_gw, df_players, df_teams, df_fixtures):
    """Worker function for multiprocessing."""
    mae = run_backtest(weights=weights, df_gw=df_gw, df_players=df_players, df_teams=df_teams, df_fixtures=df_fixtures)
    return weights, mae

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

    # Iterate over Dixon-Coles, Kalman, and GBT weights in steps of 0.10
    steps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    combinations = []

    for w_dc in steps:
        for w_kf in steps:
            w_gt = round(1.0 - w_dc - w_kf, 2)
            # Ensure the weights sum exactly to 1.0 and are all positive
            if w_gt >= 0.0 and abs((w_dc + w_kf + w_gt) - 1.0) < 0.01:
                combinations.append((w_dc, w_kf, w_gt))
    
    total_combinations = len(combinations)
    num_cores = multiprocessing.cpu_count()
    logger.info(f"Firing up {num_cores} CPU cores to process {total_combinations} combinations in parallel!\n")

    start_time = time.time()
    best_mae = 999.0
    best_weights = None

    # Use partial to bind the dataframe arguments so map() only needs to pass the weights
    worker = partial(evaluate_combination, df_gw=df_gw, df_players=df_players, df_teams=df_teams, df_fixtures=df_fixtures)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit all tasks and process as they complete
        futures = {executor.submit(worker, w): w for w in combinations}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                weights, mae = future.result()
                w_dc, w_kf, w_gt = weights
                logger.info(f"Tested Weights (DC={w_dc:.2f}, KF={w_kf:.2f}, GT={w_gt:.2f}) -> MAE: {mae:.3f}")
                if mae < best_mae:
                    best_mae = mae
                    best_weights = weights
            except Exception as exc:
                logger.error(f"A combination generated an exception: {exc}")

    elapsed = time.time() - start_time
    logger.info("\n==========================================")
    logger.info(f"GRID SEARCH COMPLETE ({total_combinations} combinations tested in {elapsed:.1f}s)")
    logger.info("==========================================")
    logger.info(f"🏆 BEST WEIGHTS: Dixon-Coles={best_weights[0]:.2f} | Kalman={best_weights[1]:.2f} | GBT={best_weights[2]:.2f}")
    logger.info(f"📉 LOWEST ERROR (MAE): {best_mae:.4f} pts per player")
    logger.info("==========================================\n")
    logger.info("To use these weights, update EnsembleForecaster in xp_model.py!")

if __name__ == "__main__":
    run_grid_search()
