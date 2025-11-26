"""
Baseline Profiling using cProfile
"""

from config import create_config
from data_generation import generate_base_data
from simulation import run_simulation_with_base_data
import cProfile
import pstats

# optimized version
from simulation_optimized import run_simulation_with_base_data as run_simulation_with_base_data_opt

def run_simulation_for_profiling():
    """functions need to be profiled"""
    config = create_config(m=32, m0=16, distribution='E', n_reps=1000, seed=12345)
    base_data = generate_base_data(config)
    results = run_simulation_with_base_data(config, base_data, 
                                           show_progress=False, 
                                           save_results=False)
    return results

def run_simulation_for_profiling_opt():
    """functions need to be profiled"""
    config = create_config(m=32, m0=16, distribution='E', n_reps=1000, seed=12345)
    base_data = generate_base_data(config)
    results = run_simulation_with_base_data_opt(config, base_data, 
                                           show_progress=False, 
                                           save_results=False)
    return results

if __name__ == "__main__":

    profiler = cProfile.Profile()
    profiler.enable()
    
    results = run_simulation_for_profiling()
    
    profiler.disable()
    # Save to file
    profiler.dump_stats('profile_stats.prof')

    # Optimized version
    profiler_opt = cProfile.Profile()
    profiler_opt.enable()
    results_opt = run_simulation_for_profiling_opt()
    profiler_opt.disable()
    # Save to file
    profiler_opt.dump_stats('profile_stats_opt.prof')
