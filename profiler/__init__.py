"""
Profiler tools for JAXTPC simulation.

Scripts:
    python3 -m profiler.find_optimal_pad --data events.h5 --config config.yaml
    python3 -m profiler.benchmark_sim --data events.h5 --config config.yaml
    python3 -m profiler.find_optimal_chunks --data events.h5 --config config.yaml
    python3 -m profiler.threshold_analysis --data events.h5 --config config.yaml
    python3 -m profiler.find_optimal_inter_thresh --data events.h5 --config config.yaml
    python3 -m profiler.find_optimal_max_keys --data events.h5 --config config.yaml
    python3 -m profiler.find_optimal_max_buckets --data events.h5 --config config.yaml
    python3 -m profiler.setup_production --data events.h5 --config config.yaml
"""
