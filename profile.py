from hta.trace_analysis import TraceAnalysis
analyzer = TraceAnalysis(trace_dir = "./profile")
# Temporal breakdown
temporal_breakdown_df = analyzer.get_temporal_breakdown()

# Idle time breakdown
idle_time_df = analyzer.get_idle_time_breakdown()

# Kernel breakdown
kernel_breakdown_df = analyzer.get_gpu_kernel_breakdown(visualize=True)
print(kernel_breakdown_df)
# Communication computation overlap
comm_comp_overlap_df = analyzer.get_comm_comp_overlap()

# Memory bandwidth time series
# memory_bw_series = analyzer.get_memory_bw_time_series()

# Memory bandwidth summary
# memory_bw_summary = analyzer.get_memory_bw_summary()

# Queue length time series
# ql_series = analyzer.get_queue_length_time_series()

# Queue length summary
# ql_summary = analyzer.get_queue_length_summary()

# CUDA kernel launch statistics
# cuda_kernel_launch_stats = analyzer.get_cuda_kernel_launch_stats()
# print(kernel_metrics_df)