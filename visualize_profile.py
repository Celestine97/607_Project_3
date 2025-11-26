"""
Visualize profiling results from profile_stats.prof
Creates informative component-level timing and time distribution visualizations
"""

import pstats
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_profile_stats(filename='profile_stats.prof'):
    """Load profiling statistics"""
    stats = pstats.Stats(filename)
    return stats

def get_component_stats(stats):
    """Aggregate timing statistics by component"""

    component_times = {}

    # nc: total number of calls
    # tt: total time spent in this function (excluding subcalls)
    # ct: cumulative time spent in this function (including subcalls)
    for func, (_, nc, tt, ct, _) in stats.stats.items():
        filename, _, func_name = func

        # Map function to component category
        component = None
        if 'generate' in func_name.lower() or 'data_generation' in filename:
            component = 'Data Generation'
        elif 'compute_pvalues' in func_name:
            component = 'P-value computation\n(scipy.stats.norm.cdf)'
        elif 'cdf' in func_name and 'scipy' in filename:
            component = 'P-value computation\n(scipy.stats.norm.cdf)'
        elif 'benjamini_hochberg_method' in func_name:
            component = 'BH method'
        elif 'hochberg_method' in func_name:
            component = 'Hochberg method'
        elif 'bonferroni' in func_name:
            component = 'Bonferroni'
        elif 'compute_power' in func_name:
            component = 'Power computation'
        elif 'compute_fdr' in func_name:
            component = 'FDR computation'
        elif 'run_simulation' in func_name and 'project3' in filename:
            component = 'Other'

        # Aggregate statistics for each component
        if component:
            if component not in component_times:
                component_times[component] = {'tottime': 0, 'cumtime': 0, 'ncalls': 0}
            component_times[component]['tottime'] += tt
            component_times[component]['cumtime'] += ct
            component_times[component]['ncalls'] += nc

    return component_times

def create_visualizations(profile_file='profile_stats.prof', output_path = 'profile_visualization.png'):
    """Create component-level timing and distribution visualizations"""

    stats = load_profile_stats(profile_file)
    component_times = get_component_stats(stats)

    # Calculate per-call times (divide by number of simulation iterations)
    n_iterations = 1000

    # Prepare data for visualizations
    components = []
    per_call_times = []
    cumulative_times = []

    for comp, times in component_times.items():
        if comp != 'Other':
            components.append(comp)
            # Per call time in milliseconds
            per_call_times.append(times['cumtime'] / n_iterations * 1000)
            cumulative_times.append(times['cumtime'])

    total_time = stats.total_tt
    accounted_time = sum([times['tottime'] for times in component_times.values()])
    other_time = max(0, total_time - accounted_time)
    

    if other_time > 0:
        components.append('Other')
        per_call_times.append(other_time / n_iterations * 1000)
        cumulative_times.append(other_time)

    # Sort by per-call time
    sorted_indices = np.argsort(per_call_times)[::-1]
    components = [components[i] for i in sorted_indices]
    per_call_times = [per_call_times[i] for i in sorted_indices]
    cumulative_times = [cumulative_times[i] for i in sorted_indices]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Component-Level Timing (Bar chart)
    y_pos = np.arange(len(components))

    ax1.barh(y_pos, per_call_times)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(components, fontsize=11)
    ax1.invert_yaxis()
    ax1.set_xlabel('Time per call (milliseconds)', fontsize=12)
    ax1.set_title('Component-Level Timing\n(Single function call)', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for i, v in enumerate(per_call_times):
        ax1.text(v, i, f'  {v:.4f}', va='center', fontsize=9)

    # 2. Time Distribution (Pie chart)
    total_cumtime = sum(cumulative_times)
    percentages = [(ct / total_cumtime * 100) for ct in cumulative_times]

    # Only show labels for segments > 1%
    labels = []
    for i, (comp, pct) in enumerate(zip(components, percentages)):
        if pct > 1:
            labels.append(f'{comp}\n{pct:.1f}%')
        else:
            labels.append('')

    wedges, texts = ax2.pie(cumulative_times, labels=labels,
                             startangle=90, counterclock=False, textprops={'fontsize': 10})

    ax2.set_title('Profiling: Time Distribution in Simulation Loop\n(Cumulative time breakdown)',
                  fontsize=13, fontweight='bold')

    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
   create_visualizations(profile_file='profile_stats.prof', output_path = 'profile_visualization.png')
   create_visualizations(profile_file='profile_stats_opt.prof', output_path = 'profile_visualization_opt.png')
