import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

font = 30

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.family'] = 'DejaVu Sans'

# Data
versions = [
    "Basic, 1",
    "Checkerb., 1",
    "Exp lookup, 1",
    "Xorshiro, 1",
    "Threads, 14",
    "Bit-parallel, 14",
    "Troyer, 14",
    "Metal, 1 S",
    "Metal, 64 S"
]

# Performance numbers (spin flips per second)
# Fill in your measured numbers
performance = [
    1.62E+07,    # Basic Implementation
    1.64E+08,  
    2.05E+08,   
    3.25E+08,
    1.54E+09,
    2.79E+10,
    2.84E+09,
    6.24E+09,
    1.47E+11
]

performance_ns = [p / 1e9 for p in performance] # to nanoseconds

# Key optimizations for each version
optimizations = [
    "Base version with basic RNG",
    "Checkerboard updates",
    "Exp lookup",
    "Fast Xorshiro random number generator",
    "Multithreading",
    "64 parallel simulations using bit operations",
    "Troyer",
    "Apple Metal",
    "Apple Metal 64"
]

# Create figure and axis
fig, (ax1) = plt.subplots(1, 1, gridspec_kw={'hspace': 0.3})

# Create the bar plot
bars = ax1.bar(versions, performance_ns, width=0.7)

# Customize the bars
for i, bar in enumerate(bars):
    bar.set_alpha(0.7)
    bar.set_edgecolor('black')
    bar.set_linewidth(1.5)
    
    # Add value labels on top of bars
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# Customize the plot
#ax1.set_title('Ising Model Performance Optimization Progress, CPU M4 Pro with 14 cores', pad=20, fontsize=16, fontweight='bold')
ax1.set_ylabel('Spin Flips per ns', fontsize=font, fontweight='bold')
ax1.tick_params(axis='x', rotation=90, labelsize=25)

ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='y', labelsize=font)

# Example of adding a vertical line
vertical_line_x = 6.5  # x-coordinate where the vertical line will be drawn
ax1.axvline(x=vertical_line_x, color='black', linestyle='--', linewidth=2, label='Optimization Milestone')

# Adding text annotations
ax1.text(vertical_line_x + 0.5, max(performance_ns)*0.8, 'GPU',
         color='black', fontsize=14, fontweight='bold', ha='center')

# Adding text annotations
ax1.text(vertical_line_x - 0.5, max(performance_ns)*0.8, 'CPU',
         color='black', fontsize=14, fontweight='bold', ha='center')

# Overall layout adjustments
plt.gcf().set_size_inches(10, 6)
plt.tight_layout()

# Save the plot
plt.savefig('ising_optimization_progress.png', dpi=300, bbox_inches='tight')
plt.close()

# Optional: Print speedup factors
baseline = performance[0]
print("\nSpeedup factors compared to baseline:")
for v, p in zip(versions[1:], performance[1:]):
    speedup = p / baseline
    #baseline = p
    print(f"{v}: {speedup:.1f}x faster")


threads = np.arange(1, 15)

### Threading benchmark
performance_threads = [ 1.61E+08,
                        3.13E+08,
                        4.50E+08,
                        5.64E+08,
                        4.51E+08,
                        6.54E+08,
                        7.49E+08,
                        8.26E+08,
                        9.02E+08,
                        9.98E+08,
                        1.03E+09,
                        1.11E+09,
                        1.24E+09,
                        1.54E+09]

performance_threads_ns = [p / 1e9 for p in performance_threads] # to nanoseconds

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(threads, 3.25E+08/1e9  * np.ones(14), label="Xorshiro")
plt.plot(threads, performance_threads_ns, marker='o', label="Simple Multithreading")
plt.plot(threads, performance_threads_ns[0] * np.arange(1,15), marker='d', linestyle = 'dashed', label="Linear improvement")

# Labels and legend
#plt.title("Performance vs Threads", fontsize=16)
plt.xlabel("Number of Threads", fontsize=font)
plt.ylabel("Spin Flips per ns", fontsize=font)
plt.yticks(fontsize=font)
plt.xticks(threads,fontsize=font)
plt.grid(alpha=0.3)
plt.legend(fontsize=25)
#plt.yscale('log')

plt.tight_layout()


# Save the plot
plt.savefig('threading_performance.png', dpi=300, bbox_inches='tight')

# Show plot
plt.close()
