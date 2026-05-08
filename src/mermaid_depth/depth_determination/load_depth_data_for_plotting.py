import numpy as np
import matplotlib.pyplot as plt
from mermaid_depth.misc.read_tomocat1 import read_tomocat1

def plot_depth_hist(data):
    depth_diffs = data["depth_differences"]
    
    plt.figure(figsize=(10, 6))
    plt.hist(depth_diffs, bins=20, edgecolor='black')
    plt.title('Histogram of Depth Differences')
    plt.xlabel('Found Depth minus GEBCO Depth (m)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def plot_depthdiff_epicentral(data):
    depth_diffs = data["depth_differences"]
    
    data_indices = data["data_indices"]
    print(f"failed data indices: {data["failed_data_indices"]}")
    
    mermaid_all = read_tomocat1("./tomocat1.txt")
    epicentral_distances = np.asarray(mermaid_all["gcarc_1D"])[data_indices]

    plt.figure(figsize=(10, 6))
    plt.scatter(epicentral_distances, depth_diffs, alpha=0.3)
    plt.title('Depth Difference vs Epicentral Distance')
    plt.xlabel('Epicentral Distance (degrees)')
    plt.ylabel('Found Depth minus GEBCO Depth (m)')
    plt.grid(True)
    plt.show()

def plot_depthdiff_evdp(data):
    depth_diffs = data["depth_differences"]
    
    data_indices = data["data_indices"]
    print(f"failed data indices: {data["failed_data_indices"]}")
    
    mermaid_all = read_tomocat1("./tomocat1.txt")
    evdps = np.asarray(mermaid_all["evdp"])[data_indices]

    plt.figure(figsize=(10, 6))
    plt.scatter(evdps, depth_diffs, alpha=0.3)
    plt.title('Depth Difference vs Event Depth')
    plt.xlabel('Event Depth (km)')
    plt.ylabel('Found Depth minus GEBCO Depth (m)')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    data = np.load("./saved_data/depth_search_results.npz", allow_pickle=True)

    # plot_depth_hist(data)
    # plot_depthdiff_epicentral(data)
    plot_depthdiff_evdp(data)