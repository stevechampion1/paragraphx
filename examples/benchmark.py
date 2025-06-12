import sys
import os
import time
import numpy as np
import networkx as nx
# ... (上面的import)
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 新增代码：设置Matplotlib以支持中文 ---
try:
    # 优先使用黑体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 解决负号'-'显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"警告：未能设置中文字体。图表中的中文可能无法正常显示。错误: {e}")
# --- 新增代码结束 ---

# 将 src/ 目录添加到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ... (文件其余部分保持不变)

# 将 src/ 目录添加到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.paragraphx.graph import Graph as PxGraph
from src.paragraphx.algorithms.bfs import bfs as px_bfs

def generate_random_graph(num_nodes, density=0.01):
    """生成一个随机图并返回两种格式的图对象。"""
    # 生成一个 NetworkX 图
    g_nx = nx.fast_gnp_random_graph(num_nodes, density, directed=True)
    
    # 转换为我们的 ParaGraphX 格式
    edges_px = [(u, v) for u, v in g_nx.edges()]
    g_px = PxGraph(num_nodes=num_nodes, edges=edges_px)
    
    return g_px, g_nx

def time_paragraphx_bfs(graph, start_node):
    """测量 ParaGraphX BFS 的执行时间。"""
    # 第一次运行用于JIT编译
    px_bfs(graph, start_node).block_until_ready()
    
    # 现在开始计时
    start_time = time.perf_counter()
    px_bfs(graph, start_node).block_until_ready()
    end_time = time.perf_counter()
    return end_time - start_time

def time_networkx_bfs(graph, start_node):
    """测量 NetworkX BFS 的执行时间。"""
    start_time = time.perf_counter()
    # NetworkX的BFS算法返回一个生成器
    # 我们需要消耗掉它来确保工作已完成
    _ = list(nx.bfs_edges(graph, start_node))
    end_time = time.perf_counter()
    return end_time - start_time

def main():
    print("--- 基准测试: ParaGraphX vs. NetworkX (BFS) ---")
    
    # 定义要测试的图的规模
    node_counts = [100, 500, 1000, 2000, 4000, 8000]
    
    px_times = []
    nx_times = []
    
    start_node = 0
    
    for n in tqdm(node_counts, desc="分析图..."):
        # 为基准测试生成随机图
        # 我们使用一个较低的密度，这在真实世界的图中很常见
        g_px, g_nx = generate_random_graph(n, density=0.01)
        
        # 测量 ParaGraphX 的时间
        px_time = time_paragraphx_bfs(g_px, start_node)
        px_times.append(px_time)
        
        # 测量 NetworkX 的时间
        nx_time = time_networkx_bfs(g_nx, start_node)
        nx_times.append(nx_time)

    # 打印结果
    print("\n--- 基准测试结果 (单位: 秒) ---")
    for i, n in enumerate(node_counts):
        print(f"节点数: {n:5d} | ParaGraphX: {px_times[i]:.6f}s | NetworkX: {nx_times[i]:.6f}s")
        
    # 绘制结果图表
    plt.figure(figsize=(10, 6))
    plt.plot(node_counts, px_times, 'o-', label='ParaGraphX (JAX)', color='red')
    plt.plot(node_counts, nx_times, 's-', label='NetworkX (CPU)', color='blue')
    
    plt.xlabel('图中的节点数量')
    plt.ylabel('执行时间 (秒)')
    plt.title('BFS 性能基准测试: ParaGraphX vs. NetworkX')
    plt.legend()
    plt.grid(True)
    plt.yscale('log') # 对y轴使用对数刻度，以便更好地观察
    
    # 将图表保存到文件
    output_filename = 'bfs_benchmark.png'
    plt.savefig(output_filename)
    
    print(f"\n基准测试图表已保存为 '{output_filename}'")
    
if __name__ == "__main__":
    main()