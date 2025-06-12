import sys
import os

# 这是一个常用技巧，目的是让 src/ 目录可以被导入
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.paragraphx.graph import Graph
from src.paragraphx.algorithms.bfs import bfs

def main():
    """
    主函数，用于创建图、运行BFS并打印结果。
    """
    print("--- ParaGraphX 并行BFS示例 ---")

    # 1. 定义图的结构
    #
    #      0 -- 1 -- 3
    #      |    |
    #      |    2 -- 4 -- 5
    #      |         |
    #      +---------+
    #
    num_nodes = 6
    edges = [
        (0, 1), (1, 0),
        (0, 2), (2, 0),
        (1, 2), (2, 1),
        (1, 3), (3, 1),
        (2, 4), (4, 2),
        (4, 5), (5, 4),
        (0, 4), (4, 0) # 从0到4有一条更短的直连边
    ]
    
    # 2. 创建图对象
    graph = Graph(num_nodes=num_nodes, edges=edges)
    print(f"已创建图: {graph}")
    print("邻接矩阵:")
    print(graph.get_adj())

    # 3. 选择一个起始节点并运行 BFS 算法
    start_node = 0
    print(f"\n从起始节点 {start_node} 开始运行BFS...")
    
    # 这里是调用我们 JIT 编译的 BFS 函数
    distances = bfs(graph, start_node)

    # 第一次运行可能会因为 JIT 编译而稍慢。
    # 后续在同样大小的图上运行时会快得多。

    # 4. 以可读的格式打印结果
    print("\n--- 运行结果 ---")
    print(f"从节点 {start_node} 出发的距离:")
    for i in range(num_nodes):
        dist = distances[i]
        if dist == -1:
            print(f"  节点 {i}: 无法到达")
        else:
            print(f"  节点 {i}: 距离 = {dist}")

    print("\n--- 结果验证 ---")
    expected_distances = [0, 1, 1, 2, 1, 2]
    print(f"预期距离: {expected_distances}")
    
    # 将JAX数组转换为NumPy数组以便进行比较
    import numpy as np
    is_correct = np.array_equal(distances, np.array(expected_distances))
    
    if is_correct:
        print("✅ 成功！结果与预期输出匹配。")
    else:
        print("❌ 错误！结果与预期输出不匹配。")
        print(f"实际得到: {distances}")

if __name__ == "__main__":
    main()