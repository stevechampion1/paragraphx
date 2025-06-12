import sys
import os
import numpy as np

# 将 src/ 目录添加到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.paragraphx.graph import Graph
from src.paragraphx.algorithms.sssp import sssp

def main():
    """
    主函数，用于创建带权图，运行SSSP算法，并验证结果。
    """
    print("--- ParaGraphX 并行SSSP (贝尔曼-福特) 示例 ---")

    # --- 场景一：无负权重环路的图 ---
    print("\n--- 场景 1: 带负权重的图 (无负环路) ---")
    #
    #      (1)--> 1 --(3)--> 3
    #     / ^    /          ^
    #    0  |(-2)|         (1)
    #     \ v    \          |
    #      (4)--> 2 --(-1)-> 4
    #
    num_nodes_1 = 5
    edges_1 = [
        (0, 1, 1.0),
        (0, 2, 4.0),
        (1, 2, -2.0), # 负权重边
        (1, 3, 3.0),
        (2, 4, -1.0),
        (4, 3, 1.0)
    ]
    graph_1 = Graph(num_nodes=num_nodes_1, edges=edges_1)
    start_node_1 = 0

    print(f"创建了一个 {graph_1}")
    print(f"运行 SSSP，起始节点为: {start_node_1}")

    # 调用我们的 SSSP 函数
    distances_1, cycle_detected_1 = sssp(graph_1, start_node_1)

    print("\n--- 结果验证 (场景 1) ---")
    # 最短路径应该是：
    # 0 -> 0: 0
    # 0 -> 1: 1
    # 0 -> 1 -> 2: 1 + (-2) = -1
    # 0 -> 1 -> 2 -> 4: -1 + (-1) = -2
    # 0 -> 1 -> 2 -> 4 -> 3: -2 + 1 = -1
    expected_distances_1 = np.array([0., 1., -1., -1., -2.])
    
    print(f"计算出的距离: {distances_1}")
    print(f"预期的距离:   {expected_distances_1}")
    print(f"检测到负环路? {cycle_detected_1} (预期: False)")

    is_correct_1 = np.allclose(distances_1, expected_distances_1) and not cycle_detected_1
    if is_correct_1:
        print("✅ 成功！场景1的结果与预期输出匹配。")
    else:
        print("❌ 错误！场景1的结果与预期输出不匹配。")

    # --- 场景二：有负权重环路的图 ---
    print("\n--- 场景 2: 包含负权重环路的图 ---")
    #
    #      (1)--> 1 --(-2)--> 2
    #             ^         /
    #             |        /
    #            (-1)-----
    #
    num_nodes_2 = 3
    edges_2 = [
        (0, 1, 1.0),
        (1, 2, -2.0),
        (2, 1, -1.0) # 1 -> 2 -> 1 形成负环路
    ]
    graph_2 = Graph(num_nodes=num_nodes_2, edges=edges_2)
    start_node_2 = 0

    print(f"创建了一个 {graph_2}")
    print(f"运行 SSSP，起始节点为: {start_node_2}")

    # 调用我们的 SSSP 函数
    distances_2, cycle_detected_2 = sssp(graph_2, start_node_2)

    print("\n--- 结果验证 (场景 2) ---")
    print(f"计算出的距离: {distances_2}") # 距离可能是-inf或极大负数
    print(f"检测到负环路? {cycle_detected_2} (预期: True)")

    if cycle_detected_2:
        print("✅ 成功！场景2正确地检测到了负权重环路。")
    else:
        print("❌ 错误！场景2未能检测到负权重环路。")

if __name__ == "__main__":
    main()