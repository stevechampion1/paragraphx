import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, List, Tuple, Union

class Graph:
    """
    一个用JAX表示的图结构，为GPU上的并行计算优化。
    内部使用一个带权重的邻接矩阵来表示图。
    它现在可以智能地处理带权重和不带权重的边。
    """
    # 更新了类型提示，使其更通用
    def __init__(self, num_nodes: int, edges: Optional[List[Union[Tuple[int, int], Tuple[int, int, float]]]] = None):
        """
        初始化一个图。

        Args:
            num_nodes (int): 图中的节点数量。
            edges (Optional[...]): 边的列表。
                可以是无权重的格式: [(u, v), ...]
                也可以是带权重的格式: [(u, v, weight), ...]
        """
        self.num_nodes = num_nodes
        
        # 对于BFS等无权图算法，默认权重为1。对于SSSP，则表示距离。
        # 我们用无穷大初始化矩阵，表示节点之间默认不可达。
        adj = np.full((num_nodes, num_nodes), jnp.inf, dtype=jnp.float32)
        
        # 节点到自身的距离为0。
        np.fill_diagonal(adj, 0)
        
        if edges:
            # --- 这里是修正的核心 ---
            # 检查边的格式，让类变得更智能
            if len(edges[0]) == 3:
                # 这是带权重的边列表 (u, v, weight)
                for u, v, weight in edges:
                    if u < num_nodes and v < num_nodes:
                        adj[u, v] = weight
            elif len(edges[0]) == 2:
                # 这是无权重的边列表 (u, v)，我们为其赋予默认权重1.0
                for u, v in edges:
                    if u < num_nodes and v < num_nodes:
                        adj[u, v] = 1.0
            else:
                raise ValueError("边的格式不正确。请提供 (u, v) 或 (u, v, weight) 格式的元组列表。")
        
        # 将最终的邻接矩阵移动到默认的JAX设备上
        self.adjacency_matrix = jax.device_put(adj)

    def __repr__(self) -> str:
        return f"Graph(num_nodes={self.num_nodes}) with weighted adjacency matrix on {self.adjacency_matrix.device}"

    def add_edge(self, u: int, v: int, weight: float = 1.0):
        """
        在图中添加一条带权重的边。
        """
        self.adjacency_matrix = self.adjacency_matrix.at[u, v].set(weight)

    def get_adj(self) -> jnp.ndarray:
        """返回带权重的邻接矩阵"""
        return self.adjacency_matrix