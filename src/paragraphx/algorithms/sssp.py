import jax
import jax.numpy as jnp
from ..graph import Graph

def _sssp_internal(graph: Graph, start_node: int) -> tuple[jnp.ndarray, bool]:
    """
    内部函数，使用并行的贝尔曼-福特算法计算单源最短路径 (SSSP)。

    Args:
        graph (Graph): 输入的带权图对象。
        start_node (int): 起始节点的索引。

    Returns:
        tuple[jnp.ndarray, bool]: 
            - 一个JAX数组，包含从起始节点到各节点的最短距离。
            -一个布尔值，如果检测到负权重环路，则为True，否则为False。
    """
    num_nodes = graph.num_nodes
    adj_matrix = graph.get_adj()

    # 初始化距离：起始节点为0，其他所有节点为无穷大。
    initial_distances = jnp.full(num_nodes, jnp.inf, dtype=jnp.float32).at[start_node].set(0)

    def main_loop_body(iteration, distances):
        # 核心并行松弛步骤：
        # 1. 使用广播计算所有可能的 one-hop 路径长度： D[u] + weight(u, v)
        #    distances.T 将 (N,) 的向量变为 (N, 1)，可以和 (N, N) 的邻接矩阵进行广播相加。
        candidate_distances = distances[:, None] + adj_matrix
        
        # 2. 对每个节点v，从所有可能的中间节点u中找到最短路径。
        #    这相当于取候选矩阵每一列的最小值。
        new_distances_from_relaxation = jnp.min(candidate_distances, axis=0)
        
        # 3. 更新距离：取当前距离和新计算出的距离中较小的一个。
        updated_distances = jnp.minimum(distances, new_distances_from_relaxation)
        
        return updated_distances

    # 运行 N-1 轮松弛操作。
    # jax.lax.fori_loop 是一个可JIT编译的for循环。
    final_distances = jax.lax.fori_loop(0, num_nodes - 1, main_loop_body, initial_distances)
    
    # 负权重环路检测：
    # 再进行一轮松弛操作。如果距离还能被缩短，说明存在负权重环路。
    distances_after_final_check = main_loop_body(num_nodes, final_distances)
    has_negative_cycle = jnp.any(distances_after_final_check < final_distances)

    return final_distances, has_negative_cycle

# 创建最终的、经过JIT编译的、可供外部使用的sssp函数。
sssp = jax.jit(_sssp_internal, static_argnames='graph')