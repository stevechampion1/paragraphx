import jax
import jax.numpy as jnp
from ..graph import Graph  # 相对导入

# --- 这里是修正的地方 ---
# 我们不再使用 @jax.jit 装饰器语法，而是显式地定义一个内部函数，
# 然后手动调用 jax.jit 来创建已编译的最终版本。
# 这可以绕过某些环境下装饰器语法解析的歧义。

def _bfs_internal(graph: Graph, start_node: int) -> jnp.ndarray:
    """
    这是一个内部函数，包含了BFS的核心逻辑。它本身未被JIT编译。
    """
    num_nodes = graph.num_nodes
    adj_matrix = graph.get_adj()

    initial_distances = jnp.full(num_nodes, -1, dtype=jnp.int32).at[start_node].set(0)
    initial_frontier = jnp.zeros(num_nodes, dtype=jnp.int32).at[start_node].set(1)
    initial_level = 0
    
    def condition_fun(state):
        _level, _distances, frontier = state
        return jnp.any(frontier)

    def body_fun(state):
        level, distances, frontier = state
        next_level = level + 1
        new_frontier = jnp.dot(frontier, adj_matrix).astype(jnp.int32)
        is_visited = distances != -1
        new_frontier = jnp.where(is_visited, 0, new_frontier)
        new_distances = jnp.where(new_frontier, next_level, distances)
        return next_level, new_distances, new_frontier

    _final_level, final_distances, _final_frontier = jax.lax.while_loop(
        condition_fun,
        body_fun,
        (initial_level, initial_distances, initial_frontier)
    )
    return final_distances

# 现在，我们创建最终的、可供外部使用的、经过JIT编译的bfs函数。
# 我们把内部函数传给jit，并告诉它'graph'是一个静态参数。
bfs = jax.jit(_bfs_internal, static_argnames='graph')