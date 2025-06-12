import pytest
from paragraphx import Graph, bfs

def test_bfs_simple():
    # 准备一个图
    graph = Graph(num_nodes=4, edges=[(0, 1), (1, 2), (2, 3)])
    
    # 运行BFS
    distances = bfs(graph, start_node=0)
    
    # 验证结果是否符合预期
    assert distances[0] == 0
    assert distances[1] == 1
    assert distances[2] == 2
    assert distances[3] == 3