from collections import Counter


def remove_low_degree_nodes(edge_list, k_in, k_out):
    follower_degree, followee_degree = degree_calculation(edge_list)

    filtered_edge_list = [
        (followee, follower) for (followee, follower) in edge_list
        if follower_degree[follower] >= k_out and followee_degree[followee] >= k_in
    ]

    return filtered_edge_list

def re_weight_edges(edge_list):
    # 假设用户是关注者，对于连接矩阵上的点(i, j)，三个重要的值是
    # 1. 用户i关注了多少议会成员（度较高的用户）
    # 2. 议会成员j被多少用户关注（度较高的议会成员）
    # 3. 一共有多少个edge
    # 对于邻接矩阵上的点(i, j)，其重新分配的权重为3/1*2
    follower_degree, followee_degree = degree_calculation(edge_list)
    edge_to_weight_dict = {}

    for edge in edge_list:
        follower = edge[1]
        followee = edge[0]
        weight = len(edge_list) / follower_degree[follower] * followee_degree[followee]
        edge_to_weight_dict[(followee, follower)] = weight

    return edge_to_weight_dict

def degree_calculation(edge_list):
    follower_degree = Counter()
    followee_degree = Counter()

    for edge in edge_list:
        followee, follower = edge
        follower_degree[follower] += 1
        followee_degree[followee] += 1

    return follower_degree, followee_degree