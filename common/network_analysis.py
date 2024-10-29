from collections import Counter


def remove_low_degree_nodes(edge_list, k_in, k_out):
    follower_degree, followee_degree = degree_calculation(edge_list)

    filtered_edge_list = [
        (followee, follower) for (followee, follower) in edge_list
        if follower_degree[follower] >= k_out and followee_degree[followee] >= k_in
    ]

    return filtered_edge_list

def degree_calculation(edge_list):
    follower_degree = Counter()
    followee_degree = Counter()

    for edge in edge_list:
        followee, follower = edge
        follower_degree[follower] += 1
        followee_degree[followee] += 1

    return follower_degree, followee_degree