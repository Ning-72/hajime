import torch


def build_dict(edge_list):
    followee_set = list(set(i[0] for i in edge_list))
    follower_set = list(set(i[1] for i in edge_list))

    followee_map = {label: idx for idx, label in enumerate(followee_set)}
    follower_map = {label: idx for idx, label in enumerate(follower_set)}

    return follower_map, followee_map


def one_hot_encode(user_list, user_map):
    vector_length = len(user_map)
    user_vectors = []
    user_to_vector_dict = {}

    for user in user_list:
        if user not in user_map:
            continue

        one_hot_vector = torch.zeros(vector_length, dtype=torch.float)

        user_index = user_map[user]
        one_hot_vector.scatter_(0, torch.tensor(user_index), value = 1.0)

        user_to_vector_dict[user] = one_hot_vector

        user_vectors.append(one_hot_vector)

    return user_vectors, user_to_vector_dict

def calculate_embeddings(user_to_vector_dict, weights, bias):
    user_to_embedding_dict = {}

    for user, vector in user_to_vector_dict.items():
        if bias is None:
            embedding = torch.matmul(vector, weights.t())
        else:
            embedding = torch.matmul(vector, weights.t()) + bias
        
        user_to_embedding_dict[user] = embedding

    return user_to_embedding_dict

def get_labels(user_list, user_dict):
    labels = []

    for user in user_list:
        if user not in user_dict:
            continue

        labels.append(user_dict[user])

    labels_tensor = torch.tensor(labels)

    return labels_tensor