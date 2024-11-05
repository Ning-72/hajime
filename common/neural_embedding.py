import torch.nn as nn
import torch.optim as optim

from common.encoding import *
from common.network_analysis import *
from utils.file_utils import *


class ShallowNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, bias=False):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=bias),
            nn.Linear(hidden_dim, output_dim, bias=bias)
        )

    def forward(self, x):
        x = self.linear_stack(x)
        return x

def calculate_loss_by_weights(loss_function, output_vectors, target_labels, weights):
    total_loss = 0.0
    batch_size = len(weights)

    normalized_weights = [w * batch_size / sum(weights) for w in weights]

    for i in range(batch_size):
        # 这里的weights中的数据是否需要等比例放大？
        total_loss += normalized_weights[i] * loss_function(output_vectors[i], target_labels[i])

    loss = total_loss / batch_size

    return loss

"""
one hot编码后的向量构成的list太大了，无法存储在内存堆栈中
解决方案：不再生成完整的one hot vector list，而是每需要用到一个/部分，临时生成对应的vector (list)
"""
def train_network(model, epoch_num, batch_size,
                  loss_function, optimizer, weighted_edge_list, follower_map, followee_map):

    training_sample_size = len(weighted_edge_list)

    loss_history = []

    for epoch in range(epoch_num):
        model.train()

        epoch_loss = 0.0
        batch_num = 0

        for i in range(0, training_sample_size, batch_size):
            batch_training_samples = weighted_edge_list[i:i + batch_size]
            edge_list = [edge for edge, _ in batch_training_samples]
            weight_list = [weight for _, weight in batch_training_samples]

            input_list = list(j[1] for j in edge_list)
            target_list = list(j[0] for j in edge_list)

            input_vectors, _ = one_hot_encode(input_list, follower_map)
            target_labels = get_labels(target_list, followee_map)

            input_vectors = torch.stack(input_vectors)

            output_vectors = model(input_vectors)

            # print(input_vectors.shape, output_vectors.shape, target_vectors.shape)

            # loss = loss_function(output_vectors, target_labels)
            loss = calculate_loss_by_weights(loss_function, output_vectors, target_labels, weight_list)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_num += 1

        avg_loss = epoch_loss / batch_num
        loss_history.append(avg_loss)

        print(f'Epoch [{epoch + 1}/{epoch_num}], Loss: {avg_loss:.4f}')

    return loss_history


def train_network_with_filtered_dataset(output_file_directory, edge_list, k_list,
                                        embedding_vector_dim_list, epoch_num,
                                        batch_size):
    k_dim_to_loss_history = {}

    for k in k_list:
        # Remove nodes with low degree and save filtered edge list as a csv file.
        filtered_edge_list = remove_low_degree_nodes(edge_list, k[0], k[1])
        filtered_edge_list_df = pd.DataFrame(filtered_edge_list, columns=['followee', 'follower'])
        edge_file_name = 'global_edge_list_kin' + str(k[0]) + '_kout' + str(k[1]) + '.csv'
        save_to_csv(output_file_directory, edge_file_name, filtered_edge_list_df)

        # Add weights to filtered edge list.
        filtered_edge_to_weight_dict = {}
        for edge in filtered_edge_list:
            filtered_edge_to_weight_dict[edge] = 1.00
        weighted_edge_list_for_training = list(filtered_edge_to_weight_dict.items())

        # Obtain follower_map and followee map, which are dictionaries mapping usernames to integer values.
        congress_map, followee_map = build_dict(filtered_edge_list)

        for embedding_vector_dim in embedding_vector_dim_list:
            congress_map_length = len(congress_map)
            followee_map_length = len(followee_map)

            print('Input dimension is ' + str(congress_map_length) + '.')
            print('Hidden dimension is ' + str(embedding_vector_dim) + '.')
            print('Output dimension is ' + str(followee_map_length) + '.')

            print(
                'Training NN with k_in >=' + str(k[0]) + ', kout >=' + str(k[1]) + ' and embedding_vector_dim = ' + str(
                    embedding_vector_dim) + ' ...')

            # Build NN.
            shallow_nn = ShallowNN(congress_map_length, embedding_vector_dim, followee_map_length)

            # Training configurations.
            loss_function = nn.CrossEntropyLoss()
            optimizer = optim.Adam(shallow_nn.parameters(), lr=0.001)

            # Train NN and get loss value of every epoch.
            loss_history = train_network(shallow_nn, epoch_num, batch_size, loss_function,
                                         optimizer, weighted_edge_list_for_training, congress_map, followee_map)
            k_dim_to_loss_history[(k, embedding_vector_dim)] = loss_history

            # Get weight matrix and bias.
            input_to_hidden_weights = shallow_nn.linear_stack[0].weight.data
            input_to_hidden_bias = shallow_nn.linear_stack[0].bias

            # Get congress-to-embedding dict and save.
            # A congress list is needed to determine one hot vector length.
            congress_list = [item[1] for item in filtered_edge_list]
            _, congress_to_vector_dict = one_hot_encode(congress_list, congress_map)
            user_to_embedding_dict = calculate_embeddings(congress_to_vector_dict, input_to_hidden_weights,
                                                          input_to_hidden_bias)

            embedding_file_name = 'global_embedding_kin' + str(k[0]) + '_kout' + str(k[1]) + '_dim' + str(
                embedding_vector_dim) + '.pth'
            save_to_pth(output_file_directory, embedding_file_name, user_to_embedding_dict)

    return k_dim_to_loss_history

def train_network_with_reweighted_dataset(output_file_directory, weighted_edge_to_weight_list,
                                          embedding_vector_dim_list, epoch_num, batch_size):
    # Remove nodes with low degree and save filtered edge list as a csv file.
    edge_list = [edge for edge, _ in weighted_edge_to_weight_list]

    # Obtain follower_map and followee map, which are dictionaries mapping usernames to integer values.
    congress_map, followee_map = build_dict(edge_list)

    dim_to_loss_history = {}

    for embedding_vector_dim in embedding_vector_dim_list:
        congress_map_length = len(congress_map)
        followee_map_length = len(followee_map)

        print('Input dimension is ' + str(congress_map_length) + '.')
        print('Hidden dimension is ' + str(embedding_vector_dim) + '.')
        print('Output dimension is ' + str(followee_map_length) + '.')

        print(
            'Training NN with re-weighted edges and embedding_vector_dim = ' + str(
                embedding_vector_dim) + ' ...')

        # Build NN.
        shallow_nn = ShallowNN(congress_map_length, embedding_vector_dim, followee_map_length)

        # Training configurations.
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(shallow_nn.parameters(), lr=0.001)

        # Train NN and get loss value of every epoch.
        loss_history = train_network(shallow_nn, epoch_num, batch_size, loss_function,
                                     optimizer, weighted_edge_to_weight_list, congress_map, followee_map)
        dim_to_loss_history[embedding_vector_dim] = loss_history

        # Get weight matrix and bias.
        input_to_hidden_weights = shallow_nn.linear_stack[0].weight.data
        input_to_hidden_bias = shallow_nn.linear_stack[0].bias

        # Get congress-to-embedding dict and save.
        # A congress list is needed to determine one hot vector length.
        congress_list = [item[1] for item in edge_list]
        _, congress_to_vector_dict = one_hot_encode(congress_list, congress_map)
        user_to_embedding_dict = calculate_embeddings(congress_to_vector_dict, input_to_hidden_weights,
                                                      input_to_hidden_bias)

        embedding_file_name = 'global_embedding_for_reweighted_edges' + '_dim' + str(
            embedding_vector_dim) + '.pth'
        save_to_pth(output_file_directory, embedding_file_name, user_to_embedding_dict)

    return dim_to_loss_history