import torch
import torch.nn as nn
import torch.optim as optim

from common.encoding import *


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

"""
one hot编码后的向量构成的list太大了，无法存储在内存堆栈中
解决方案：不再生成完整的one hot vector list，而是每需要用到一个/部分，临时生成对应的vector (list)
"""
def train_network(model, epoch_num, batch_size,
                  loss_function, optimizer, edge_list, follower_map, followee_map):

    training_sample_size = len(edge_list)

    loss_history = []

    for epoch in range(epoch_num):
        model.train()

        epoch_loss = 0.0
        batch_num = 0

        for i in range(0, training_sample_size, batch_size):
            batch_training_samples = edge_list[i:i + batch_size]
            input_list = list(j[1] for j in batch_training_samples)
            target_list = list(j[0] for j in batch_training_samples)

            input_vectors, _ = one_hot_encode(input_list, follower_map)
            target_vectors = get_labels(target_list, followee_map)

            input_vectors = torch.stack(input_vectors)

            output_vectors = model(input_vectors)

            # print(input_vectors.shape, output_vectors.shape, target_vectors.shape)

            loss = loss_function(output_vectors, target_vectors)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_num += 1

        avg_loss = epoch_loss / batch_num
        loss_history.append(avg_loss)

        print(f'Epoch [{epoch + 1}/{epoch_num}], Loss: {avg_loss:.4f}')

    return loss_history