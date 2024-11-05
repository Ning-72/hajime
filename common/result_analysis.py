import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt

def pca_reduction(user_to_vector_dict, n=2):
    # Extract names and vectors
    names = list(user_to_vector_dict.keys())
    # vectors = np.array(list(user_to_vector_dict.values()))
    vectors = np.array(list(user_to_vector_dict.values()))

    # Perform PCA
    pca = PCA(n_components=n)
    reduced_vectors = pca.fit_transform(vectors)

    # Create a new dictionary with the reduced vectors
    reduced_dict = {name: reduced_vectors[i] for i, name in enumerate(names)}

    return reduced_dict

def correlation_estimate(user_to_pc1_dw):
    pc1s = [value[0] for value in user_to_pc1_dw.values()]
    dw_values = [value[1] for value in user_to_pc1_dw.values()]

    # 计算皮尔逊相关系数（线性相关性）
    pearson_corr, _ = pearsonr(pc1s, dw_values)

    # 计算斯皮尔曼相关系数（单调相关性）
    spearman_corr, _ = spearmanr(pc1s, dw_values)

    return pearson_corr, spearman_corr

def split_dict_by_party(democrats, republicans, user_dict):
    democrat_dict = {}
    republican_dict = {}

    for user, value in user_dict.items():
        if user in democrats['twitter_name'].tolist():
            democrat_dict[user] = value
        elif user in republicans['twitter_name'].tolist():
            republican_dict[user] = value
        else:
            continue

    return democrat_dict, republican_dict

def plot_pc2(democrats, republicans, user_to_2pc, user_to_pc1_dw, ax, title):
    democrat_to_pc1_dw, republican_to_pc1_dw = split_dict_by_party(democrats, republicans, user_to_pc1_dw)

    pearsonr_corr, spearman_corr = correlation_estimate(user_to_pc1_dw)
    democrats_pearsonr_corr, democrats_spearman_corr = correlation_estimate(democrat_to_pc1_dw)
    republicans_pearsonr_corr, republicans_spearman_corr = correlation_estimate(republican_to_pc1_dw)

    # 提取共和党和民主党向量的 x 和 y 坐标
    democrat_to_2pc, republican_to_2pc = split_dict_by_party(democrats, republicans, user_to_2pc)

    sign = -1 if pearsonr_corr < 0 else 1

    republican_x = [sign * vec[0] for vec in republican_to_2pc.values()]
    republican_y = [sign * vec[1] for vec in republican_to_2pc.values()]
    democrat_x = [sign * vec[0] for vec in democrat_to_2pc.values()]
    democrat_y = [sign * vec[1] for vec in democrat_to_2pc.values()]

    # 创建二维散点图
    ax.scatter(republican_x, republican_y, color='red', label='Republican', alpha=0.7)
    ax.scatter(democrat_x, democrat_y, color='blue', label='Democrat', alpha=0.7)

    # 添加图例和标签
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title)
    ax.legend()

    # 将相关性参数标注在图像上
    textstr = '\n'.join((
        f'Overall Pearson: {pearsonr_corr:.2f}',
        f'Overall Spearman: {spearman_corr:.2f}',
        f'Democrats Pearson: {democrats_pearsonr_corr:.2f}',
        f'Democrats Spearman: {democrats_spearman_corr:.2f}',
        f'Republicans Pearson: {republicans_pearsonr_corr:.2f}',
        f'Republicans Spearman: {republicans_spearman_corr:.2f}',
    ))

    # 设置文本框的位置和样式
    # plt.gcf().text(0.75, 0.85, textstr, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    ax.text(0.75, 0.85, textstr, transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.5))