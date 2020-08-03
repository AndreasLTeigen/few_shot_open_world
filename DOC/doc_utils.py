from torch import nn
from scipy.stats import norm as dist_model

from few_shot.utils import pairwise_distances

#fit a gaussian model
def fit(prob_pos_X):
    prob_pos = [p for p in prob_pos_X]+[1-p for p in prob_pos_X]
    pos_mu, pos_std = dist_model.fit(prob_pos)
    return pos_mu, pos_std


#calculate mu, std of each seen class
def get_class_fit(support, prototypes, k_way, n_shot, distance_metric):
    sigmoid = nn.Sigmoid()
    support_distances = pairwise_distances(support, prototypes, distance_metric)
    support_prob = sigmoid(-support_distances)
    support_prob = support_prob.reshape(k_way, n_shot, -1)
    #print(support_prob.reshape(k_way, n_shot, -1).mean(dim=1))
    support_prob = support_prob.cpu().detach().numpy()
    mu_stds = []
    for i in range(k_way):
        pos_mu, pos_std = fit(support_prob[i,:, i])
        mu_stds.append([pos_mu, pos_std])

    return mu_stds
