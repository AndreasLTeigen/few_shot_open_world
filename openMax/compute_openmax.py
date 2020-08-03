import numpy as np
import matplotlib.pyplot as plt


def inverse_activation_function(input):
    a = 10
    b = 0
    c = 1
    x = input/21.9                                                  #21.9 is the mean distance from queries to all prototypes for unknown classes for the trained classifier for the whas dataset
    gaussian_function = a*np.exp(-(x-b)*(x-b)/(2*c))
    return gaussian_function

def computeOpenMaxProbability(openmax_embeddingLayer, openmax_score_u):

    print('openmax_embeddingLayer')
    print(openmax_embeddingLayer)
    #print(np.array(openmax_embeddingLayer).shape)
    print('openmax_score_u')
    print(openmax_score_u)
    #print(np.array(openmax_score_u).shape)
    #print(np.sum(openmax_score_u))

    class_scores = np.exp(openmax_embeddingLayer)
    total_denominator = np.sum(np.exp(openmax_embeddingLayer)) + np.exp(np.sum(openmax_score_u))
    prob_scores = class_scores/total_denominator
    prob_unknowns = np.exp(np.sum(openmax_score_u))/total_denominator

    print('total_denominator')
    print(total_denominator)
    #print('prob_scores')
    #print(prob_scores)
    #print('prob_unknowns')
    #print(prob_unknowns)
    print('sum')
    print(np.sum(prob_scores) + prob_unknowns)

    return prob_scores, prob_unknowns


def recalibrate_scores(weibull_model, query2prototype_distance, alpharank=3, distance_type='l2'):
    num_classes = len(weibull_model)

    ranked_list = (np.amax(query2prototype_distance)/query2prototype_distance).argsort()[::-1]
    alpha_weights = [((alpharank+1) - i)/float(alpharank) for i in range(1, alpharank+1)]
    ranked_alpha = np.ones(num_classes)
    activation_vector = inverse_activation_function(query2prototype_distance)
    #for i in range(len(alpha_weights)):
    #    ranked_alpha[ranked_list[i]] = alpha_weights[i]

    print('ranked_alpha')
    print(ranked_alpha)

    openmax_embeddingLayer = []
    openmax_score_u = []

    print('Distance')
    print(query2prototype_distance)
    print('Activation vector')
    print(activation_vector)

    for class_id in range(num_classes):
        class_weibull = weibull_model[class_id]

        w_score = class_weibull['weibull_model'].w_score(query2prototype_distance[class_id])
        #modified_fc8_score = (np.amax(query2prototype_distance)/query2prototype_distance[class_id])*( 1 - w_score*ranked_alpha[class_id] )
        modified_fc8_score = (activation_vector[class_id])*( 1 - w_score*ranked_alpha[class_id] )
        openmax_embeddingLayer += [modified_fc8_score]
        openmax_score_u += [activation_vector[class_id] - modified_fc8_score]
        #openmax_score_u += [np.amax(query2prototype_distance)/query2prototype_distance[class_id] - modified_fc8_score]

        #print('Class ID')
        #print(class_id)
    print('w_score')
        #print(w_score)

    print([weibull_model[class_id]['weibull_model'].w_score(query2prototype_distance[class_id]) for class_id in range(num_classes)])

    #openmax_score_u += [-query2prototype_distance[np.argmax(openmax_embeddingLayer)] + modified_fc8_score]

    openmax_class_prob, openmax_unknown_prob = computeOpenMaxProbability(openmax_embeddingLayer, openmax_score_u)

    print('openMax scores: ')
    print(openmax_class_prob, openmax_unknown_prob)

    return openmax_class_prob, openmax_unknown_prob



def unknown_prob_histogram(unknown_probs, unknown_probs_open_world):
    print("HEHERHERHEHRHEHR")
    print(unknown_probs)
    print('DSUAIHJODIASJ')
    print(unknown_probs_open_world)
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist([unknown_probs, unknown_probs_open_world], color=['b', 'g'])
    #n, bins, patches = ax.hist(unknown_probs_open_world, color='g')
    ax.set_title('Unknown probs')
    plt.show()
