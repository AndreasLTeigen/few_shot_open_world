import numpy as np

try:
    import libmr
except ImportError:
    print("LibMr not installed or libmr.so not found")
    sys.exit()

def weibull_tailfitting(prototypes, distances, tailsize=5, distance_type='l2'):
    weibull_model = {}
    for class_nr in range(len(prototypes)):
        weibull_model[class_nr] = {}

        weibull_model[class_nr]['distances_' + str(distance_type)] = distances[class_nr]
        weibull_model[class_nr]['prototype'] = prototypes[class_nr]

        mr = libmr.MR()
        #print(distances)
        tailtofit = sorted(distances[class_nr])[-tailsize:]
        mr.fit_high(tailtofit, len(tailtofit))
        #mr.fit_low(tailtofit, len(tailtofit))

        weibull_model[class_nr]['weibull_model'] = mr

    return weibull_model


def np_pairwise_distances(prototype, support, distance_type='l2'):
    if distance_type=='l2':
        distances = support - prototype
        distances = np.power(distances, 2)
        distances = distances.sum(axis=1)
    else:
        print('Error: distance type not supported')
        print('Location: np_pairwise_distances')
        sys.exit()

    #print('pairwise distances: ')
    #print(distances)
    return distances
