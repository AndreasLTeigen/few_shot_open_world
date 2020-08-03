import io
import torch
from torch.utils.data import DataLoader, Sampler
from torch.optim import Adam
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Iterable, Callable, Tuple
import numpy as np
from PIL import Image

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lmdd import LMDD
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.loci import LOCI
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.sod import SOD
from pyod.models.sos import SOS
#from pyod.models.xgbod import XGBOD



from config import PATH
from few_shot.utils import pairwise_distances
from few_shot.core import NShotTaskSampler, prepare_nshot_task
from few_shot.datasets import Whoas
from few_shot.proto import compute_prototypes, proto_net_episode
from few_shot.models import get_few_shot_encoder
#from proto_testing_utils import TSNEAlgo

from few_shot.metrics import categorical_accuracy

def normalize_tensor(tensor):
    tensor = tensor / tensor.sum(0).expand_as(tensor)
    return tensor

def get_distance(x1, x2, distance_metric):
    if distance_metric == 'l2':
        return (x1*x1 + x2*x2)
    else:
        raise(ValueError, 'Unsuported distance_metric')

def tensorlist_to_numpy(tensor_list):
    list = []
    for tensor in tensor_list:
        list.append(tensor.numpy())
    np_list = np.array(list)
    return np_list


def get_distance(embedding1, embedding2):
    distance = np.add(embedding1,-embedding2)
    distance = np.power(distance, 2)
    return distance

def get_distance_mappings(embeddings):
    distances = []
    for embedding1 in embeddings:
        for embedding2 in embeddings:
            d_m = get_distance(embedding1, embedding2)
            distances.append(d_m)
    distances = np.array(distances)
    return distances

class NShotCustomTaskSampler(Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 episodes_per_epoch: int = None,
                 n: int = None,
                 k: int = None,
                 q: int = None,
                 num_tasks: int = 1,
                 fixed_tasks: List[Iterable[int]] = None,
                 open_world_testing: bool = False):

        super(NShotCustomTaskSampler, self).__init__(dataset)
        self.task_counter = 0
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset
        if num_tasks < 1:
            raise ValueError('num_tasks must be > 1.')

        self.num_tasks = num_tasks
        # TODO: Raise errors if initialise badly
        self.k = k
        self.n = n
        self.q = q
        self.fixed_tasks = fixed_tasks
        self.open_world_testing = open_world_testing
        self.i_task = 0

        self.support_df_last_task = None
        self.support_df_last_task_filepath = None
        self.query_df_last_task = None
        self.query_df_last_task_filepath = None

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            self.support_df_last_task = []
            self.support_df_last_task_filepath = []
            self.query_df_last_task = []
            self.query_df_last_task_filepath = []
            batch = []
            print('task_counter')
            print(self.task_counter)

            for task in range(self.num_tasks):
                if self.fixed_tasks is None:
                    # Get random classes
                    episode_classes = np.random.choice(self.dataset.df['class_id'].unique(), size=self.k, replace=False)
                    if self.open_world_testing and self.task_counter%2:
                        open_world_classes = np.random.choice(self.dataset.df[~self.dataset.df['class_id'].isin(episode_classes)]['class_id'].unique(), size=self.k, replace=False)
                        #df[df.line_race != 0]
                else:
                    # Loop through classes in fixed_tasks
                    episode_classes = self.fixed_tasks[self.i_task % len(self.fixed_tasks)]
                    self.i_task += 1

                df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]

                if self.open_world_testing and self.task_counter%2:
                    open_world_df = self.dataset.df[self.dataset.df['class_id'].isin(open_world_classes)]

                support_k = {k: None for k in episode_classes}
                for k in episode_classes:
                    # Select support examples
                    support = df[df['class_id'] == k].sample(self.n)
                    support_k[k] = support

                    for i, s in support.iterrows():
                        batch.append(s['id'])
                        self.support_df_last_task.append(s.get('class_name'))
                        self.support_df_last_task_filepath.append(s.get('filepath'))

                #print(episode_classes)
                for i in range(len(episode_classes)):
                    if self.open_world_testing and self.task_counter%2 == 1 :
                        k = open_world_classes[i]
                        #print(self.dataset.df[~self.dataset.df['class_id'].isin(episode_classes)])
                        query = open_world_df[(open_world_df['class_id'] == k)].sample(self.q)
                    else:
                        k = episode_classes[i]
                        query = df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]['id']))].sample(self.q)
                    for i, q in query.iterrows():
                        batch.append(q['id'])
                        self.query_df_last_task.append(q.get('class_name'))
                        self.query_df_last_task_filepath.append(q.get('filepath'))


            self.task_counter += 1

            yield np.stack(batch)

class ResultExperimentation():

    def __init__(self,
                 dataset: str,
                 num_tasks: int,
                 n_shot: int,
                 k_way: int,
                 q_queries: int,
                 distance_metric: str,
                 open_world_testing: bool
                 ):
        self.num_tasks = num_tasks
        self.n_shot = n_shot
        self.k_way = k_way
        self.q_queries = q_queries
        self.episodes_per_epoch=10
        self.distance_metric='l2'
        self.open_world_testing = open_world_testing
        self.prepare_batch = prepare_nshot_task(self.n_shot, self.k_way, self.q_queries)

        self.num_different_models = 0

        if dataset == 'whoas':
            self.evaluation_dataset = Whoas('evaluation')
            #self.evaluation_dataset = Whoas('background')
        else:
            raise(ValueError, 'Unsupported dataset')

        self.batch_sampler = NShotCustomTaskSampler(self.evaluation_dataset,self.episodes_per_epoch, n_shot, k_way, q_queries, num_tasks, None, open_world_testing)
        self.evaluation_taskloader = DataLoader(
            self.evaluation_dataset,
            batch_sampler=self.batch_sampler
        )

        assert torch.cuda.is_available()
        self.device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True

        self.model = get_few_shot_encoder(self.evaluation_dataset.num_input_channels)
        self.model.to(self.device, dtype=torch.double)

        self.optimiser = Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = torch.nn.NLLLoss().cuda()

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def get_batch(self, taskloader):
        for batch_index, batch in enumerate(taskloader):
            x,y = batch
            break
        x, y = self.prepare_batch(batch)
        return x,y

    def evaluate_batch(self,x,y):
        embeddings = self.model(x)
        support = embeddings[:self.n_shot*self.k_way]
        queries = embeddings[self.n_shot*self.k_way:]
        prototypes = compute_prototypes(support, self.k_way, self.n_shot)
        distances = pairwise_distances(queries, prototypes, self.distance_metric)
        log_p_y = (-distances).log_softmax(dim=1)
        loss = self.loss_fn(log_p_y, y)
        y_pred = (-distances).softmax(dim=1)
        return loss, y_pred, distances, prototypes, support, queries

    def initialise_pyod_classifiers(self, outlier_fraction):
        classifiers = {}
        #Proximity based
        classifiers['K Nearest Neighbors (KNN)'] = []
        classifiers['Average K Nearest Neighbors (AvgKNN)'] = []
        classifiers['Median K Nearest Neighbors (MedKNN)'] = []
        classifiers['Local Outlier Factor (LOF)'] = []
        classifiers['Connectivity-Based Outlier Factor (COF)'] = []
        #classifiers['Clustering-Based Local Outlier Factor (CBLOF)'] = []
        classifiers['LOCI'] = []
        #classifiers['Histogram-based Outlier Score (HBOS)'] = []
        classifiers['Subspace Outlier Detection (SOD)'] = []
        #Linear models
        classifiers['Principal Component Analysis (PCA)'] = []
        #classifiers['Minimum Covariance Determinant (MCD)'] = []           #To slow
        classifiers['One-Class Support Vector Machines (OCSVM)'] = []
        classifiers['Deviation-based Outlier Detection (LMDD)'] = []
        #Probabilistic
        classifiers['Angle-Based Outlier Detection (ABOD)'] = []
        classifiers['Stochastic Outlier Selection (SOS)'] = []
        #Outlier Ensembles
        classifiers['Isolation Forest (IForest)'] = []
        classifiers['Feature Bagging'] = []
        #classifiers['Extreme Boosting Based Outlier Detector (XGBOD)'] = []
        classifiers['Lightweight On-line Detector of Anomalies (LODA)'] = []
        for i in range(self.k_way):
            for i in range(self.k_way):
                classifiers['K Nearest Neighbors (KNN)'].append(KNN(method='largest', n_neighbors=int(self.n_shot/3)+1,contamination=outlier_fraction))
                classifiers['Average K Nearest Neighbors (AvgKNN)'].append(KNN(method='mean', n_neighbors=int(self.n_shot/3)+1,contamination=outlier_fraction))
                classifiers['Median K Nearest Neighbors (MedKNN)'].append(KNN(method='median', n_neighbors=int(self.n_shot/3)+1,contamination=outlier_fraction))
                classifiers['Local Outlier Factor (LOF)'].append(LOF(n_neighbors=int(self.n_shot/3)+1,contamination=outlier_fraction))
                classifiers['Connectivity-Based Outlier Factor (COF)'].append(COF(n_neighbors=int(self.n_shot/3)+1,contamination=outlier_fraction))
                #classifiers['Clustering-Based Local Outlier Factor (CBLOF)'].append(CBLOF(n_clusters=5,contamination=outlier_fraction))
                classifiers['LOCI'].append(LOCI(contamination=outlier_fraction))
                #classifiers['Histogram-based Outlier Score (HBOS)'].append(CBLOF(contamination=outlier_fraction))
                classifiers['Subspace Outlier Detection (SOD)'].append(COF(n_neighbors=int(self.n_shot/3)+1,contamination=outlier_fraction))
                classifiers['Principal Component Analysis (PCA)'].append(PCA(contamination=outlier_fraction))
                #classifiers['Minimum Covariance Determinant (MCD)'].append(MCD(contamination=outlier_fraction))
                classifiers['One-Class Support Vector Machines (OCSVM)'].append(OCSVM(contamination=outlier_fraction))
                classifiers['Deviation-based Outlier Detection (LMDD)'].append(LMDD(contamination=outlier_fraction))
                classifiers['Angle-Based Outlier Detection (ABOD)'].append(ABOD(contamination=outlier_fraction))
                classifiers['Stochastic Outlier Selection (SOS)'].append(SOS(contamination=outlier_fraction))
                classifiers['Isolation Forest (IForest)'].append(IForest(contamination=outlier_fraction))
                classifiers['Feature Bagging'].append(FeatureBagging(contamination=outlier_fraction))
                #classifiers['Extreme Boosting Based Outlier Detector (XGBOD)'].append(XGBOD())
                classifiers['Lightweight On-line Detector of Anomalies (LODA)'].append(LODA(contamination=outlier_fraction))
        self.num_different_models = len(classifiers)
        return classifiers

    def initialise_models_score(self, classifiers):
        models_score = {}
        for i, (model_name, model_clf) in enumerate(classifiers.items()):
            models_score[model_name] = {}
            models_score[model_name]['confusion_matrix'] ={
                                                          "true_positive": 0,
                                                          "true_negative": 0,
                                                          "false_positive": 0,
                                                          "false_negative": 0
                                                        }
            models_score[model_name]['openworld_correct_classification'] = 0
        return models_score


    def train_outlier_classifiers(self, classifiers, support):
        for i, (clf_name, clf_models) in enumerate(classifiers.items()):
            class_models = []
            print(clf_name)
            for j, class_support in enumerate(support):
                clf_models[j].fit(class_support)
                #print(clf_models[j].predict(class_support))
            '''
            print("DIVIDE")
            for j, class_j_support in enumerate(support):
                class_j_model = clf_models[j]
                #class_j_support = support[j]
                #print(class_j_support)
                print(class_j_model.predict(class_j_support))
            '''
        return classifiers

    def get_outliers(self, models, support, queries):
        model_outliers = {}
        for i, (model_name, class_models) in enumerate(models.items()):
            outliers = []
            print('OUTLIER PREDICTION')
            print(model_name)
            for i in range(self.k_way):
                query = queries[i]
                query_outliers = []
                for j in range(self.k_way):
                    class_support = support[j]
                    clf_test_data = np.append(class_support,[query], axis=0)
                    query_outliers.append(class_models[j].predict(clf_test_data)[-1])
                    print(class_models[j].predict(clf_test_data))
                outliers.append(query_outliers)
            model_outliers[model_name] = outliers
        return model_outliers





    def classification_accuracy(self, predictions):
        correct = 0
        missclassifications =[]
        for i in range(len(predictions)):
            prediction = predictions[i]
            print(i%self.k_way)
            print(prediction)
            if np.argmax(prediction)==(i%self.k_way):
                 correct+=1
            else:
                missclassifications.append(i)
        return correct, missclassifications

    def get_performance_measures(self, model_score):
        TP = model_score['confusion_matrix']['true_positive']
        FP = model_score['confusion_matrix']['false_positive']
        TN = model_score['confusion_matrix']['true_negative']
        FN = model_score['confusion_matrix']['false_negative']
        accuracy = (TP + TN)/(TP + TN + FP + FN)
        if TP+FP == 0:
            precision = 0
        else:
            precision = (TP)/(TP+FP)
        if (TP+FN) == 0:
            recall = 0
        else:
            recall = (TP)/(TP+FN)
        if (recall+precision) == 0:
            f1 = 0
        else:
            f1 = 2*(recall*precision)/(recall+precision)
        return accuracy, precision, recall, f1

    def get_correct_incorrect_probs(self, predictions):
        correct_probs = []
        incorrect_probs = []
        for i in range(len(predictions)):
            prediction = predictions[i]
            if np.argmax(prediction)== i:
                correct_probs.append(prediction[np.argmax(prediction)])
            else:
                incorrect_probs.append(prediction[np.argmax(prediction)])
        return correct_probs, incorrect_probs

    def get_correct_incorrect_dist(self, predictions, distances):
        correct_dist = []
        incorrect_dist = []
        for i in range(len(predictions)):
            prediction = predictions[i]
            distance = distances[i]
            if np.argmax(prediction)== i:
                correct_dist.append(distance[np.argmax(prediction)])
            else:
                incorrect_dist.append(distance[np.argmax(prediction)])
        return correct_dist, incorrect_dist

    def get_dist_and_pred_distrubition(self, predictions, distances):
        correct_sol_pred = []
        incorrect_sol_pred = []
        correct_sol_dist = []
        incorrect_sol_dist = []

        for i in range(len(predictions)):
            prediction = predictions[i]
            distance = distances[i]
            for j in range(len(prediction)):
                if j == i:
                    correct_sol_pred.append(prediction[j])
                    correct_sol_dist.append(distance[j])
                else:
                    incorrect_sol_pred.append(prediction[j])
                    incorrect_sol_dist.append(distance[j])
        return correct_sol_pred, incorrect_sol_pred, correct_sol_dist, incorrect_sol_dist

    def get_closest_distance(self, distances):
        closest_distances = []
        for i in range(len(distances)):
            distance = distances[i]
            closest_distances.append(np.argmin(distance))
        return closest_distances

    def update_open_world_performance_metrics(self, models_score, predictions, outlier_models, batch_nr):
        print(outlier_models)
        for i, (model_name, outliers) in enumerate(outlier_models.items()):

            for i in range(len(predictions)):
                pred_class = np.argmax(predictions[i])
                true_positive = False
                true_negative = False
                if outliers[i][i] == 0 and not (self.open_world_testing and (batch_nr%2)):
                    true_positive = True
                    models_score[model_name]['confusion_matrix']['true_positive'] += 1
                elif outliers[i][i] == 1 and (self.open_world_testing and (batch_nr%2)):
                    true_negative = True
                    models_score[model_name]['confusion_matrix']['true_negative'] += 1
                elif outliers[i][i] == 0 and (self.open_world_testing and (batch_nr%2)):
                    models_score[model_name]['confusion_matrix']['false_positive'] += 1
                elif outliers[i][i] == 1 and not (self.open_world_testing and (batch_nr%2)):
                    models_score[model_name]['confusion_matrix']['false_negative'] += 1

                if (true_positive and pred_class == i) or true_negative:
                    models_score[model_name]['openworld_correct_classification'] += 1

            print([np.argmax(predictions[i]) for i in range(len(predictions))])

        return models_score




    def plot_scatterplot(self, prototypes, queries, support, axes):
        class_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x_prototypes = prototypes[:, [axes[0]]]
        y_prototypes = prototypes[:, [axes[1]]]
        z_prototypes = prototypes[:, [axes[2]]]
        ax.scatter(x_prototypes, y_prototypes, z_prototypes, c=class_colors[:self.k_way], marker='*')

        x_queries = queries[:, [axes[0]]]
        y_queries = queries[:, [axes[1]]]
        z_queries = queries[:, [axes[2]]]
        ax.scatter(x_queries, y_queries, z_queries, c=class_colors[:self.k_way], marker='^')

        x_support = support[:, [axes[0]]]
        y_support = support[:, [axes[1]]]
        z_support = support[:, [axes[2]]]
        support_color = []
        for i in range(self.k_way):
            for j in range(self.n_shot):
                support_color.append(class_colors[i])
        ax.scatter(x_support, y_support, z_support, c=support_color)

        legend_array = []
        for k in range(self.k_way):
            legend_array.append('Class: ' + str(k))

        #ax.legend(legend_array)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    def create_collage(self, y_pred, query_image_number, width=500, height=500):
        support_images = [Image.open(im) for im in self.batch_sampler.support_df_last_task_filepath]
        query_image = Image.open(self.batch_sampler.query_df_last_task_filepath[query_image_number])

        image_width = width//self.n_shot
        image_height = height//self.k_way
        size = image_width, image_height

        collage = Image.new('RGB', (width, height))
        ims = []
        for p in support_images:
            im = p
            im.thumbnail(size)
            ims.append(im)
        i = 0
        x = 0
        y = 0
        for row in range(self.n_shot):
            for col in range(self.k_way):
                collage.paste(ims[i], (x, y))
                i += 1
                x += image_width
            y += image_height
            x = 0

        collage.show()

        query_image.show()

    def plt_dual_histogram(self, list1, list2, xlabel='x', ylabel='y', name='Dual Histogram', labels=['list1, list2']):
        weights1 = np.ones(len(list1))/len(list1)
        weights2 = np.ones(len(list2))/len(list2)
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist([list1, list2], weights=[weights1, weights2], color=['b', 'g'], label=labels)
        #n, bins, patches = ax.hist(unknown_probs_open_world, color='g')
        labels = []
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ax.set_title(name)
        plt.show()

    def write_score_to_file(self, models_score):
        filename = 'pyod_num-batches-' + str(self.num_tasks) + '_n-test-' + str(self.n_shot) + '_k-test-' + str(self.k_way) +'.txt'
        f = open(filename, "w")
        for i, (model_name, model_score) in enumerate(models_score.items()):
            f.write('-----------------------------')
            f.write( "\n")
            f.write(model_name +': ')
            f.write( "\n")
            f.write('Total open world accuracy')
            f.write( "\n")
            f.write(str(model_score['openworld_correct_classification']/total_predicitons))
            f.write( "\n")
            f.write('Total outlier classifier confusion matrix')
            f.write( "\n")
            f.write(str(model_score['confusion_matrix']))
            f.write( "\n")
            outlier_accuracy, outlier_precision, outlier_recall, outlier_f1 = self.get_performance_measures(model_score)
            f.write('Outlier classifier accuracy: ' + str(outlier_accuracy))
            f.write( "\n")
            f.write('Outlier classifier presicion: ' + str(outlier_precision))
            f.write( "\n")
            f.write('Outlier classifier recall: ' + str(outlier_recall))
            f.write( "\n")
            f.write('Outlier classifier F1: ' + str(outlier_f1))
            f.write( "\n")
        f.close()



parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--distance', default='l2')
parser.add_argument('--n-test', default=5, type=int)
parser.add_argument('--k-test', default=5, type=int)
parser.add_argument('--q-test', default=1, type=int)
parser.add_argument('--num-tasks', default=1, type=int)
parser.add_argument('--open-world', default=False, type=bool)
args = parser.parse_args()

print('open_world_testing')
print(args.open_world)

experiment = ResultExperimentation(args.dataset, args.num_tasks, args.n_test, args.k_test, args.q_test, args.distance, args.open_world)
experiment.load_model(PATH + '/models/proto_nets/whoas_97-9.pth')



num_batches = 1000
unknown_probs = []
unknown_probs_open_world = []
total_correct = 0
total_correct = 0
total_correct = 0
total_correct = 0
openMax_total_correct = 0
total_predicitons = 0
all_distances = []
outlier_fraction = 0.0001

correct_probs = []
incorrect_probs = []
correct_dist = []
incorrect_dist = []
correct_probs_openworld = []
incorrect_probs_openworld = []
correct_dist_openworld = []
incorrect_dist_openworld = []
closest_distances = []
closest_distances_openworld = []

classifiers = experiment.initialise_pyod_classifiers(0.0001)
models_score = experiment.initialise_models_score(classifiers)

for batch_nr in range(num_batches):
    x, y = experiment.get_batch(experiment.evaluation_taskloader)
    loss, y_pred, distances, prototypes, support, queries = experiment.evaluate_batch(x,y)


    prototypes = prototypes.cpu().detach().numpy()
    support = support.cpu().detach().numpy()
    queries = queries.cpu().detach().numpy()
    distances=distances.cpu().detach().numpy()
    y_pred=y_pred.cpu().detach().numpy()
    y=y.cpu().detach().numpy()

    #print('Distances: ', distances[0])
    #print('Predictions: ', y_pred[0])
    #print('Distances full: ', distances)
    #print('Queries lenght')
    #print(len(queries))

    support = support.reshape(args.k_test, args.n_test, -1)
    outlier_models = experiment.train_outlier_classifiers(classifiers, support)
    model_outliers = experiment.get_outliers(outlier_models, support, queries)


    task_support_classes = list(dict.fromkeys(experiment.batch_sampler.support_df_last_task))
    task_queries = experiment.batch_sampler.query_df_last_task
    print(task_support_classes)
    print(task_queries)

    all_distances.append(distances)

    #print('Prototypes: ', prototypes)


    correct, missclassifications = experiment.classification_accuracy(y_pred)
    models_score = experiment.update_open_world_performance_metrics(models_score, y_pred, model_outliers, batch_nr)
    print(models_score)




    if args.open_world and batch_nr%2 == 1:
        new_correct_sol_probs_openworld, new_incorrect_sol_probs_openworld, new_correct_sol_dist_openworld, new_incorrect_sol_dist_openworld = experiment.get_dist_and_pred_distrubition(y_pred, distances)

        closest_distances_openworld += experiment.get_closest_distance(distances)
        if new_correct_sol_probs_openworld:
            correct_probs_openworld += new_correct_sol_probs_openworld
            correct_dist_openworld += new_correct_sol_dist_openworld
        if new_incorrect_sol_probs_openworld:
            incorrect_probs_openworld += new_incorrect_sol_probs_openworld
            incorrect_dist_openworld += new_incorrect_sol_dist_openworld
    else:
        closest_distances += experiment.get_closest_distance(distances)
        new_correct_probs, new_incorrect_probs = experiment.get_correct_incorrect_probs(y_pred)
        new_correct_dist, new_incorrect_dist = experiment.get_correct_incorrect_dist(y_pred, distances)

        if new_correct_probs:
            correct_probs += new_correct_probs
            correct_dist += new_correct_dist
        if new_incorrect_probs:
            incorrect_probs += new_incorrect_probs
            incorrect_dist += new_incorrect_dist




    total_correct += correct
    total_predicitons += len(y_pred)
    accuracy = correct/len(y_pred)
    print(accuracy)

    if not missclassifications:
        print_case = 0
    else:
        print_case = missclassifications[0]

    #prototype_distances = get_distance_mappings(prototypes)
    #mean_distances = prototype_distances.mean(axis=0)
    #most_significant_axes = np.argpartition(mean_distances, -3)[-3:]
    #experiment.plot_scatterplot(prototypes, queries, support, most_significant_axes)

    #TSNE = TSNEAlgo(experiment.k_way, experiment.n_shot)
    #TSNE.tsne_fit(support, experiment.batch_sampler.support_df_last_task, title='TSNE')
    print('-------------------------------------------------------------------------------------------------------------------------------------------------')


    plot=False
    if plot==True:
        experiment.create_collage(y_pred[print_case], print_case)

        plot1 = plt.figure(1)
        plot1.suptitle('Distances: Correct class: '+str(print_case))
        plt.barh(y, distances[print_case])
        plt.gca().invert_yaxis()

        plot2 = plt.figure(2)
        plot2.suptitle('Predictions: Correct class: '+str(print_case))
        plt.barh(y,y_pred[print_case])
        plt.gca().invert_yaxis()

plot_hist=False
if plot_hist==True:
    #if args.open_world:
    #    unknown_prob_histogram(unknown_probs, unknown_probs_open_world)

    correct_probs = np.array(correct_probs).ravel()
    incorrect_probs = np.array(incorrect_probs).ravel()
    experiment.plt_dual_histogram(correct_probs, incorrect_probs, xlabel='certainty', ylabel='percentage distribution', name='Probability distribution', labels=['correct_pedictions', 'incorrect_predicitons'])

    correct_dist = np.array(correct_dist).ravel()
    incorrect_dist = np.array(incorrect_dist).ravel()
    experiment.plt_dual_histogram(correct_dist, incorrect_dist, xlabel='distance', ylabel='percentage distribution', name='Distance distribution', labels=['correct_pedictions', 'incorrect_predicitons'])

    if args.open_world:
        closest_distances = np.array(closest_distances).ravel()
        closest_distances_openworld = np.array(closest_distances_openworld).ravel()
        experiment.plt_dual_histogram(closest_distances, closest_distances_openworld, xlabel='distance', ylabel='percentage distribution', name='Closest distance distribution', labels=['closed_world', 'open_world'])

        '''
        correct_probs_openworld = np.array(correct_probs_openworld).ravel()
        incorrect_probs_openworld = np.array(incorrect_probs_openworld).ravel()
        experiment.plt_dual_histogram(correct_probs_openworld, incorrect_probs_openworld, xlabel='certainty', ylabel='percentage distribution', name='Probability distribution Openworld', labels=['correct_pedictions', 'incorrect_predicitons'])

        correct_dist_openworld = np.array(correct_dist_openworld).ravel()
        incorrect_dist_openworld = np.array(incorrect_dist_openworld).ravel()
        experiment.plt_dual_histogram(correct_dist_openworld, incorrect_dist_openworld, xlabel='distance', ylabel='percentage distribution', name='Distance distribution Openworld', labels=['correct_pedictions', 'incorrect_predicitons'])
        '''


print('Total closed world accuracy')
print(total_correct/total_predicitons)
for i, (model_name, model_score) in enumerate(models_score.items()):
    print('-----------------------------')
    print(model_name,': ')
    print('Total open world accuracy')
    print(model_score['openworld_correct_classification']/total_predicitons)
    print('Total outlier classifier confusion matrix')
    print(model_score['confusion_matrix'])
    outlier_accuracy, outlier_precision, outlier_recall, outlier_f1 = experiment.get_performance_measures(model_score)
    print('Outlier classifier accuracy', outlier_accuracy)
    print('Outlier classifier presicion', outlier_precision)
    print('Outlier classifier recall', outlier_recall)
    print('Outlier classifier F1', outlier_f1)

all_distances = np.array(all_distances).ravel()
print('Mean distances')
print(all_distances.mean())
print('distance variance')
print(all_distances.var())

experiment.write_score_to_file(models_score)

plt.show()
