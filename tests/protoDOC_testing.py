import io
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler
from torch.optim import Adam
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Iterable, Callable, Tuple
import numpy as np
import time
from PIL import Image

from config import PATH
from DOC.doc_utils import get_class_fit
from few_shot.utils import pairwise_distances
from few_shot.core import NShotTaskSampler, prepare_nshot_task
from few_shot.datasets import Whoas, Kaggle
from few_shot.proto import compute_prototypes
from few_shot.models import get_few_shot_encoder
from openMax.evt_fitting import weibull_tailfitting, np_pairwise_distances
from openMax.compute_openmax import recalibrate_scores, unknown_prob_histogram
#from proto_testing_utils import TSNEAlgo

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
        self.dataset_name = dataset
        self.num_tasks = num_tasks
        self.n_shot = n_shot
        self.k_way = k_way
        self.q_queries = q_queries
        self.episodes_per_epoch=10
        self.distance_metric='l2'
        self.open_world_testing = open_world_testing
        self.prepare_batch = prepare_nshot_task(self.n_shot, self.k_way, self.q_queries)

        if dataset == 'whoas':
            self.evaluation_dataset = Whoas('evaluation')
        elif dataset == 'kaggle':
            self.evaluation_dataset = Kaggle('evaluation')
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
        self.loss_fn = torch.nn.BCELoss().cuda()

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
        pred_classes = []
        embeddings = self.model(x)
        support = embeddings[:self.n_shot*self.k_way]
        queries = embeddings[self.n_shot*self.k_way:]
        prototypes = compute_prototypes(support, self.k_way, self.n_shot)
        distances = pairwise_distances(queries, prototypes, self.distance_metric)
        start_time = time.time()
        mu_stds = get_class_fit(support, prototypes, self.k_way, self.n_shot, self.distance_metric)
        sigmoid = nn.Sigmoid()
        sigmoid_distances = sigmoid(-distances)
        for prediction in sigmoid_distances:
            max_class = torch.argmax(prediction).item()
            max_value = torch.max(prediction)
            threshold = max(0.25, 0.5 - mu_stds[max_class][1])
            if max_value > threshold:
                pred_classes.append(max_class)
            else:
                pred_classes.append(self.k_way)
        print('Time Used: ')
        print(1/(time.time()-start_time))
        y = torch.eye(self.k_way)[y].double().cuda()
        loss = self.loss_fn(sigmoid_distances, y)
        y_pred = sigmoid_distances
        return loss, y_pred, distances, prototypes, support, queries, pred_classes

    def classification_accuracy(self, predictions, pred_classes):
        correct_probabilites = 0
        correct_classifications = 0
        missclassifications =[]
        for i in range(len(predictions)):
            prediction = predictions[i]
            print(i%self.k_way)
            print(prediction)
            if np.argmax(prediction)==(i%self.k_way):
                 correct_probabilites+=1
            else:
                missclassifications.append(i)
            if pred_classes[i] == (i%self.k_way):
                correct_classifications += 1

        return correct_probabilites, correct_classifications, missclassifications

    def get_open_world_performance_metrics(self, pred_classes, batch_nr):
        openworld_classification_correct = 0
        confusion_matrix = {
          "true_positive": 0,
          "true_negative": 0,
          "false_positive": 0,
          "false_negative": 0
        }

        for i in range(len(pred_classes)):
            true_positive = False
            true_negative = False
            if pred_classes[i] != self.k_way and not (self.open_world_testing and (batch_nr%2)):
                true_positive = True
                confusion_matrix['true_positive'] += 1
            elif pred_classes[i] == self.k_way and (self.open_world_testing and (batch_nr%2)):
                true_negative = True
                confusion_matrix['true_negative'] += 1
            elif pred_classes[i] != self.k_way and (self.open_world_testing and (batch_nr%2)):
                confusion_matrix['false_positive'] += 1
            elif pred_classes[i] == self.k_way and not (self.open_world_testing and (batch_nr%2)):
                confusion_matrix['false_negative'] += 1

            if (true_positive and pred_classes[i] == i) or true_negative:
                openworld_classification_correct += 1



        return confusion_matrix, openworld_classification_correct

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

    def get_performance_measures(self, model_score):
        TP = model_score['true_positive']
        FP = model_score['false_positive']
        TN = model_score['true_negative']
        FN = model_score['false_negative']
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

    def write_score_to_file(self, model_score, openworld_correct_classification):
        filename = 'DOC_' + str(self.dataset_name) + '_num-batches-' + str(self.num_tasks) + '_n-test-' + str(self.n_shot) + '_k-test-' + str(self.k_way) +'.txt'
        f = open(filename, "a")
        f.write('-----------------------------')
        f.write( "\n")
        f.write('DOC: ')
        f.write( "\n")
        f.write('Total open world accuracy')
        f.write( "\n")
        f.write(str(openworld_correct_classification/total_predicitons))
        f.write( "\n")
        f.write('Total outlier classifier confusion matrix')
        f.write( "\n")
        f.write(str(model_score))
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

print('open_world_data')
print(args.open_world)

experiment = ResultExperimentation(args.dataset, args.num_tasks, args.n_test, args.k_test, args.q_test, args.distance, args.open_world)
experiment.load_model(PATH + '/models/protoDOC_nets/kaggle_nt=5_kt=10_qt=15_nv=5_kv=5_qv=1.pth')



num_batches = 1
unknown_probs = []
unknown_probs_open_world = []
total_correct_probabilities = 0
total_correct_classifications = 0
total_open_world_classification_correct = 0
total_predicitons = 0
total_confusion_matrix = {
          "true_positive": 0,
          "true_negative": 0,
          "false_positive": 0,
          "false_negative": 0
        }
for batch_nr in range(num_batches):
    x, y = experiment.get_batch(experiment.evaluation_taskloader)
    loss, y_pred, distances, prototypes, support, queries, pred_classes = experiment.evaluate_batch(x,y)
    print(pred_classes)



    prototypes = prototypes.cpu().detach().numpy()
    support = support.cpu().detach().numpy()
    queries = queries.cpu().detach().numpy()
    distances=distances.cpu().detach().numpy()
    y_pred=y_pred.cpu().detach().numpy()
    y=y.cpu().detach().numpy()

    print('Distances: ', distances[0])
    print('Predictions: ', y_pred)#[0])
    #print('Prototypes: ', prototypes)
    #print(prototypes.shape)
    #print('Distances full: ', distances)
    print('Queries lenght')
    print(len(queries))

    task_support_classes = list(dict.fromkeys(experiment.batch_sampler.support_df_last_task))
    task_queries = experiment.batch_sampler.query_df_last_task
    print(task_support_classes)
    print(task_queries)



    correct_probabilites, correct_classifications, missclassifications = experiment.classification_accuracy(y_pred, pred_classes)
    confusion_matrix, open_world_classification_correct = experiment.get_open_world_performance_metrics(pred_classes, batch_nr)



    total_correct_classifications += correct_classifications
    total_correct_probabilities += correct_probabilites
    total_open_world_classification_correct += open_world_classification_correct
    total_predicitons += len(y_pred)
    classification_accuracy = correct_classifications/len(y_pred)
    probability_accuracy = correct_probabilites/len(y_pred)
    openworld_classification_accuracy = open_world_classification_correct/len(y_pred)
    for value in confusion_matrix:
        total_confusion_matrix[value] += confusion_matrix[value]
    print('Closed world accuracy: ', probability_accuracy)
    print('Open world classifier accuracy: ', classification_accuracy)
    print('Openworld classification: ', openworld_classification_accuracy)
    print('confusion_matrix', confusion_matrix)

    if not missclassifications:
        print_case = 0
    else:
        print_case = missclassifications[0]

    prototype_distances = get_distance_mappings(prototypes)
    mean_distances = prototype_distances.mean(axis=0)
    most_significant_axes = np.argpartition(mean_distances, -3)[-3:]
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

#if args.open_world:
#    unknown_prob_histogram(unknown_probs, unknown_probs_open_world)

print('Total closed world accuracy: ')
print(total_correct_probabilities/total_predicitons)
print('Total open world classifier accuracy')
print(total_correct_classifications/total_predicitons)
print('Total open world accuracy')
print(total_open_world_classification_correct/total_predicitons)
print('Total confusion matrix')
print(total_confusion_matrix)
outlier_accuracy, outlier_precision, outlier_recall, outlier_f1 = experiment.get_performance_measures(total_confusion_matrix)
print('Outlier classifier accuracy', outlier_accuracy)
print('Outlier classifier presicion', outlier_precision)
print('Outlier classifier recall', outlier_recall)
print('Outlier classifier F1', outlier_f1)

experiment.write_score_to_file(total_confusion_matrix, total_open_world_classification_correct)

'''
DEFINITIONS
Classification accuracy: The number of correct classification including the new class
Probability accuracy: The number of correct classifications on probabilites excluding new class
'''

plt.show()
