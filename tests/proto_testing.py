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

from config import PATH
from few_shot.utils import pairwise_distances
from few_shot.core import NShotTaskSampler, prepare_nshot_task
from few_shot.datasets import Whoas, Kaggle, MiniImageNet
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

        if dataset == 'whoas':
            self.evaluation_dataset = Whoas('evaluation')
            #self.evaluation_dataset = Whoas('background')
        elif dataset == 'kaggle':
            self.evaluation_dataset = Kaggle('evaluation')
        elif dataset == 'miniImageNet':
            self.evaluation_dataset = MiniImageNet('evaluation')
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
experiment.load_model(PATH + '/models/proto_nets/'+ args.dataset + '_nt=5_kt=10_qt=15_nv=5_kv=5_qv=1.pth')



num_batches = 500
unknown_probs = []
unknown_probs_open_world = []
total_correct = 0
total_correct = 0
total_correct = 0
total_correct = 0
openMax_total_correct = 0
total_predicitons = 0
all_distances = []

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


    task_support_classes = list(dict.fromkeys(experiment.batch_sampler.support_df_last_task))
    task_queries = experiment.batch_sampler.query_df_last_task
    print(task_support_classes)
    print(task_queries)

    all_distances.append(distances)

    #print('Prototypes: ', prototypes)


    correct, missclassifications = experiment.classification_accuracy(y_pred)




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

plot_hist=True
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


print('Total accuracy')
print(total_correct/total_predicitons)

all_distances = np.array(all_distances).ravel()
print('Mean distances')
print(all_distances.mean())
print('distance variance')
print(all_distances.var())


plt.show()
