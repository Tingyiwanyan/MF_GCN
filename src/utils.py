import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
import math
from model import model


class utils(model):
    def __init__(self):
        model.__init__(self)
        self.batch_size = 20
        self.attribute_size = 5
        self.walk_length = 8
        #self.mean_count = model.mean_count
        #self.mean_top = model.mean_top
    def assign_value(self, node_index):
        attribute_vector = np.zeros(5)
        """
        attribute_vector[0] = np.float(1/(1+np.exp(-(G.nodes[node_index]['ageDiversity']-mean_age))))
        attribute_vector[1] = np.float(1/(1+np.exp(-(G.nodes[node_index]['an_citation']-mean_an))))
        attribute_vector[2] = np.float(1/(1+np.exp(-(G.nodes[node_index]['citationDiversity']-mean_cit))))
        attribute_vector[3] = np.float(1/(1+np.exp(-(G.nodes[node_index]['publicationDiversity']-mean_pub))))
        attribute_vector[4] = np.float(1/(1+np.exp(-(G.nodes[node_index]['topicDiversity']-mean_top))))
        """
        """
        attribute_vector[0] = np.float((G.nodes[node_index]['p_year']-min_year)/(max_year-min_year))
        attribute_vector[1] = np.float((G.nodes[node_index]['citation']-min_cit)/(max_cit-min_cit))
        attribute_vector[2] = np.float((G.nodes[node_index]['avg_citation']-min_avg)/(max_avg-min_avg))
        attribute_vector[3] = np.float((G.nodes[node_index]['country_diversity']-min_count)/(max_count-min_count))
        attribute_vector[4] = np.float((G.nodes[node_index]['topic_diversity']-min_top)/(max_top-min_top))
        attribute_vector[5] = np.float((G.nodes[node_index]['productivity_diversity']-min_prod)/(max_prod-min_prod))
        attribute_vector[6] = np.float((G.nodes[node_index]['impact_diversity']-min_impact)/(max_impact-min_impact))
        attribute_vector[7] = np.float((G.nodes[node_index]['scientific_age_diversity']-min_sci)/(max_sci-min_sci))
        """
        """
        if np.float(G.nodes[node_index]['p_year']) > std_max_year:
          attribute_vector[0] = 1
        elif np.float(G.nodes[node_index]['p_year']) < std_min_year:
          attribute_vector[0] = 0
        else:
          attribute_vector[0] = (np.float(G.nodes[node_index]['p_year'])-std_min_year)/(2*std_year)
    
        if np.float(G.nodes[node_index]['citation']) > std_max_cit:
          attribute_vector[1] = 1
        elif np.float(G.nodes[node_index]['citation']) < std_min_cit:
          attribute_vector[1] = 0
        else:
          attribute_vector[1] = (np.float(G.nodes[node_index]['citation'])-std_min_cit)/(2*std_cit)
    
        if np.float(G.nodes[node_index]['avg_citation']) > std_max_avg:
          attribute_vector[2] = 1
        elif np.float(G.nodes[node_index]['avg_citation']) < std_min_avg:
          attribute_vector[2] = 0
        else:
          attribute_vector[2] = (np.float(G.nodes[node_index]['avg_citation'])-std_min_avg)/(2*std_avg)
        """

        # if np.float(G.nodes[node_index]['country_diversity']) > std_max_count:
        # attribute_vector[0] = 1
        # elif np.float(G.nodes[node_index]['country_diversity']) < std_min_count:
        # attribute_vector[0] = 0
        # else:
        attribute_vector[0] = (np.float(self.G.node[node_index]['country_diversity']) - self.mean_count) / self.std_count

        # if np.float(G.nodes[node_index]['topic_diversity']) > std_max_top:
        # attribute_vector[4] = 1
        # elif np.float(G.nodes[node_index]['topic_diversity']) < std_min_top:
        # attribute_vector[4] = 0
        # else:
        attribute_vector[1] = (np.float(self.G.node[node_index]['topic_diversity']) - self.mean_top) / self.std_top

        # if np.float(G.nodes[node_index]['productivity_diversity']) > std_max_prod:
        # attribute_vector[5] = 1
        # elif np.float(G.nodes[node_index]['productivity_diversity']) < std_min_prod:
        #  attribute_vector[5] = 0
        # else:
        attribute_vector[2] = (np.float(self.G.node[node_index]['productivity_diversity']) - self.mean_prod) / self.std_prod

        # if np.float(G.nodes[node_index]['impact_diversity']) > std_max_impact:
        # attribute_vector[6] = 1
        # elif np.float(G.nodes[node_index]['impact_diversity']) < std_min_impact:
        # attribute_vector[6] = 0
        # else:
        attribute_vector[3] = (np.float(self.G.node[node_index]['impact_diversity']) - self.mean_impact) / self.std_impact

        # if np.float(G.nodes[node_index]['scientific_age_diversity']) > std_max_sci:
        # attribute_vector[7] = 1
        # elif np.float(G.nodes[node_index]['scientific_age_diversity']) < std_min_sci:
        # attribute_vector[7] = 0
        # else:
        attribute_vector[4] = (np.float(self.G.node[node_index]['scientific_age_diversity']) - self.mean_sci) / self.std_sci

        return attribute_vector


    """
    mean_pooling skip_gram
    """


    def mean_pooling(self, skip_gram_vector):
        attribute_vector_total = np.zeros(self.attribute_size)
        for index in skip_gram_vector:
            attribute_vector_total += self.assign_value(index)

        return attribute_vector_total / self.walk_length


    """
    Get Neighborhood data
    """


    def get_neighborhood_data(self, node_index):
        neighbors = []
        for g in self.G.neighbors(node_index):
            neighbors.append(np.int(g))

        return neighbors, len(neighbors)


    """
    Get neighborhood from new data
    """


    def find_neighbor(self, node_index):
        author_id = self.G.nodes[node_index]['author_id']
        author_id2 = self.G.nodes[node_index]['author_id2']
        if author_id2 == 0:
            author_id2_v = -1
        else:
            author_id2_v = author_id2
        author_id3 = self.G.nodes[node_index]['author_id3']
        if author_id3 == 0:
            author_id3_v = -1
        else:
            author_id3_v = author_id3
        neighbor = [x for x, y in self.G.nodes(data=True) if
                    y['author_id'] == author_id or y['author_id'] == author_id2_v or y['author_id'] == author_id3_v or
                    y['author_id2'] == author_id or y['author_id2'] == author_id2_v or y['author_id2'] == author_id3_v or
                    y['author_id3'] == author_id or y['author_id3'] == author_id2_v or y['author_id3'] == author_id3_v]
        size = len(neighbor)
        return neighbor, size

    """
    BFS search for nodes
    """


    def BFS_search(self, start_node):
        walk_ = []
        visited = [start_node]
        BFS_queue = [start_node]
       # neighborhood = get_neighborhood_data
        while len(walk_) < self.walk_length:
            cur = np.int(BFS_queue.pop(0))
            walk_.append(cur)
            cur_nbrs = sorted(self.get_neighborhood_data(cur)[0])
            for node_bfs in cur_nbrs:
                if not node_bfs in visited:
                    BFS_queue.append(node_bfs)
                    visited.append(node_bfs)
            if len(BFS_queue) == 0:
                visited = [start_node]
                BFS_queue = [start_node]
        return walk_


    """
    compute average for one neighborhood node
    """


    def average_neighborhood(self, node_index, center_neighbor_size):
        neighbor_vec = self.assign_value(node_index)
        neighbor, neighbor_size = self.get_neighborhood_data(node_index)
        average_factor = 1 / np.sqrt(neighbor_size * center_neighbor_size)

        return neighbor_vec * average_factor


    """
    GCN Neighborhood extractor
    """


    def GCN_aggregator(self, node_index):
        neighbors, size = self.get_neighborhood_data(node_index)
        aggregate_vector = np.zeros(self.attribute_size)
        for index in neighbors:
            neighbor_average_vec = self.average_neighborhood(index, size)
            aggregate_vector += neighbor_average_vec

        return aggregate_vector


    """
    mean_pooling neighborhood
    """


    def mean_pooling_neighbor(self, node_index):
        neighbors, size = self.get_neighborhood_data(node_index)
        attribute_vector_total = np.zeros(self.attribute_size)
        for index in neighbors:
            attribute_vector_total += self.assign_value(index)

        return attribute_vector_total / size


    """
    Define get batch 
    """


    def get_batch_BFS(self,start_index):
        walk = np.zeros((self.batch_size, self.walk_length))
        batch_start_nodes = []
        nodes = np.array(self.G.nodes())
        for i in range(self.batch_size):
            walk_single = np.array(self.BFS_search(nodes[i + start_index]))
            batch_start_nodes.append(nodes[i + start_index])
            walk[i, :] = walk_single
        return walk, batch_start_nodes


    """
    get minibatch center data
    """


    def get_minibatch(self, index_vector):
        mini_batch = np.zeros((self.batch_size, self.attribute_size))
        index = 0
        for node_index in index_vector:
            x_center1 = self.assign_value(node_index)
            mini_batch[index, :] = x_center1
            index += 1

        return mini_batch


    """
    get batch neighbor_GCN_aggregate
    """


    def get_batch_GCNagg(self, index_vector):
        mini_batch_gcn_agg = np.zeros((self.batch_size, self.attribute_size))
        index = 0
        for node_index in index_vector:
            single_gcn = self.GCN_aggregator(node_index)
            mini_batch_gcn_agg[index, :] = single_gcn
            index += 1

        return mini_batch_gcn_agg


    """
    get batch negative sampling 
    """


    def get_batch_negative(self,negative_samples):
        mini_batch_negative = np.zeros((self.batch_size, self.negative_sample_size, self.attribute_size))
        for i in range(self.batch_size):
            index = 0
            for node in negative_samples[i, :]:
                negative_sample = self.assign_value(node)
                mini_batch_negative[i, index, :] = negative_sample
                index += 1
        return mini_batch_negative


    """
    get batch skip_gram samples
    """


    def get_batch_skip_gram(self, skip_gram_vecs):
        mini_batch_skip_gram = np.zeros((self.batch_size, self.walk_length, self.attribute_size))
        for i in range(self.batch_size):
            index = 0
            for node in skip_gram_vecs[i, :]:
                skip_gram_sample = self.assign_value(node)
                mini_batch_skip_gram[i, index, :] = skip_gram_sample
                index += 1
        return mini_batch_skip_gram


    """
    Uniform sample negative data
    """


    def uniform_get_negative_sample(self, skip_gram_vec, center_node):
        negative_samples = []
        node_neighbors, neighbor_size = self.get_neighborhood_data(center_node)
        total_negative_samples = 0
        while (total_negative_samples < self.negative_sample_size):
            index_sample = np.int(np.floor(np.random.uniform(0, np.array(self.G.nodes()).shape[0], 1)))
            sample = np.int(np.array(self.G.nodes())[index_sample])
            correct_negative_sample = 1
            for positive_sample in skip_gram_vec:
                if sample == np.int(positive_sample):
                    correct_negative_sample = 0
                    break
            if correct_negative_sample == 0:
                continue
            for neighborhood_sample in node_neighbors:
                if sample == neighborhood_sample:
                    correct_negative_sample = 0
            if sample == center_node:
                correct_negative_sample = 0
            if correct_negative_sample == 1:
                total_negative_samples += 1
                negative_samples.append(sample)

        return negative_samples


    """
    get batch mean_pooling
    """


    def get_batch_mean_pooling(self, index_vector):
        mini_batch_mean_pooling = np.zeros((self.batch_size, self.attribute_size))
        mini_batch_skip_gram_vectors = np.zeros((self.batch_size, self.walk_length))
        index = 0
        for node_index in index_vector:
            # skip_gram_vector = np.array(get_attribute_data_skip_gram(node_index, walk_length))
            skip_gram_vector = np.array(self.BFS_search(node_index))
            y_mean_pooling1 = self.mean_pooling(skip_gram_vector)
            mini_batch_mean_pooling[index, :] = y_mean_pooling1
            mini_batch_skip_gram_vectors[index, :] = skip_gram_vector
            index += 1

        return mini_batch_mean_pooling, mini_batch_skip_gram_vectors



    def get_data(self, start_index):
        # mini_batch_raw = np.array(Node2vec.simulate_walks(batch_size,walk_length))
        mini_batch_raw, start_nodes = get_batch_BFS(start_index)
        negative_samples = np.zeros((self.batch_size, self.negative_sample_size))
        negative_samples_vectors = np.zeros((self.batch_size, self.negative_sample_size, 8))
        skip_gram_vectors = np.zeros((self.batch_size, self.walk_length, 8))
        mini_batch_y = np.zeros((self.batch_size, self.length))
        mini_batch_x_label = np.zeros((self.batch_size, self.length))
        mini_batch_x_y = np.zeros((self.batch_size, self.length * 2))
        mini_batch_x = self.get_minibatch(start_nodes)
        batch_GCN_agg = self.get_batch_GCNagg(start_nodes)
        mini_batch_y_mean_pool, mini_batch_skip_gram = self.get_batch_mean_pooling(start_nodes)
        #batch_b_concat = convert_binary(mini_batch_y_mean_pool)

        # for i in range(batch_size):
        # negative_samples[i,:] = uniform_get_negative_sample(G,mini_batch_raw[i,:],start_nodes[i],negative_sample_size)

        # negative_samples_vectors = get_batch_negative(G,negative_samples,batch_size,negative_sample_size)

        # skip_gram_vectors = get_batch_skip_gram(G,mini_batch_raw,batch_size,walk_length)

        for i in range(self.batch_size):
            # index_node = G.nodes[mini_batch_raw[i][0]]['node_index']
            # mini_batch_x[i,:] = 1
            # prob = 1/walk_length
            for j in range(self.walk_length):
                indexy = self.G.nodes[mini_batch_raw[i][j]]['index']
                mini_batch_y[i][indexy] = 1 / walk_length  # += prob
            # mini_batch_x_y[i] = np.concatenate((mini_batch_x[i], mini_batch_y[i]),axis=None)
            indexx = self.G.nodes[start_nodes[i]]['index']
            mini_batch_x_label[i][indexx] = 1
            mini_batch_x_y[i] = np.concatenate((mini_batch_x_label[i], mini_batch_y[i]), axis=None)

        mini_batch_concat_x_y = np.concatenate((mini_batch_x, mini_batch_y_mean_pool), axis=1)

        return mini_batch_x, mini_batch_y, batch_GCN_agg, negative_samples_vectors, skip_gram_vectors, mini_batch_x_label, mini_batch_x_y, mini_batch_y_mean_pool, mini_batch_concat_x_y


    def get_data_one_batch(self, start_index_):
        mini_batch_integral = np.zeros((self.batch_size, 1 + self.walk_length + self.negative_sample_size, self.attribute_size))
        mini_batch_raw, start_nodes = self.get_batch_BFS(start_index_)
        mini_batch_y = np.zeros((self.batch_size, self.length))
        mini_batch_x_label = np.zeros((self.batch_size, self.length))
        #mini_batch_y_mean_pool = np.zeros(3)

        batch_center_x = self.get_minibatch(start_nodes)
        batch_GCN_agg = self.get_batch_GCNagg(start_nodes)
        negative_samples = np.zeros((self.batch_size, self.negative_sample_size))
        #skip_gram_vectors = np.zeros((batch_size, walk_length, attribute_size))
        #negative_samples_vectors = np.zeros((batch_size, negative_sample_size, attribute_size))
        for i in range(self.batch_size):
            mini_batch_integral[i, 0, :] = batch_GCN_agg[i, :]

        mini_batch_y_mean_pool, mini_batch_skip_gram = self.get_batch_mean_pooling(start_nodes)
    
        for i in range(self.batch_size):
            negative_samples[i, :] = self.uniform_get_negative_sample(mini_batch_raw[i, :], start_nodes[i])
    
        negative_samples_vectors = self.get_batch_negative(negative_samples)
    
        skip_gram_vectors = self.get_batch_skip_gram(mini_batch_raw)
        for i in range(self.batch_size):
            mini_batch_integral[i, 1:self.walk_length + 1, :] = skip_gram_vectors[i, :, :]
    
        for i in range(self.batch_size):
            mini_batch_integral[i, self.walk_length + 1:, :] = negative_samples_vectors[i, :, :]
    
        """
        for i in range(batch_size):
          #index_node = G.nodes[mini_batch_raw[i][0]]['node_index']
          #mini_batch_x[i,:] = 1
          #prob = 1/walk_length
          for j in range(walk_length):
            indexy = G.nodes[mini_batch_raw[i][j]]['node_index']
            mini_batch_y[i][indexy] = 1#/walk_length#+= prob
          #mini_batch_x_y[i] = np.concatenate((mini_batch_x[i], mini_batch_y[i]),axis=None)
          indexx = G.nodes[start_nodes[i]]['node_index']
          mini_batch_x_label[i][indexx] = 1
        """
        for i in range(self.batch_size):
            indexy = self.G.node[start_nodes[i]]['node_index']
            mini_batch_y[i][indexy] = 1
            for j in self.G.neighbors(start_nodes[i]):
                indexy = self.G.node[j]['node_index']
                mini_batch_y[i][indexy] = 1

        return mini_batch_integral, mini_batch_y, mini_batch_x_label, mini_batch_y_mean_pool, batch_center_x


    def get_data_one_batch_later(self, start_index_):
        mini_batch_integral = np.zeros((self.batch_size, 1 + self.walk_length + self.negative_sample_size, self.attribute_size))
        mini_batch_raw, start_nodes = self.get_batch_BFS(start_index_)
        mini_batch_y = np.zeros((self.batch_size, self.length))
        mini_batch_x_label = np.zeros((self.batch_size, self.length))
        mini_batch_y_mean_pool = np.zeros(3)

        batch_center_x = self.get_minibatch(start_nodes)
        batch_GCN_agg = self.get_batch_GCNagg(start_nodes)
        for i in range(self.batch_size):
            mini_batch_integral[i, 0, :] = batch_GCN_agg[i, :]
        """
        mini_batch_y_mean_pool, mini_batch_skip_gram = get_batch_mean_pooling(G, start_nodes, batch_size, walk_length,
                                                                              attribute_size)
    
        for i in range(batch_size):
            negative_samples[i, :] = uniform_get_negative_sample(G, mini_batch_raw[i, :], start_nodes[i],
                                                                 negative_sample_size)
    
        negative_samples_vectors = get_batch_negative(G, negative_samples, batch_size, negative_sample_size, attribute_size)
    
        skip_gram_vectors = get_batch_skip_gram(G, mini_batch_raw, batch_size, walk_length, attribute_size)
        for i in range(batch_size):
            mini_batch_integral[i, 1:walk_length + 1, :] = skip_gram_vectors[i, :, :]
    
        for i in range(batch_size):
            mini_batch_integral[i, walk_length + 1:, :] = negative_samples_vectors[i, :, :]
    
    
        for i in range(batch_size):
          #index_node = G.nodes[mini_batch_raw[i][0]]['node_index']
          #mini_batch_x[i,:] = 1
          #prob = 1/walk_length
          for j in range(walk_length):
            indexy = G.nodes[mini_batch_raw[i][j]]['node_index']
            mini_batch_y[i][indexy] = 1#/walk_length#+= prob
          #mini_batch_x_y[i] = np.concatenate((mini_batch_x[i], mini_batch_y[i]),axis=None)
          indexx = G.nodes[start_nodes[i]]['node_index']
          mini_batch_x_label[i][indexx] = 1
        """
        for i in range(self.batch_size):
            indexy = self.G.node[start_nodes[i]]['node_index']
            mini_batch_y[i][indexy] = 1
            for j in self.G.neighbors(start_nodes[i]):
                indexy = self.G.node[j]['node_index']
                mini_batch_y[i][indexy] = 1

        return mini_batch_integral, mini_batch_y, mini_batch_x_label, mini_batch_y_mean_pool, batch_center_x

    def train(self):
        for j in range(100):
            mini_batch_integral, mini_batch_y, mini_batch_x_label, mini_batch_y_mean_pool, mini_batch_x = self.get_data_one_batch(np.int(np.floor(np.random.uniform(0, 89498))))
            err_ = self.sess.run([self.total_loss, self.train_step_auto], feed_dict={self.x_gcn: mini_batch_integral,
                                                                      self.y_mean_pooling: mini_batch_y_mean_pool,
                                                                      self.x_center: mini_batch_x})

            print(err_[0])