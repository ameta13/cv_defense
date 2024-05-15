from copy import deepcopy
from functools import partial
import time

import numpy as np
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset

import torchvision
import torchvision.transforms as transforms

from cv_defense.helpers.training import train_model, evaluate


def get_tensor_intersection(t1, t2):
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    intersection = uniques[counts > 1]
    assert len(intersection.shape) == 1, f'expected shape with one dimension, got {intersection.shape=}'
    return intersection.shape[0]


def print_adversarial_intersection(example_idxs, clusters_labels, adversarial_examples_idxs):
    adversarial_cnt = len(adversarial_examples_idxs)

    intersection_size = get_tensor_intersection(example_idxs[clusters_labels == 0], adversarial_examples_idxs)
    print(f'Adversarial examples in cluster "0" = {intersection_size} ({intersection_size / adversarial_cnt * 100: .2f}%)')

    intersection_size = get_tensor_intersection(example_idxs[clusters_labels == 1], adversarial_examples_idxs)
    print(f'Adversarial examples in cluster "1" = {intersection_size} ({intersection_size / adversarial_cnt * 100: .2f}%)')


class ActivationClusteringDefense:
    def __init__(self, model, num_classes: int = 10, layer = None, n_ica_components: int = 5000, device='cuda', verbose=False):
        """
        Initializes the Activation Clustering defense method.

        :param model: model from which activations will be taken
        :param num_classes: count of classes
        :param layer: The layer whose activations will be clustered, by default - last layer
        :param n_ica_components: The number components for Independent Component Analysis (ICA)
        :param device: The device ('cpu' or 'cuda') to perform computations on
        """
        self.model = model.to(device)
        self.num_classes = num_classes
        self.layer = layer if layer is not None else list(model.modules())[-1]
        self.kmeans_clusters = 2
        self.n_ica_components = n_ica_components
        self.device = device
        self.num_epochs = 5
        self.verbose = verbose


    @staticmethod
    def _save_activation_hook(activations):
        def hook(model, input, output):
            activations.append(output.detach().cpu().numpy())
        return hook

    def _get_activations(self, dataloader: DataLoader):
        # Ensure the model is in evaluation mode
        self.model.eval()

        activations = []
        predictions = []
        # Register a hook to save the activations of the specified layer
        hook = self.layer.register_forward_hook(self._save_activation_hook(activations))

        with torch.no_grad():
            for images, _ in dataloader:
                inputs = images.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.append(predicted)
        hook.remove()

        activations = np.concatenate(activations)

        all_activations = torch.tensor(activations).to(self.device2)
        print(f'{str(all_activations.dtype)=}, {not np.isfinite(all_activations.cpu()).all()=}')
        all_predictions = torch.concat(predictions)
        res_activations = []  # idx is predicted class
        input_idxs = []
        for num_class in range(self.num_classes):
            mask_num_class = all_predictions == num_class
            res_activations.append(all_activations[mask_num_class])
            idxs = torch.squeeze(mask_num_class.nonzero())
            print(f'{idxs.shape=}')
            input_idxs.append(idxs)
        return res_activations, input_idxs

    def _reduce_dimensions(self, activations):
        activations_numpy = activations.cpu().numpy()

        n_ica_components = min(self.n_ica_components, activations_numpy.shape[1])
        transformer = FastICA(n_ica_components, random_state=0, max_iter=500, whiten='unit-variance')
        activations_ica_reduced = transformer.fit_transform(activations_numpy)
        return activations_ica_reduced

    def _get_kmeans_clusters(self, activations):
        # Fit K-means clustering on the activations
        kmeans = KMeans(n_clusters=self.kmeans_clusters, n_init=10, random_state=0)
        labels = kmeans.fit_predict(activations)
        return labels

    def _analyze_by_relative_size(self, cluster_labels, threshold: float):
        assert len(cluster_labels.shape) == 1, f'got bad {cluster_labels.shape=}'
        relative_size = torch.tensor(cluster_labels).bincount(minlength=2) / cluster_labels.shape[0]
        if self.verbose:
            print(f'{relative_size=}')
        for num_cluster, cluster_rel_size in enumerate(relative_size):
            if cluster_rel_size.item() < threshold:
                if self.verbose:
                    print(f'Found cluster with adversarial examples. {num_cluster=}')
                return num_cluster
        return None

    def analyze_by_relative_size(self, dataloader: DataLoader, threshold: float):
        assert 0 < threshold < 1, f'Threshold value expected between 0 and 1, got {threshold=}'
        activations_by_classes, example_idxs_by_classes = self._get_activations(dataloader)
        adversarial_input_idxs = {}
        for num_class, (activations, example_idxs) in enumerate(zip(activations_by_classes, example_idxs_by_classes)):
            if self.verbose:
                print(f'\n\nAnalizyng {num_class=}')
                print(f'{activations.shape=}, {example_idxs.shape=}')
            activations = activations.cpu()
            if activations.shape[1] > self.n_ica_components:
                activations = self._reduce_dimensions(activations)
                if self.verbose:
                    print(f'activations_ica_reduced.shape={activations.shape}')
            clusters_labels = self._get_kmeans_clusters(activations)

            adversarial_cluster = self._analyze_by_relative_size(clusters_labels, threshold)
            if adversarial_cluster is not None:
                adversarial_input_idxs[num_class] = example_idxs[clusters_labels == adversarial_cluster]
        return adversarial_input_idxs

    def _analyze_cluster_data(self, num_class, images_inside_cluster: DataLoader, images_outside_cluster: DataLoader, new_model, threshold) -> bool:
        def _get_other_class(y_pred, num_class):
            classes, freqs = np.unique(y_pred[y_pred != num_class], return_counts=True)
            other_class = classes[freqs.argmax().item()].item()
            return other_class

        clean_model = deepcopy(new_model)
        if self.verbose:
            print(f'Start training, train_dataset size =', len(images_outside_cluster.dataset.indices), ', eval dataset size =', len(images_inside_cluster.dataset.indices))
        start = time.time()
        train_model(clean_model, images_outside_cluster, num_epochs=self.num_epochs, device=self.device)

        # Predict class for examples from another cluster
        y_true, y_pred = evaluate(clean_model, images_inside_cluster, device=self.device)
        label_count = np.sum(y_pred == num_class)
        other_class = _get_other_class(y_pred, num_class)
        other_class_count = np.sum(y_pred == other_class)
        # Proportion of data classified as the source class over the data classified as their original label.
        ratio = label_count / other_class_count if other_class_count > 0 else threshold

        is_poison = ratio < threshold
        if self.verbose:
            print(f"The removed cluster is poisonous: {is_poison}. {ratio=}, {threshold=}, {label_count=}, {other_class=}, {other_class_count=}")
        return is_poison

    @staticmethod
    def _split_data_by_cluster(train_dataloader: DataLoader, example_idxs, clusters_labels, num_cluster: int) -> tuple[DataLoader, DataLoader]:
        batch_size = train_dataloader.batch_size
        dataset = train_dataloader.dataset

        idx_inside_cluster = example_idxs[clusters_labels == num_cluster]
        images_in_cluster = Subset(dataset, idx_inside_cluster)
        images_in_cluster_loader = DataLoader(images_in_cluster, batch_size=batch_size, shuffle=True)

        set_idx_inside_cluster = set(idx_inside_cluster.tolist())
        other_idxs = torch.tensor([idx for idx in range(len(dataset)) if idx not in set_idx_inside_cluster])
        images_outside_cluster = Subset(dataset, other_idxs)
        images_outside_cluster_loader = DataLoader(images_outside_cluster, batch_size=batch_size, shuffle=False)

        return images_in_cluster_loader, images_outside_cluster_loader

    def analyze_by_exclusion(self, dataloader: DataLoader, new_model, adversarial_examples_idxs = None, threshold: float = 1.) -> list[int]:
        assert threshold > 0, f'Threshold value expected greater than 0, got {threshold=}'
        if adversarial_examples_idxs is not None:
            adversarial_examples_idxs = torch.tensor(adversarial_examples_idxs).to(self.device)
        activations_by_classes, example_idxs_by_classes = self._get_activations(dataloader)
        for num_class, example_idxs in enumerate(example_idxs_by_classes):
            intersection_size = get_tensor_intersection(example_idxs, adversarial_examples_idxs)
            if self.verbose:
                print(f'Class {num_class} has {intersection_size} ({intersection_size / len(adversarial_examples_idxs) * 100:.2f}%) adversarial examples')

        found_adversarial_examples = []
        for num_class, (activations, example_idxs) in enumerate(zip(activations_by_classes, example_idxs_by_classes)):
            if self.verbose:
                print(f'\n\nAnalizyng {num_class=}')
                print(f'{activations.shape=}, {example_idxs.shape=}')
            activations = activations.cpu()
            if activations.shape[1] > self.n_ica_components:
                activations = self._reduce_dimensions(activations)
                if self.verbose:
                    print(f'activations_ica_reduced.shape={activations.shape}')
            clusters_labels = self._get_kmeans_clusters(activations)

            if adversarial_examples_idxs is not None:
                print_adversarial_intersection(example_idxs, clusters_labels, adversarial_examples_idxs)

            for num_cluster in range(self.kmeans_clusters):
                images_inside_cluster, images_outside_cluster = self._split_data_by_cluster(dataloader, example_idxs, clusters_labels, num_cluster)
                is_poison = self._analyze_cluster_data(num_class, images_inside_cluster, images_outside_cluster, new_model, threshold)
                if is_poison:
                    found_adversarial_examples += images_inside_cluster.dataset.get_all_image_paths()
        return found_adversarial_examples
