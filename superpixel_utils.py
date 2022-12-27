import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd
import h5py
from sklearn.cluster import MiniBatchKMeans, KMeans
import random 
from PIL import Image
from dgl.data.utils import save_graphs

from histocartography.utils import download_example_data
from histocartography.preprocessing import (
    ColorMergedSuperpixelExtractor,
    DeepFeatureExtractor
)
from histocartography.visualization import OverlayGraphVisualization

from skimage.measure import regionprops
import joblib
import cv2

import logging
import multiprocessing
import os
import sys
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import h5py
import pandas as pd
from tqdm.auto import tqdm

import logging
import multiprocessing
import os
import sys
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import h5py
import pandas as pd
from tqdm.auto import tqdm

class PipelineStep(ABC):
    """Base pipelines step"""

    def __init__(
        self,
        save_path: Union[None, str, Path] = None,
        precompute: bool = True,
        link_path: Union[None, str, Path] = None,
        precompute_path: Union[None, str, Path] = None,
    ) -> None:
        """Abstract class that helps with saving and loading precomputed results
        Args:
            save_path (Union[None, str, Path], optional): Base path to save results to.
                When set to None, the results are not saved to disk. Defaults to None.
            precompute (bool, optional): Whether to perform the precomputation necessary
                for the step. Defaults to True.
            link_path (Union[None, str, Path], optional): Path to link the output directory
                to. When None, no link is created. Only supported when save_path is not None.
                Defaults to None.
            precompute_path (Union[None, str, Path], optional): Path to save the output of
                the precomputation to. If not specified it defaults to the output directory
                of the step when save_path is not None. Defaults to None.
        """
        assert (
            save_path is not None or link_path is None
        ), "link_path only supported when save_path is not None"

        name = self.__repr__()
        self.save_path = save_path
        if self.save_path is not None:
            self.output_dir = Path(self.save_path) / name
            self.output_key = "default_key"
            self._mkdir()
            if precompute_path is None:
                precompute_path = save_path

        if precompute:
            self.precompute(
                link_path=link_path,
                precompute_path=precompute_path)

    def __repr__(self) -> str:
        """Representation of a pipeline step.
        Returns:
            str: Representation of a pipeline step.
        """
        variables = ",".join(
            [f"{k}={v}" for k, v in sorted(self.__dict__.items())])
        return (
            f"{self.__class__.__name__}({variables})".replace(" ", "")
            .replace('"', "")
            .replace("'", "")
            .replace("..", "")
            .replace("/", "_")
        )

    def _mkdir(self) -> None:
        """Create path to output files"""
        assert (
            self.save_path is not None
        ), "Can only create directory if base_path was not None when constructing the object"
        if not self.output_dir.exists():
            self.output_dir.mkdir()

    def _link_to_path(self, link_directory: Union[None, str, Path]) -> None:
        """Links the output directory to the given directory.
        Args:
            link_directory (Union[None, str, Path]): Directory to link to
        """
        if link_directory is None or Path(
                link_directory).parent.resolve() == Path(self.output_dir):
            logging.info("Link to self skipped")
            return
        assert (
            self.save_path is not None
        ), f"Linking only supported when saving is enabled, i.e. when save_path is passed in the constructor."
        if os.path.islink(link_directory):
            if os.path.exists(link_directory):
                logging.info("Link already exists: overwriting...")
                os.remove(link_directory)
            else:
                logging.critical(
                    "Link exists, but points nowhere. Ignoring...")
                return
        elif os.path.exists(link_directory):
            os.remove(link_directory)
        os.symlink(self.output_dir, link_directory, target_is_directory=True)

    def precompute(
        self,
        link_path: Union[None, str, Path] = None,
        precompute_path: Union[None, str, Path] = None,
    ) -> None:
        """Precompute all necessary information for this step
        Args:
            link_path (Union[None, str, Path], optional): Path to link the output to. Defaults to None.
            precompute_path (Union[None, str, Path], optional): Path to load/save the precomputation outputs. Defaults to None.
        """
        pass

    def process(
        self, *args: Any, output_name: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Main process function of the step and outputs the result. Try to saves the output when output_name is passed.
        Args:
            output_name (Optional[str], optional): Unique identifier of the passed datapoint. Defaults to None.
        Returns:
            Any: Result of the pipeline step
        """
        if output_name is not None and self.save_path is not None:
            return self._process_and_save(
                *args, output_name=output_name, **kwargs)
        else:
            return self._process(*args, **kwargs)

    @abstractmethod
    def _process(self, *args: Any, **kwargs: Any) -> Any:
        """Abstract method that performs the computation of the pipeline step
        Returns:
            Any: Result of the pipeline step
        """

    def _get_outputs(self, input_file: h5py.File) -> Union[Any, Tuple]:
        """Extracts the step output from a given h5 file
        Args:
            input_file (h5py.File): File to load from
        Returns:
            Union[Any, Tuple]: Previously computed output of the step
        """
        outputs = list()
        nr_outputs = len(input_file.keys())

        # Legacy, remove at some point
        if nr_outputs == 1 and self.output_key in input_file.keys():
            return tuple([input_file[self.output_key][()]])

        for i in range(nr_outputs):
            outputs.append(input_file[f"{self.output_key}_{i}"][()])
        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)

    def _set_outputs(self, output_file: h5py.File,
                     outputs: Union[Tuple, Any]) -> None:
        """Save the step output to a given h5 file
        Args:
            output_file (h5py.File): File to write to
            outputs (Union[Tuple, Any]): Computed step output
        """
        if not isinstance(outputs, tuple):
            outputs = tuple([outputs])
        for i, output in enumerate(outputs):
            output_file.create_dataset(
                f"{self.output_key}_{i}",
                data=output,
                compression="gzip",
                compression_opts=9,
            )

    def _process_and_save(
        self, *args: Any, output_name: str, **kwargs: Any
    ) -> Any:
        """Process and save in the provided path as as .h5 file
        Args:
            output_name (str): Unique identifier of the the passed datapoint
        Raises:
            read_error (OSError): When the unable to read to self.output_dir/output_name.h5
            write_error (OSError): When the unable to write to self.output_dir/output_name.h5
        Returns:
            Any: Result of the pipeline step
        """
        assert (
            self.save_path is not None
        ), "Can only save intermediate output if base_path was not None when constructing the object"
        output_path = self.output_dir / f"{output_name}.h5"
        if output_path.exists():
            logging.info(
                f"{self.__class__.__name__}: Output of {output_name} already exists, using it instead of recomputing"
            )
            try:
                with h5py.File(output_path, "r") as input_file:
                    output = self._get_outputs(input_file=input_file)
            except OSError as read_error:
                print(f"\n\nCould not read from {output_path}!\n\n")
                raise read_error
        else:
            output = self._process(*args, **kwargs)
            try:
                with h5py.File(output_path, "w") as output_file:
                    self._set_outputs(output_file=output_file, outputs=output)
            except OSError as write_error:
                print(f"\n\nCould not write to {output_path}!\n\n")
                raise write_error
        return output

def fast_histogram(input_array: np.ndarray, nr_values: int) -> np.ndarray:
    """Calculates a histogram of a matrix of the values from 0 up to (excluding) nr_values
    Args:
        x (np.array): Input tensor
        nr_values (int): Possible values. From 0 up to (exclusing) nr_values.
    Returns:
        np.array: Output tensor
    """
    output_array = np.empty(nr_values, dtype=int)
    for i in range(nr_values):
        output_array[i] = (input_array == i).sum()
    return output_array


def load_image(image_path: Path) -> np.ndarray:
    """Loads an image from a given path and returns it as a numpy array
    Args:
        image_path (Path): Path of the image
    Returns:
        np.ndarray: Array representation of the image
    """
    assert image_path.exists()
    try:
        with Image.open(image_path) as img:
            image = np.array(img)
    except OSError as e:
        logging.critical("Could not open %s", image_path)
        raise OSError(e)
    return image

"""This module handles all the graph building"""

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import cv2
import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
from dgl.data.utils import load_graphs, save_graphs
from skimage.measure import regionprops
from sklearn.neighbors import kneighbors_graph

# from ..pipeline import PipelineStep
# from .utils import fast_histogram



LABEL = "label"
CENTROID = "centroid"
FEATURES = "feat"


def two_hop_neighborhood(graph: dgl.DGLGraph) -> dgl.DGLGraph:
    """Increases the connectivity of a given graph by an additional hop
    Args:
        graph (dgl.DGLGraph): Input graph
    Returns:
        dgl.DGLGraph: Output graph
    """
    A = graph.adjacency_matrix().to_dense()
    A_tilde = (1.0 * ((A + A.matmul(A)) >= 1)) - torch.eye(A.shape[0])
    ngraph = nx.convert_matrix.from_numpy_matrix(A_tilde.numpy())
    new_graph = dgl.DGLGraph()
    new_graph.from_networkx(ngraph)
    for k, v in graph.ndata.items():
        new_graph.ndata[k] = v
    for k, v in graph.edata.items():
        new_graph.edata[k] = v
    return new_graph


class BaseGraphBuilder(PipelineStep):
    """
    Base interface class for graph building.
    """

    def __init__(
            self,
            nr_annotation_classes: int = 5,
            annotation_background_class: Optional[int] = None,
            add_loc_feats: bool = False,
            **kwargs: Any
    ) -> None:
        """
        Base Graph Builder constructor.
        Args:
            nr_annotation_classes (int): Number of classes in annotation. Used only if setting node labels.
            annotation_background_class (int): Background class label in annotation. Used only if setting node labels.
            add_loc_feats (bool): Flag to include location-based features (ie normalized centroids)
                                  in node feature representation.
                                  Defaults to False.
        """
        self.nr_annotation_classes = nr_annotation_classes
        self.annotation_background_class = annotation_background_class
        self.add_loc_feats = add_loc_feats
        super().__init__(**kwargs)

    def _process(  # type: ignore[override]
        self,
        instance_map: np.ndarray,
        features: torch.Tensor,
        annotation: Optional[np.ndarray] = None,
    ) -> dgl.DGLGraph:
        """Generates a graph from a given instance_map and features
        Args:
            instance_map (np.array): Instance map depicting tissue components
            features (torch.Tensor): Features of each node. Shape (nr_nodes, nr_features)
            annotation (Union[None, np.array], optional): Optional node level to include.
                                                          Defaults to None.
        Returns:
            dgl.DGLGraph: The constructed graph
        """
        # add nodes
        num_nodes = features.shape[0]
        graph = dgl.DGLGraph()
        graph.add_nodes(num_nodes)

        # add image size as graph data
        image_size = (instance_map.shape[1], instance_map.shape[0])  # (x, y)

        # get instance centroids
        centroids = self._get_node_centroids(instance_map)

        # add node content
        self._set_node_centroids(centroids, graph)
        self._set_node_features(features, image_size, graph)
        if annotation is not None:
            self._set_node_labels(instance_map, annotation, graph)

        # build edges
        self._build_topology(instance_map, centroids, graph)
        return graph

    def _process_and_save(  # type: ignore[override]
        self,
        instance_map: np.ndarray,
        features: torch.Tensor,
        annotation: Optional[np.ndarray] = None,
        output_name: str = None,
    ) -> dgl.DGLGraph:
        """Process and save in provided directory
        Args:
            output_name (str): Name of output file
            instance_map (np.ndarray): Instance map depicting tissue components
                                       (eg nuclei, tissue superpixels)
            features (torch.Tensor): Features of each node. Shape (nr_nodes, nr_features)
            annotation (Optional[np.ndarray], optional): Optional node level to include.
                                                         Defaults to None.
        Returns:
            dgl.DGLGraph: [description]
        """
        assert (
            self.save_path is not None
        ), "Can only save intermediate output if base_path was not None during construction"
        output_path = self.output_dir / f"{output_name}.bin"
        if output_path.exists():
            logging.info(
                f"Output of {output_name} already exists, using it instead of recomputing"
            )
            graphs, _ = load_graphs(str(output_path))
            assert len(graphs) == 1
            graph = graphs[0]
        else:
            graph = self._process(
                instance_map=instance_map,
                features=features,
                annotation=annotation)
            save_graphs(str(output_path), [graph])
        return graph

    def _get_node_centroids(
            self, instance_map: np.ndarray
    ) -> np.ndarray:
        """Get the centroids of the graphs
        Args:
            instance_map (np.ndarray): Instance map depicting tissue components
        Returns:
            centroids (np.ndarray): Node centroids
        """
        regions = regionprops(instance_map)
        centroids = np.empty((len(regions), 2))
        for i, region in enumerate(regions):
            center_y, center_x = region.centroid  # (y, x)
            center_x = int(round(center_x))
            center_y = int(round(center_y))
            centroids[i, 0] = center_x
            centroids[i, 1] = center_y
        return centroids

    def _set_node_centroids(
            self,
            centroids: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Set the centroids of the graphs
        Args:
            centroids (np.ndarray): Node centroids
            graph (dgl.DGLGraph): Graph to add the centroids to
        """
        graph.ndata[CENTROID] = torch.FloatTensor(centroids)

    def _set_node_features(
            self,
            features: torch.Tensor,
            image_size: Tuple[int, int],
            graph: dgl.DGLGraph
    ) -> None:
        """Set the provided node features
        Args:
            features (torch.Tensor): Node features
            image_size (Tuple[int,int]): Image dimension (x, y)
            graph (dgl.DGLGraph): Graph to add the features to
        """
        if not torch.is_tensor(features):
            features = torch.FloatTensor(features)
        if not self.add_loc_feats:
            graph.ndata[FEATURES] = features
        elif (
                self.add_loc_feats
                and image_size is not None
        ):
            # compute normalized centroid features
            centroids = graph.ndata[CENTROID]

            normalized_centroids = torch.empty_like(centroids)  # (x, y)
            normalized_centroids[:, 0] = centroids[:, 0] / image_size[0]
            normalized_centroids[:, 1] = centroids[:, 1] / image_size[1]

            if features.ndim == 3:
                normalized_centroids = normalized_centroids \
                    .unsqueeze(dim=1) \
                    .repeat(1, features.shape[1], 1)
                concat_dim = 2
            elif features.ndim == 2:
                concat_dim = 1

            concat_features = torch.cat(
                (
                    features,
                    normalized_centroids
                ),
                dim=concat_dim,
            )
            graph.ndata[FEATURES] = concat_features
        else:
            raise ValueError(
                "Please provide image size to add the normalized centroid to the node features."
            )

    @abstractmethod
    def _set_node_labels(
            self,
            instance_map: np.ndarray,
            annotation: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Set the node labels of the graphs
        Args:
            instance_map (np.ndarray): Instance map depicting tissue components
            annotation (np.ndarray): Annotations, eg node labels
            graph (dgl.DGLGraph): Graph to add the centroids to
        """

    @abstractmethod
    def _build_topology(
            self,
            instance_map: np.ndarray,
            centroids: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Generate the graph topology from the provided instance_map
        Args:
            instance_map (np.array): Instance map depicting tissue components
            centroids (np.array): Node centroids
            graph (dgl.DGLGraph): Graph to add the edges
        """

    def precompute(
        self,
        link_path: Union[None, str, Path] = None,
        precompute_path: Union[None, str, Path] = None,
    ) -> None:
        """Precompute all necessary information
        Args:
            link_path (Union[None, str, Path], optional): Path to link to. Defaults to None.
            precompute_path (Union[None, str, Path], optional): Path to save precomputation outputs. Defaults to None.
        """
        if self.save_path is not None and link_path is not None:
            self._link_to_path(Path(link_path) / "graphs")


class RAGGraphBuilder(BaseGraphBuilder):
    """
    Super-pixel Graphs class for graph building.
    """

    def __init__(self, kernel_size: int = 3, hops: int = 1, **kwargs) -> None:
        """Create a graph builder that uses a provided kernel size to detect connectivity
        Args:
            kernel_size (int, optional): Size of the kernel to detect connectivity. Defaults to 5.
        """
        logging.debug("*** RAG Graph Builder ***")
        assert hops > 0 and isinstance(
            hops, int
        ), f"Invalid hops {hops} ({type(hops)}). Must be integer >= 0"
        self.kernel_size = kernel_size
        self.hops = hops
        super().__init__(**kwargs)

    def _set_node_labels(
            self,
            instance_map: np.ndarray,
            annotation: np.ndarray,
            graph: dgl.DGLGraph) -> None:
        """Set the node labels of the graphs using annotation map"""
        assert (
            self.nr_annotation_classes < 256
        ), "Cannot handle that many classes with 8-bits"
        regions = regionprops(instance_map)
        labels = torch.empty(len(regions), dtype=torch.uint8)

        for region_label in np.arange(1, len(regions) + 1):
            histogram = fast_histogram(
                annotation[instance_map == region_label],
                nr_values=self.nr_annotation_classes
            )
            mask = np.ones(len(histogram), np.bool)
            mask[self.annotation_background_class] = 0
            if histogram[mask].sum() == 0:
                assignment = self.annotation_background_class
            else:
                histogram[self.annotation_background_class] = 0
                assignment = np.argmax(histogram)
            labels[region_label - 1] = int(assignment)
        graph.ndata[LABEL] = labels

    def _build_topology(
            self,
            instance_map: np.ndarray,
            centroids: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Create the graph topology from the instance connectivty in the instance_map"""
        regions = regionprops(instance_map)
        instance_ids = torch.empty(len(regions), dtype=torch.uint8)

        kernel = np.ones((3, 3), np.uint8)
        adjacency = np.zeros(shape=(len(instance_ids), len(instance_ids)))

        for instance_id in np.arange(1, len(instance_ids) + 1):
            mask = (instance_map == instance_id).astype(np.uint8)
#             print("mask:{}".format(mask))
            dilation = cv2.dilate(mask,kernel, iterations=1)
#             print("dilation:{}".format(dilation))
            boundary = dilation - mask
#             print("boundary:{}".format(boundary))
#             print(sum(sum(boundary)))
            idx = pd.unique(instance_map[boundary.astype(bool)])
#             print("idx:{}".format(idx))
#             print(len(idx))
            instance_id -= 1  # because instance_map id starts from 1
            idx -= 1  # because instance_map id starts from 1
#             print("new idx:{}".format(idx))
#             print(type(idx))
            idx = idx.tolist()
#             print(type(idx))
            if -1 in idx:
                idx.remove(-1)
            idx = np.array(idx)
#             print(type(idx))
#             print("new new idx:{}".format(idx))
            if idx.shape[0] != 0:    #可能存在没有的情况
                adjacency[instance_id, idx] = 1
#         print(adjacency)

        edge_list = np.nonzero(adjacency)
        graph.add_edges(list(edge_list[0]), list(edge_list[1]))

        for _ in range(self.hops - 1):
            graph = two_hop_neighborhood(graph)