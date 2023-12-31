# Copyright 2018 The Lucid Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Helper for using sklearn.decomposition on high-dimensional tensors.
Provides ChannelReducer, a wrapper around sklearn.decomposition to help them
apply to arbitrary rank tensors. It saves lots of annoying reshaping.
"""

import numpy as np
import sklearn.decomposition
import sklearn.cluster

try:
    from sklearn.base import BaseEstimator as DecompositionBaseEstimator
except AttributeError:
    from sklearn.base import BaseEstimator as DecompositionBaseEstimator


try:
    from sklearn.cluster.base import BaseEstimator as ClusterBaseEstimator
except AttributeError:
    from sklearn.base import BaseEstimator as ClusterBaseEstimator
except ModuleNotFoundError:
    from sklearn.base import BaseEstimator as ClusterBaseEstimator


class ChannelReducer(object):
  """Helper for dimensionality reduction to the innermost dimension of a tensor.
  This class wraps sklearn.decomposition classes to help them apply to arbitrary
  rank tensors. It saves lots of annoying reshaping.
  See the original sklearn.decomposition documentation:
  http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
  """

  def __init__(self, n_components=3, reduction_alg="NMF", **kwargs):
    """Constructor for ChannelReducer.
    Inputs:
      n_components: Numer of dimensions to reduce inner most dimension to.
      reduction_alg: A string or sklearn.decomposition class. Defaults to
        "NMF" (non-negative matrix facotrization). Other options include:
        "PCA", "FastICA", and "MiniBatchDictionaryLearning". The name of any of
        the sklearn.decomposition classes will work, though.
      kwargs: Additional kwargs to be passed on to the reducer.
    """

    if not isinstance(n_components, int):
      raise ValueError("n_components must be an int, not '%s'." % n_components)

    # Defensively look up reduction_alg if it is a string and give useful errors.
    algorithm_map = {}
    for name in dir(sklearn.decomposition):
      obj = sklearn.decomposition.__getattribute__(name)
      if isinstance(obj, type) and issubclass(obj, DecompositionBaseEstimator):
        algorithm_map[name] = obj
    if isinstance(reduction_alg, str):
      if reduction_alg in algorithm_map:
        reduction_alg = algorithm_map[reduction_alg]
      else:
        raise ValueError("Unknown dimensionality reduction method '%s'." % reduction_alg)


    self.n_components = n_components
    self._reducer = reduction_alg(n_components=n_components, **kwargs)
    self._is_fit = False

  @classmethod
  def _apply_flat(cls, f, acts):
    """Utility for applying f to inner dimension of acts.
    Flattens acts into a 2D tensor, applies f, then unflattens so that all
    dimesnions except innermost are unchanged.
    """
    orig_shape = acts.shape
    acts_flat = acts.reshape([-1, acts.shape[-1]])
    new_flat = f(acts_flat)
    if not isinstance(new_flat, np.ndarray):
      return new_flat
    shape = list(orig_shape[:-1]) + [-1]
    return new_flat.reshape(shape)

  def fit(self, acts):
    self._is_fit = True
    return ChannelReducer._apply_flat(self._reducer.fit, acts)

  def fit_transform(self, acts):
    self._is_fit = True
    return ChannelReducer._apply_flat(self._reducer.fit_transform, acts)

  def transform(self, acts):
    return ChannelReducer._apply_flat(self._reducer.transform, acts)

  def inverse_transform(self, acts):
    return ChannelReducer._apply_flat(self._reducer.inverse_transform, acts)
'''
  def __call__(self, acts):
    if self._is_fit:
      return self.transform(acts)
    else:
      return self.fit_transform(acts)

  def __getattr__(self, name):
    if name in self.__dict__:
      return self.__dict__[name]
    elif name + "_" in self._reducer.__dict__:
      return self._reducer.__dict__[name+"_"]

  def __dir__(self):
    dynamic_attrs = [name[:-1]
                     for name in dir(self._reducer)
                     if name[-1] == "_" and name[0] != "_"
                    ]

    return list(ChannelReducer.__dict__.keys()) + list(self.__dict__.keys()) + dynamic_attrs

'''


class ClusterReducer(object):

  def __init__(self, n_clusters=3, reduction_alg="MiniBatchKMeans", **kwargs):


    if not isinstance(n_clusters, int):
      raise ValueError("n_clusters must be an int, not '%s'." % n_clusters)

    # Defensively look up reduction_alg if it is a string and give useful errors.
    algorithm_map = {}
    for name in dir(sklearn.cluster):
      obj = sklearn.cluster.__getattribute__(name)
      if isinstance(obj, type) and issubclass(obj, ClusterBaseEstimator):
        algorithm_map[name] = obj
    if isinstance(reduction_alg, str):
      if reduction_alg in algorithm_map:
        reduction_alg = algorithm_map[reduction_alg]
      else:
        raise ValueError("Unknown dimensionality reduction method '%s'." % reduction_alg)


    self.n_clusters = n_clusters
    self._reducer = reduction_alg(n_clusters=n_clusters, **kwargs)
    self._is_fit = False

  def _apply_flat(self, f, acts):
    """Utility for applying f to inner dimension of acts.
    Flattens acts into a 2D tensor, applies f, then unflattens so that all
    dimesnions except innermost are unchanged.
    """
    orig_shape = acts.shape
    acts_flat = acts.reshape([-1, acts.shape[-1]])
    new_flat = f(acts_flat)
    if not isinstance(new_flat, np.ndarray):
      return new_flat
    shape = list(orig_shape[:-1]) + [-1]
    new_flat = new_flat.reshape(shape)


    if new_flat.shape[-1] == 1:
      new_flat = new_flat.reshape(-1)
      t_flat = np.zeros([new_flat.shape[0],self.n_clusters])
      t_flat[np.arange(new_flat.shape[0]),new_flat] = 1
      new_flat = t_flat.reshape(shape)

    return new_flat

  def fit(self, acts):
    self._is_fit = True
    res = ClusterReducer._apply_flat(self,self._reducer.fit, acts)
    self._reducer.components_ = self._reducer.cluster_centers_
    return res

  def fit_predict(self, acts):
    self._is_fit = True
    res = ClusterReducer._apply_flat(self,self._reducer.fit_predict, acts)
    self._reducer.components_ = self._reducer.cluster_centers_
    return res

  def transform(self, acts):
    res = ClusterReducer._apply_flat(self,self._reducer.predict, acts)
    return res

  def inverse_transform(self, acts):

    res = np.dot(acts,self._reducer.cluster_centers_)
    return res
