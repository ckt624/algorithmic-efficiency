"""OGBG workload implemented in Jax."""
from typing import Dict, Tuple, Any, Optional

from . import input_pipeline
from . import models
from clu import metrics
import flax
from flax.training import train_state
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import sklearn.metrics
import spec
import tensorflow as tf


def predictions_match_labels(*, logits: jnp.ndarray, labels: jnp.ndarray,
                             **kwargs) -> jnp.ndarray:
  """Returns a binary array indicating where predictions match the labels."""
  del kwargs  # Unused.
  preds = (logits > 0)
  return (preds == labels).astype(jnp.float32)


@flax.struct.dataclass
class MeanAveragePrecision(
    metrics.CollectingMetric.from_outputs(('labels', 'logits', 'mask'))):
  """Computes the mean average precision (mAP) over different tasks."""

  def compute(self):
    # Matches the official OGB evaluation scheme for mean average precision.
    labels = self.values['labels']
    logits = self.values['logits']
    mask = self.values['mask']

    assert logits.shape == labels.shape == mask.shape
    assert len(logits.shape) == 2

    probs = jax.nn.sigmoid(logits)
    num_tasks = labels.shape[1]
    average_precisions = np.full(num_tasks, np.nan)

    for task in range(num_tasks):
      # AP is only defined when there is at least one negative data
      # and at least one positive data.
      if np.sum(labels[:, task] == 0) > 0 and np.sum(labels[:, task] == 1) > 0:
        is_labeled = mask[:, task]
        average_precisions[task] = sklearn.metrics.average_precision_score(
            labels[is_labeled, task], probs[is_labeled, task])

    # When all APs are NaNs, return NaN. This avoids raising a RuntimeWarning.
    if np.isnan(average_precisions).all():
      return np.nan
    return np.nanmean(average_precisions)


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):

  accuracy: metrics.Average.from_fun(predictions_match_labels)
  loss: metrics.Average.from_output('loss')
  mean_average_precision: MeanAveragePrecision


class OGBGWorkload(spec.Workload):
  """A ogbg workload."""

  def __init__(self):
    self._datasets = None
    self._eval_state = None

  def binary_cross_entropy_with_mask(self, logits: jnp.ndarray,
                                     labels: jnp.ndarray, mask: jnp.ndarray):
    """Binary cross entropy loss for unnormalized logits, with masked elements."""
    assert logits.shape == labels.shape == mask.shape
    assert len(logits.shape) == 2

    # To prevent propagation of NaNs during grad().
    # We mask over the loss for invalid targets later.
    labels = jnp.where(mask, labels, -1)

    # Numerically stable implementation of BCE loss.
    # This mimics TensorFlow's tf.nn.sigmoid_cross_entropy_with_logits().
    positive_logits = (logits >= 0)
    relu_logits = jnp.where(positive_logits, logits, 0)
    abs_logits = jnp.where(positive_logits, logits, -logits)
    return relu_logits - (logits * labels) + (jnp.log(1 + jnp.exp(-abs_logits)))

  def add_prefix_to_keys(self, result: Dict[str, Any],
                         prefix: str) -> Dict[str, Any]:
    """Adds a prefix to the keys of a dict, returning a new dict."""
    return {f'{prefix}_{key}': val for key, val in result.items()}

  def replace_globals(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Replaces the globals attribute with a constant feature for each graph."""
    return graphs._replace(globals=jnp.ones([graphs.n_node.shape[0], 1]))

  def get_predicted_logits(
      self, state: train_state.TrainState, graphs: jraph.GraphsTuple,
      rngs: Optional[Dict[str, jnp.ndarray]]) -> jnp.ndarray:
    """Get predicted logits from the network for input graphs."""
    pred_graphs = state.apply_fn(state.params, graphs, rngs=rngs)
    logits = pred_graphs.globals
    return logits

  def get_valid_mask(self, labels: jnp.ndarray,
                     graphs: jraph.GraphsTuple) -> jnp.ndarray:
    """Gets the binary mask indicating only valid labels and graphs."""
    # We have to ignore all NaN values - which indicate labels for which
    # the current graphs have no label.
    labels_mask = ~jnp.isnan(labels)

    # Since we have extra 'dummy' graphs in our batch due to padding, we want
    # to mask out any loss associated with the dummy graphs.
    # Since we padded with `pad_with_graphs` we can recover the mask by using
    # get_graph_padding_mask.
    graph_mask = jraph.get_graph_padding_mask(graphs)

    # Combine the mask over labels with the mask over graphs.
    return labels_mask & graph_mask[:, None]

  @jax.jit
  def evaluate_step(
      self,
      state: train_state.TrainState,
      graphs: jraph.GraphsTuple,
  ) -> metrics.Collection:
    """Computes metrics over a set of graphs."""

    # The target labels our model has to predict.
    labels = graphs.globals

    # Replace the global feature for graph classification.
    graphs = self.replace_globals(graphs)

    # Get predicted logits, and corresponding probabilities.
    logits = self.get_predicted_logits(state, graphs, rngs=None)

    # Get the mask for valid labels and graphs.
    mask = self.get_valid_mask(labels, graphs)

    # Compute the various metrics.
    loss = self.binary_cross_entropy_with_mask(
        logits=logits, labels=labels, mask=mask)

    return EvalMetrics.single_from_model_output(
        loss=loss, logits=logits, labels=labels, mask=mask)

  def evaluate(
      self, state: train_state.TrainState,
      datasets: Dict[str, tf.data.Dataset]) -> Dict[str, metrics.Collection]:
    """Evaluates the model on metrics over the specified splits."""

    # Loop over each split independently.
    eval_metrics = {}
    split_metrics = None

    # Loop over graphs.
    for graphs in datasets['validation'].as_numpy_iterator():
      split_metrics_update = self.evaluate_step(state, graphs)

      # Update metrics.
      if split_metrics is None:
        split_metrics = split_metrics_update
      else:
        split_metrics = split_metrics.merge(split_metrics_update)
    eval_metrics['validation'] = split_metrics

    return eval_metrics

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result['mean_average_precision'] > self.target_value

  def build_input_queue(self, data_rng: jax.random.PRNGKey, split: str,
                        data_dir: str, batch_size: int):
    self._datasets = input_pipeline.get_datasets(batch_size=batch_size)
    return iter(self._datasets['train'])

  @property
  def param_shapes(self):
    init_params, _ = self.init_model_fn(jax.random.PRNGKey(0))
    return jax.tree_map(lambda x: spec.ShapeTuple(x.shape), init_params)

  @property
  def target_value(self):
    return 0.25

  @property
  def loss_type(self):
    return spec.LossType.SOFTMAX_CROSS_ENTROPY

  @property
  def num_train_examples(self):
    return 350343

  @property
  def num_eval_examples(self):
    return 43793

  @property
  def train_mean(self):
    return 0.0

  @property
  def train_stddev(self):
    return 1.0

  def model_params_types(self):
    pass

  @property
  def max_allowed_runtime_sec(self):
    return 20000

  @property
  def eval_period_time_sec(self):
    return 200

  # Return whether or not a key in spec.ParameterContainer is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    pass

  def preprocess_for_train(self, selected_raw_input_batch: spec.Tensor,
                           selected_label_batch: spec.Tensor,
                           train_mean: spec.Tensor, train_stddev: spec.Tensor,
                           rng: spec.RandomState) -> spec.Tensor:
    del train_mean
    del train_stddev
    del rng
    return selected_raw_input_batch, selected_label_batch

  def preprocess_for_eval(self, raw_input_batch: spec.Tensor,
                          train_mean: spec.Tensor,
                          train_stddev: spec.Tensor) -> spec.Tensor:
    del train_mean
    del train_stddev
    return raw_input_batch

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    init_graphs = next(self._datasets['train'].as_numpy_iterator())
    init_graphs = self.replace_globals(init_graphs)
    init_net = models.GraphConvNet(deterministic=True)
    params = jax.jit(init_net.init)(rng, init_graphs)

    eval_net = models.GraphConvNet(deterministic=True)
    self._eval_state = train_state.TrainState.create(
        apply_fn=eval_net.apply,
        params=params,
        tx=optax.adam(learning_rate=1e-3))

    return params, models.GraphConvNet(deterministic=False)

  def model_fn(
      self, params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: spec.Tensor,
      model_state: spec.ModelAuxiliaryState, mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:

    return model_state.apply_fn(
        params, augmented_and_preprocessed_input_batch, rngs=rng), None

  def loss_fn(
      self,
      label_batch: spec.Tensor,  # Dense (not one-hot) labels.
      logits_batch: spec.Tensor) -> spec.Tensor:

    labels = label_batch.globals
    graphs = self.replace_globals(label_batch)
    mask = self.get_valid_mask(labels, graphs)
    loss = self.binary_cross_entropy_with_mask(
        logits=logits_batch, labels=labels, mask=mask)

    return jnp.where(mask, loss, 0)

  def output_activation_fn(self, logits_batch: spec.Tensor,
                           loss_type: spec.LossType) -> spec.Tensor:
    """Return the final activations of the model."""
    pass

  def eval_model(self, params: spec.ParameterContainer,
                 model_state: spec.ModelAuxiliaryState, rng: spec.RandomState,
                 data_dir: str):
    """Run a full evaluation of the model."""
    del data_dir

    eval_state = self._eval_state.replace(params=params)

    eval_metrics = self.evaluate(eval_state, self._datasets).compute()

    return eval_metrics

