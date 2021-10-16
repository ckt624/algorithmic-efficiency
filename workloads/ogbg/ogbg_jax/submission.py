"""Training algorithm track submission functions for WMT."""
from typing import Dict, Iterator, List, Optional, Tuple

from clu import metrics
from flax.training import common_utils
from flax.training import train_state
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import spec


def binary_cross_entropy_with_mask(logits: jnp.ndarray, labels: jnp.ndarray,
                                   mask: jnp.ndarray):
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


def replace_globals(graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
  """Replaces the globals attribute with a constant feature for each graph."""
  return graphs._replace(globals=jnp.ones([graphs.n_node.shape[0], 1]))


def get_predicted_logits(state: train_state.TrainState,
                         graphs: jraph.GraphsTuple,
                         rngs: Optional[Dict[str, jnp.ndarray]]) -> jnp.ndarray:
  """Get predicted logits from the network for input graphs."""
  pred_graphs = state.apply_fn(state.params, graphs, rngs=rngs)
  logits = pred_graphs.globals
  return logits


def get_valid_mask(labels: jnp.ndarray,
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
def train_step(
    state: train_state.TrainState, graphs: jraph.GraphsTuple,
    rngs: Dict[str, jnp.ndarray]
) -> Tuple[train_state.TrainState, metrics.Collection]:
  """Performs one update step over the current batch of graphs."""

  def loss_fn(params, graphs):
    curr_state = state.replace(params=params)

    # Extract labels.
    labels = graphs.globals

    # Replace the global feature for graph classification.
    graphs = replace_globals(graphs)

    # Compute logits and resulting loss.
    logits = get_predicted_logits(curr_state, graphs, rngs)
    mask = get_valid_mask(labels, graphs)
    loss = binary_cross_entropy_with_mask(
        logits=logits, labels=labels, mask=mask)
    mean_loss = jnp.sum(jnp.where(mask, loss, 0)) / jnp.sum(mask)

    return mean_loss, (loss, logits, labels, mask)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  _, grads = grad_fn(state.params, graphs)
  state = state.apply_gradients(grads=grads)

  return state


def get_batch_size(workload_name):
  batch_sizes = {'ogbg_jax': 256}
  return batch_sizes[workload_name]


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparamters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  del rng
  del workload

  return train_state.TrainState.create(
      apply_fn=model_state.apply,
      params=model_params,
      tx=optax.adam(learning_rate=hyperparameters.learning_rate))


def update_params(
    workload: spec.Workload,
    current_param_container: spec.ParameterContainer,
    current_params_types: spec.ParameterTypeTree,
    model_state: spec.ModelAuxiliaryState,
    hyperparameters: spec.Hyperparamters,
    input_batch: spec.Tensor,
    label_batch: spec.Tensor,
    # This will define the output activation via `output_activation_fn`.
    loss_type: spec.LossType,
    optimizer_state: spec.OptimizerState,
    eval_results: List[Tuple[int, float]],
    global_step: int,
    rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params)."""
  del workload
  del current_param_container
  del current_params_types
  del eval_results
  del global_step
  del model_state
  del loss_type
  del hyperparameters
  del label_batch

  state = train_step(optimizer_state, input_batch, rngs={'dropout': rng})

  return state, state.params, None


# Not allowed to update the model parameters, hyperparameters, global step, or
# optimzier state.
def data_selection(workload: spec.Workload,
                   input_queue: Iterator[Tuple[spec.Tensor, spec.Tensor]],
                   optimizer_state: spec.OptimizerState,
                   current_param_container: spec.ParameterContainer,
                   hyperparameters: spec.Hyperparamters, global_step: int,
                   rng: spec.RandomState) -> Tuple[spec.Tensor, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.

  Each element of the queue is a single training example and label.

  We left out `current_params_types` because we do not believe that it would
  # be necessary for this function.

  Return a tuple of input label batches.
  """
  del optimizer_state
  del current_param_container
  del global_step
  del rng
  del hyperparameters
  del workload

  return common_utils.shard(jax.tree_map(np.asarray, next(input_queue))), None

