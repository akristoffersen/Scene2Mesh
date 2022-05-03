import functools
import sys
from os import path
from PIL import Image

from absl import app
from absl import flags
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from jax import random
import mcubes
import numpy as np
from tqdm import tqdm

sys.path.append('..')
from jaxnerf.nerf import datasets
from jaxnerf.nerf import models
from jaxnerf.nerf import utils


class FLAGS:
    # You may want to modify these parameters!
    train_dir = '/home/ubuntu/models/jaxnerf_models/blender/lego'
    data_dir = '/home/ubuntu/data/nerf_synthetic/lego'
    config = 'configs/blender'

    dataset = 'blender'
    batching = 'single_image'
    white_bkgd = True
    batch_size = 1024
    factor = 2
    spherify = False
    render_path = False
    llffhold = 8
    use_pixel_centers = False
    model = 'nerf'
    near = 2.
    far = 6.
    net_depth = 8
    net_width = 256
    net_depth_condition = 1
    net_width_condition = 128
    weight_decay_mult = 0
    skip_layer = 4
    num_rgb_channels = 3
    num_sigma_channels = 1
    randomized = True
    min_deg_point = 0
    max_deg_point = 10
    deg_view = 4
    num_coarse_samples = 64
    num_fine_samples = 128
    use_viewdirs = True
    noise_std = None
    lindisp = False
    net_activation = 'relu'
    rgb_activation = 'sigmoid'
    sigma_activation = 'relu'
    legacy_posenc_order = False

    lr_init = 5e-4
    lr_final = 5e-6
    lr_delay_steps = 0
    lr_delay_mult = 1.
    grad_max_norm = 0.
    grad_max_val = 0.
    max_steps = 1000000
    save_every = 10000
    print_every = 100
    render_every = 5000
    gc_every = 10000

    eval_once = True
    save_output = True
    chunk = 4096


def display_pil_horizontal(ims):
    """Stitch list of PIL Image objects horizontally as one image."""
    dst = Image.new('RGB', (ims[0].width * len(ims), ims[0].height))
    for i, im in enumerate(ims):
        dst.paste(im, (i * im.width, 0))
    return dst


def display_tri_mesh(density_samples, color_samples):
    """Display the triangle mesh computed with Marching Cubes algorithm."""
    pass


def display_point_cloud(density_samples, color_samples):
    """Display colored point cloud extracted from NeRF."""
    pass


def normalize(x):
    """Normalize tensor output range to [0, 1]."""
    return (x - x.min()) / (x.max() - x.min())


def generate_uniform_volume(low, high, axis_samples):
    """Generate a uniform volume of points.

    Args:
        low: minimum value per axis.
        high: maximum value per axis.
        axis_samples: number of samples per axis.

    Returns:
        Tensor with shape (axis_samples, axis_samples, axis_samples, 3)
    """
    t = np.linspace(low, high, axis_samples)
    query_pts = np.stack(np.meshgrid(t, t, t), -1).astype(np.float32)
    return query_pts


def sample_volume_grid(volume_grid, render_fn, model_state):
    pass


def generate_point_cloud(density_samples, color_samples):
    pass


def main():
    rng = random.PRNGKey(20200823)

    # The code for loading a model and jitting the render_fn came from
    # https://github.com/google-research/google-research/tree/master/jaxnerf
    dataset = datasets.get_dataset("test", FLAGS)
    rng, key = random.split(rng)
    model, init_variables = models.get_model(key, dataset.peek(), FLAGS)
    optimizer = flax.optim.Adam(FLAGS.lr_init).create(init_variables)
    state = utils.TrainState(optimizer=optimizer)
    del optimizer, init_variables

    # Rendering is forced to be deterministic even if training was randomized,
    # as this eliminates "speckle" artifacts.
    def render_fn(variables, key_0, key_1, rays):
        return jax.lax.all_gather(
            model.apply(variables, key_0, key_1, rays, False), axis_name="batch")

    # pmap over only the data input (ray-level parallelism).
    render_pfn = jax.pmap(
      render_fn,
      in_axes=(None, None, None, 0),
      donate_argnums=3,
      axis_name="batch",
    )

    state = checkpoints.restore_checkpoint(FLAGS.train_dir, state)


if __name__ == "__main__":
    main()