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
import trimesh
from tqdm import tqdm

from jaxnerf.nerf import datasets
from jaxnerf.nerf import models
from jaxnerf.nerf import utils


class FLAGS:
    # You may want to modify these parameters!
    train_dir = '/home/ubuntu/models/jaxnerf_models/blender/lego'
    data_dir = '/home/ubuntu/data/nerf_synthetic/lego'
    config = 'configs/blender'
    grid_samples = 256
    pc_out_path = '/tmp/point_cloud.txt'

    # Default JaxNeRF parameters (probably best to avoid)
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


def display_tri_mesh(density_samples, threshold=50.):
    """Display the triangle mesh computed with Marching Cubes algorithm."""
    v, tri = fast_marching_cubes(density_samples, threshold=threshold)
    mesh = trimesh.Trimesh(v, tri)
    return mesh  # call mesh.show() in a jupyter notebook


def display_point_cloud(xyz, rgb=None, samples=None):
    """Display colored point cloud extracted from NeRF.

    Args:
        xyz: (N, 3) tensor of 3D points.
        rgb (optional): (N, 3) tensor mapping each point to an RGB color.
        samples (optional): number of random samples to display from the cloud.
    """
    import ipyvolume as ipv  # defer import
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    idx = None
    if samples is not None:
        idx = np.random.choice(len(x), size=samples, replace=False)

    fig = ipv.figure()
    if rgb is None:
        scatter = ipv.scatter(x[idx], y[idx], z[idx], color='#333', marker='sphere', size=0.8)
    else:
        scatter = ipv.scatter(x[idx], y[idx], z[idx], color=rgb[idx, :], marker='sphere', size=0.8)
    ipv.show()


def serialize_point_cloud(outpath, xyz, rgb=None):
    """Serialize point cloud to text file."""
    assert rgb is None or xyz.shape == rgb.shape
    out = path.expanduser(outpath)
    with open(out, 'w') as f:
        f.write('x y z r g b\n')
        for i, pt in enumerate(xyz):
            color = ''
            if rgb is not None:
                c = rgb[i]
                color = f'{c[0]} {c[1]} {c[2]}'
            line = f'{pt[0]} {pt[1]} {pt[2]} {color}\n'
            f.write(line)
    return out


def fast_marching_cubes(volume_density, threshold=50.):
    vertices, triangles = mcubes.marching_cubes(volume_density, threshold)
    return vertices, triangles


def normalize(x):
    """Normalize tensor output range to [0, 1]."""
    return (x - x.min()) / (x.max() - x.min())


def generate_uniform_volume(low, high, axis_samples):
    """Generate a uniform volume of 3D points.

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


def sample_volume_grid(volume_grid, render_fn, model_state, rng):
    """Samples raw density (sigma) and RGB value at each point in volume.

    Args:
        volume_grid: (length, height, width, 3) tensor of 3D points.
        render_fn: jitted JaxNeRF render function.
        model_state: pre-trained NeRF model parameters.
        rng: jax pseudorandom number generator.

    Returns:
        density: (length, height, width) tensor with density at each point.
        rgb: (length, height, width, 3) tensor with RGB at each point.
    """
    length, height, width, _ = volume_grid.shape
    num_samples = FLAGS.num_coarse_samples
    if FLAGS.randomized:
        num_samples = 3 * num_samples

    pred_tensors = []
    for i in tqdm(range(length)):
        frame = volume_grid[i]
        rays = utils.Rays(origins=frame,
                       directions=np.zeros_like(frame),
                       viewdirs=np.zeros_like(frame))

        rng, _ = jax.random.split(rng)
        pred_color, pred_disp, pred_acc, pred_sigma, pred_rgb = utils.render_image(
                functools.partial(render_fn, model_state.optimizer.target),
                rays,
                rng,
                FLAGS.dataset == "llff",
                chunk=FLAGS.chunk)
        del pred_color, pred_disp, pred_acc
        # We only care about the pointwise predictions for density and RGB
        pred_tensors.append((np.array(pred_sigma),
                             np.array(pred_rgb)))
    assert len(pred_tensors) == length

    sigma = []
    for (pred_sigma, pred_rgb) in pred_tensors:
        pred_sigma = np.maximum(pred_sigma[...,-1], 0.)
        sigma.append(pred_sigma)
    density = np.stack(sigma)

    colors = []
    for (pred_sigma, pred_rgb) in pred_tensors:
        pred_rgb = pred_rgb.reshape(height, width, num_samples, 3)[:, :, 0, :]
        colors.append(pred_rgb)
    rgb = np.stack(colors)

    return density, rgb


def generate_point_cloud(volume_density, volume_rgb, isovalue=50):
    """Generate a point cloud with Marching cubes preprocessing.

    Args:
        volume_density: (length, height, width, 1) density at each point.
        volume_rgb: (length, height, width, 3) tensor with RGB at each point.
        isovalue: Threshold for marching cubes preprocessing.

    Returns:
        xyz: (N, 3) list of 3D points.
        rgb: (N, 3) associated RGB colors for each of those points.
    """
    xyz, _ = fast_marching_cubes(volume_density, threshold=isovalue)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # The generated points may not be exactly quantized to the volume coords.
    qx = np.floor(x).astype(np.uint32)
    qy = np.floor(y).astype(np.uint32)
    qz = np.floor(z).astype(np.uint32)
    rgb = volume_rgb[qx, qy, qz]

    assert xyz.shape == rgb.shape
    return xyz, rgb


def init():
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
    return render_pfn, state


def main():
    print('Loading pre-trained model...')
    render_fn, state = init()
    print('Done.')

    rng = random.PRNGKey(2020082)
    volume_grid = npc.generate_uniform_volume(-1.2, 1.2, FLAGS.grid_samples)
    density, rgb = npc.sample_volume_grid(volume_grid, render_fn, state, rng)

    pc_xyz, pc_rgb = npc.generate_point_cloud(density, rgb, isovalue=50)
    npc.serialize_point_cloud(FLAGS.pc_out_path, pc_xyz, pc_rgb)
    print(f'Wrote point cloud to {FLAGS.pc_out_path}')


if __name__ == "__main__":
    main()