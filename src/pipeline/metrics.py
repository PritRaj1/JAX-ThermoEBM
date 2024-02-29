import configparser
import torch
import jax.numpy as jnp
from torchvision.utils import make_grid

# Metrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])

fid = FrechetInceptionDistance(feature=64, normalize=True)  # FID metric
mifid = MemorizationInformedFrechetInceptionDistance(feature=64, normalize=True)  # MI-FID metric
kid = KernelInceptionDistance(feature=64, subset_size=batch_size, normalize=True)  # KID metric
lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)  # LPIPS metric

def profile_image(x, x_pred, writer, epoch):
    # # NHWC -> NCHW
    x = jnp.transpose(x, (0, 3, 2, 1))
    x_pred = jnp.transpose(x_pred, (0, 3, 2, 1))

    x = torch.from_numpy(x)
    x_pred = torch.from_numpy(x_pred)

    # Convert for [-1, 1] to [0, 1], image probailities
    x_metric = ((x + 1) / 2).reshape(-1, x.shape[1], x.shape[2], x.shape[3])
    gen_metric = (x_pred + 1) / 2

    # FID score
    fid.update(x_metric, real=True)
    fid.update(gen_metric, real=False)
    fid_score = fid.compute()

    # MI-FID score
    mifid.update(x_metric, real=True)
    mifid.update(gen_metric, real=False)
    mifid_score = mifid.compute()

    # KID score
    kid.update(x_metric, real=True)
    kid.update(gen_metric, real=False)
    kid_score = kid.compute()[0]

    # LPIPS score
    lpips_score = lpips(x_metric, gen_metric)

    writer.add_scalar("val_FID", fid_score, epoch)
    writer.add_scalar("val_MI-FID", mifid_score, epoch)
    writer.add_scalar("val_KID", kid_score, epoch)
    writer.add_scalar("val_LPIPS", lpips_score, epoch)

    # Log a grid of 4x4 images
    grid = make_grid(x_pred[:16], nrow=4)
    writer.add_image("Generated Images", grid, epoch)
