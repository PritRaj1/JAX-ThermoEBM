from jax.lax import scan
from functools import partial

from src.pipeline.loss_computation.themo_integration_loop import EBM_loop, GEN_loop


def ThermoEBM_loss(key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd, temp_schedule):
    """
    Function to compute the themodynamic integration loss for the EBM model.
    Please see "discretised thermodynamic integration" using trapezoid rule
    in https://doi.org/10.1016/j.csda.2009.07.025 for details.

    To integrate over temperatures, we use the trapezoid rule:
    ∫[a, b] f(x) dx ≈ 1/2 * (f(a) + f(b)) * ∇T
    We accumulate this in a scan loop over the temperature schedule.
    This is then summed at the end to compute the integral over the temperature schedule.

    Args:
    - key: PRNG key
    - x: batch of data samples
    - EBM_params: energy-based model parameters
    - GEN_params: generator parameters
    - EBM_fwd: energy-based model forward pass, --immutable
    - GEN_fwd: generator forward pass, --immutable
    - temp_schedule: temperature schedule

    Returns:
    - total_loss: the total EBM loss for the entire thermodynamic integration loop, log(p_a(z))
    """

    # Wrap the themodynamic loop in a partial function to exploit partial immutability
    scan_loss = partial(
        EBM_loop,
        x=x,
        EBM_params=EBM_params,
        EBM_fwd=EBM_fwd,
        GEN_params=GEN_params,
        GEN_fwd=GEN_fwd,
    )

    # Scan over the temperature schedule to compute the loss at each temperature
    initial_state = (key, 0, 0)
    (_, _, _), temp_losses = scan(f=scan_loss, init=initial_state, xs=temp_schedule)

    # Sum the stacked losses over all temperature intervals to get integrated loss
    return temp_losses.sum()


def ThermoGEN_loss(key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd, temp_schedule):
    """
    Function to compute the themodynamic integration loss for the GEN model.
    Please see "discretised thermodynamic integration" using trapezoid rule
    in https://doi.org/10.1016/j.csda.2009.07.025 for details.

    To integrate over temperatures, we use the trapezoid rule:
    ∫[a, b] f(x) dx ≈ 1/2 * (f(a) + f(b)) * ∇T
    We accumulate this in a scan loop over the temperature schedule.
    This is then summed at the end to compute the integral over the temperature schedule.

    Args:
    - key: PRNG key
    - x: batch of data samples
    - EBM_params: energy-based model parameters
    - GEN_params: generator parameters
    - EBM_fwd: energy-based model forward pass, --immutable
    - GEN_fwd: generator forward pass, --immutable
    - temp_schedule: temperature schedule

    Returns:
    - total_loss: the total GEN loss for the entire thermodynamic integration loop, log(p_β(x|z))
    """

    # Wrap the themodynamic loop in a partial function to exploit partial immutability
    scan_loss = partial(
        GEN_loop,
        x=x,
        EBM_params=EBM_params,
        EBM_fwd=EBM_fwd,
        GEN_params=GEN_params,
        GEN_fwd=GEN_fwd,
    )

    # Scan over the temperature schedule to compute the loss at each temperature
    initial_state = (key, 0, 0)
    (_, _, _), temp_losses = scan(f=scan_loss, init=initial_state, xs=temp_schedule)

    # Sum the stacked losses over all temperature intervals to get integrated loss
    return temp_losses.sum()
