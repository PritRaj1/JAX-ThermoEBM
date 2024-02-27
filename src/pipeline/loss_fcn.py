import jax

from src.pipeline.sample_generate_fcns import sample_z

def ebm_loss(state, z_prior, z_posterior):
    """
    Function to compute the loss for the EBM model.
    """
    EBM_fwd = state.model_apply['EBM_apply']
    EBM_params = state.params['EBM_params']

    # Compute the energy of the posterior sample
    en_pos = EBM_fwd(EBM_params, z_posterior)

    # Compute the energy of the prior sample
    en_neg = EBM_fwd(EBM_params, z_prior)

    # Return the difference in energies
    return (en_pos - en_neg)

def gen_loss(state, x, z):
    """
    Function to compute the loss for the GEN model.
    """
    GEN_fwd = state.model_apply['GEN_apply']
    GEN_params = state.params['GEN_params']

    # Compute -log[ p_β(x | z) ]; max likelihood training
    x_pred = GEN_fwd(GEN_params, z) + (state.lkhood_sigma * jax.random.normal(state.key, x.shape))
    log_lkhood = (jax.linalg.norm(x-x_pred, axis=-1)**2) / (2.0 * state.lkhood_sigma**2)

    return log_lkhood 

class discretised_TI_loss_fcn():
    def __init__(self, loss_fcn, gen=False):
        """
        Class to compute the loss using Thermodynamic Integration.

        Args:
        - loss_fcn: the loss function to be used for the model
        - gen: boolean to indicate if the loss is for the generator model
        """
        self.loss_fcn = loss_fcn
        self.is_gen = gen

    def __call__(self, state, data):
        """
        Function to compute the loss using Thermodynamic Integration.
        Please see "discretised thermodynamic integration" using trapezoid rule
        in https://doi.org/10.1016/j.csda.2009.07.025 for details.

        Args:
        - state: current train state of the model
        - y: 
            - z_prior, if EBM loss
            - x data, if GEN loss

        Returns:
        - total_loss: the total loss for the entire thermodynamic integration loop
        """
        total_loss = 0
        temp_schedule = state.temp['schedule']

        # Thermodynamic Integration Loop
        for idx, t in enumerate(temp_schedule):
            state.temp['current'] = t

            z_prior, z_posterior = sample_z(state, data)

            # If the loss is for the generator model, y = x
            y = data if self.is_gen else z_prior

            loss_current = self.loss_fcn(state, y, z_posterior)

            # If there are more than one temperature in the schedule
            if len(state.temp['schedule']) > 1:

                # ∇T = t_i - t_{i-1}
                delta_T = t - temp_schedule[idx-1] if idx != 0 else 0

                # # 1/2 * (f(x_i) + f(x_{i-1})) * ∇T
                total_loss += (0.5 * (loss_current + total_loss) * delta_T)

            # Vanilla model
            else:
                total_loss += loss_current

        return total_loss


