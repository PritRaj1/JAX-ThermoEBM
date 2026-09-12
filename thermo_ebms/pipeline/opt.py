from collections.abc import Callable

import optax

from ..config import AdamWConfig, OptConfig


def network_opt(config: AdamWConfig, total_steps: int) -> Callable:
	warmup_steps = int(config.warmup_fraction * total_steps)

	schedule = optax.warmup_cosine_decay_schedule(
		init_value=0.0,
		peak_value=config.lr_init,
		warmup_steps=warmup_steps,
		decay_steps=total_steps,
		end_value=config.lr_end,
	)

	return optax.chain(
		optax.clip_by_global_norm(1.0),
		optax.adamw(
			learning_rate=schedule,
			b1=config.beta1,
			b2=config.beta2,
			weight_decay=config.weight_decay,
		),
	)


def label_fn(path: tuple) -> str:
	path_str = str(path).lower()
	if "ebm" in path_str:
		return "ebm"
	if "kan" in path_str:
		return "kan"
	return "gen"


def coupled_opt(
	config: OptConfig,
	total_steps: int,
):

	return optax.multi_transform(
		{
			"ebm": network_opt(config.ebm, total_steps),
			"gen": network_opt(config.gen, total_steps),
			"kan": network_opt(config.kaem, total_steps),
		},
		label_fn,
	)
