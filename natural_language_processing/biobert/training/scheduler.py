from transformers import get_polynomial_decay_schedule_with_warmup


def get_learning_rate_scheduler(optimizer, n_steps: int, warm_up_prop: float):
    return get_polynomial_decay_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(warm_up_prop * n_steps),
        num_training_steps=n_steps,
    )
