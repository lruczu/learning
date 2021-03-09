from transformers import AdamW


def get_optimizer_for_model(
    model,
    weight_decay: float,
    lr: float,
):
    def apply_decay(name: str) -> bool:
        if 'bias' in name:
            return False
        elif 'LayerNorm.weight' in name:
            return False
        return True

    params = [
        {'params':
            [tensor for name, tensor in model.named_parameters() if apply_decay(name)],
            'weight_decay': weight_decay,
        },
        {'params':
             [tensor for name, tensor in model.named_parameters() if not apply_decay(name)],
            'weight_decay': 0.0,
        }
    ]

    return AdamW(
        params=params,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-6,
    )
