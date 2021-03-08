from transformers import AdamW


def get_optimizer_for_model(model):
    def apply_decay(name: str) -> bool:
        if 'bias' in name:
            return False
        elif 'LayerNorm.weight' in name:
            return False
        return True

    params = [
        {'params':
            [tensor for name, tensor in model.named_parameters() if apply_decay(name)],
            'weight_decay': 0.01,
        },
        {'params':
             [tensor for name, tensor in model.named_parameters() if not apply_decay(name)],
            'weight_decay': 0.0,
        }
    ]

    return AdamW(
        params=params,
        lr=5e-5,
        betas=(0.9, 0.999),
        eps=1e-6,
    )
