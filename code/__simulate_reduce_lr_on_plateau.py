# -*- coding: utf-8 -*- noqa
"""
Created on Sun May 18 23:51:29 2025

@author: JoelT
"""


def simulate_reduce_lr_on_plateau(
    losses,
    mode='min',
    factor=0.5,
    patience=2,
    threshold=1e-5,
    threshold_mode='rel',
    cooldown=0,
    min_lr=0,
    eps=1e-8,
    initial_lr=0.0003,
    verbose=True
):
    lr = initial_lr
    best = float('inf') if mode == 'min' else -float('inf')
    num_bad_epochs = 0
    cooldown_counter = 0

    def is_better(current, best):
        if threshold_mode == 'rel':
            if mode == 'min':
                return current < best * (1 - threshold)
            else:
                return current > best * (1 + threshold)
        else:  # 'abs'
            if mode == 'min':
                return current < best - threshold
            else:
                return current > best + threshold

    print(f"{'Epoch':>5} | {'Loss':>8} | {'LR':>10} | {'Note'}")
    print("-" * 40)

    for epoch, loss in enumerate(losses):
        note = ""
        if cooldown_counter > 0:
            cooldown_counter -= 1
            note = "Cooldown"

        if is_better(loss, best):
            best = loss
            num_bad_epochs = 0
            note = "Improved"
        else:
            if cooldown_counter == 0:
                num_bad_epochs += 1

        if num_bad_epochs > patience:
            if lr - min_lr > eps:
                old_lr = lr
                lr = max(lr * factor, min_lr)
                cooldown_counter = cooldown
                num_bad_epochs = 0
                note = f"Reduced LR: {old_lr:.6f} → {lr:.6f}"
            else:
                note = "LR already at min"

        print(f"{epoch+1:5d} | {loss:8.6f} | {lr:10.6f} | {note}")


# Example losses (edit this to test your real loss values)
# example_losses = [
#     0.22, 0.18, 0.162, 0.1615, 0.1614, 0.1613,  # Stalled, should trigger
#     0.16, 0.1599, 0.1598,                      # Improved slightly
#     0.1598, 0.1597, 0.1597, 0.1596              # Stalls again
# ]

# simulate_reduce_lr_on_plateau(example_losses)
