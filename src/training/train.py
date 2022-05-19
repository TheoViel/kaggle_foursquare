import gc
import time
import torch
import numpy as np

from transformers import get_linear_schedule_with_warmup

from data.loader import define_loaders

from training.losses import TripletLoss
from training.optim import custom_params, define_optimizer, trim_tensors


def evaluate(model, val_loader, data_config, loss_fct):
    model.eval()
    avg_val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            ids = torch.cat([data['ref_ids'], data['pos_ids'], data['neg_ids']], 0)
            ids, = trim_tensors([ids], pad_token=data_config["pad_token"])

            fts = data['fts'].transpose(0, 1).reshape(-1, data['fts'].size(-1))

            _, pred = model(
                ids.cuda(),
                # token_type_ids.cuda(),
                fts=fts.cuda()
            )

            loss = loss_fct(pred).mean()
            avg_val_loss += loss / len(val_loader)

    return avg_val_loss


def fit(
    model,
    train_dataset,
    val_dataset,
    data_config,
    loss_config,
    optimizer_config,
    epochs=1,
    acc_steps=1,
    verbose_eval=1,
    device="cuda",
    use_fp16=False,
    gradient_checkpointing=False,
):
    """
    Training functiong.
    TODO

    Args:

    Returns:
    """
    scaler = torch.cuda.amp.GradScaler()

    if gradient_checkpointing:
        model.transformer.gradient_checkpointing_enable()

    opt_params = custom_params(
        model,
        lr=optimizer_config["lr"],
        weight_decay=optimizer_config["weight_decay"],
        lr_transfo=optimizer_config["lr_transfo"],
        lr_decay=optimizer_config["lr_decay"],
    )
    optimizer = define_optimizer(
        optimizer_config["name"],
        opt_params,
        lr=optimizer_config["lr"],
        betas=optimizer_config["betas"],
    )
    optimizer.zero_grad()

    loss_fct = TripletLoss(loss_config, device=device)

    train_loader, val_loader = define_loaders(
        train_dataset,
        val_dataset,
        batch_size=data_config["batch_size"],
        use_len_sampler=data_config["use_len_sampler"],
        pad_token=data_config["pad_token"],
    )

    # LR Scheduler
    num_training_steps = epochs * len(train_loader) // acc_steps
    num_warmup_steps = int(optimizer_config["warmup_prop"] * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    step = 1
    avg_losses = []
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        for data in train_loader:
            ids = torch.cat([data['ref_ids'], data['pos_ids'], data['neg_ids']])
            ids, = trim_tensors([ids], pad_token=data_config["pad_token"])

            fts = data['fts'].transpose(0, 1).reshape(-1, data['fts'].size(-1))

            with torch.cuda.amp.autocast(enabled=use_fp16):
                _, pred = model(
                    ids.cuda(),
                    # token_type_ids.cuda(),
                    fts=fts.cuda(),
                )
                loss = loss_fct(pred).mean() / acc_steps

            scaler.scale(loss).backward()
            avg_losses.append(loss.item() * acc_steps)

            if step % acc_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    optimizer_config["max_grad_norm"],
                    error_if_nonfinite=False,
                )

                scaler.step(optimizer)
                scale = scaler.get_scale()
                scaler.update()

                if scale == scaler.get_scale():
                    scheduler.step()

                for param in model.parameters():
                    param.grad = None

            step += 1
            if step % (verbose_eval * acc_steps) == 0 or step - 1 == epochs * len(
                train_loader
            ):
                if 0 <= epochs * len(train_loader) - step < verbose_eval * acc_steps:
                    continue

                avg_val_loss = evaluate(
                    model, val_loader, data_config, loss_fct
                )

                dt = time.time() - start_time
                lr = scheduler.get_last_lr()[0]

                s = f"Epoch {epoch:02d}/{epochs:02d}  (step {step // acc_steps:04d})\t"
                s = s + f"lr={lr:.1e}\t t={dt:.0f}s\t loss={np.mean(avg_losses):.3f}"
                s = s + f"\t val_loss={avg_val_loss:.3f}" if avg_val_loss else s
                print(s)

                start_time = time.time()
                avg_losses = []
                model.train()

    del (train_loader, val_loader, optimizer)
    torch.cuda.empty_cache()
    gc.collect()

    return
