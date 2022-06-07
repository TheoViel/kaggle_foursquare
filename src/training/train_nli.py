import gc
import time
import torch
import numpy as np
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score
from transformers import get_linear_schedule_with_warmup

from data.loader import define_loaders

from training.losses import NLILoss
from training.optim import custom_params, define_optimizer, trim_tensors


def evaluate(model, val_loader, data_config, loss_fct, loss_config):
    model.eval()
    avg_val_loss = 0.
    preds = []
    with torch.no_grad():
        for data in val_loader:
            ids, = trim_tensors([data["ids"]], pad_token=data_config["pad_token"])

            y_pred = model(
                ids.cuda(),
                # token_type_ids.cuda(),
                fts=data['fts'].cuda()
            )

            loss = loss_fct(y_pred.detach(), data["target"].cuda()).mean()
            avg_val_loss += loss / len(val_loader)

            if loss_config['activation'] == "sigmoid":
                y_pred = y_pred.sigmoid().view(-1)
            elif loss_config['activation'] == "softmax":
                y_pred = y_pred.softmax(-1)

            preds.append(y_pred.detach().cpu().numpy())

    return avg_val_loss, np.concatenate(preds)


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

    loss_fct = NLILoss(loss_config, device=device)

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
        for data in tqdm(train_loader):
            ids, = trim_tensors([data["ids"]], pad_token=data_config["pad_token"])

            with torch.cuda.amp.autocast(enabled=use_fp16):
                y_pred = model(
                    ids.cuda(),
                    # token_type_ids.cuda(),
                    fts=data["fts"].cuda(),
                )
                loss = loss_fct(y_pred, data["target"].cuda()).mean() / acc_steps

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

                avg_val_loss, preds = evaluate(
                    model, val_loader, data_config, loss_fct, loss_config
                )

                auc = roc_auc_score(val_dataset.target, preds)

                dt = time.time() - start_time
                lr = scheduler.get_last_lr()[0]

                s = f"Epoch {epoch:02d}/{epochs:02d}  (step {step // acc_steps:04d})\t"
                s = s + f"lr={lr:.1e}\t t={dt:.0f}s\t loss={np.mean(avg_losses):.3f}"
                s = s + f"\t val_loss={avg_val_loss:.3f}" if avg_val_loss else s
                s = s + f"\t auc={auc:.3f}" if auc else s
                print(s)

                start_time = time.time()
                avg_losses = []
                model.train()

    del (train_loader, val_loader, optimizer)
    torch.cuda.empty_cache()
    gc.collect()

    return preds
