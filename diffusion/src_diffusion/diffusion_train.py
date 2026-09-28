"""
Author: Swapan Mallick
Date : 10 March 2025
TrainLoop that uses diffusion.training_losses(...) and passes
model_kwargs={'cond': era5_batch} so UNet receives cross-attention conditioning.
"""

import os
import torch
from torch.optim import AdamW
import numpy as np
from copy import deepcopy
from torch.amp import GradScaler, autocast
# from torch.cuda.amp import GradScaler, autocast

from . import diffusion_dist, logger


class TrainLoop:
    def __init__(
        self,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval=1,
        save_interval=2,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        steps=50000,
        device=None,
        outdir="outputs",
        weight_decay=0.0,
        schedule_sampler=None,
        lr_anneal_steps=0,
        cond_dropout_prob=0.0,
    ):

        print("=" * 80, flush=True)
        print("ENTERING TrainLoop.__init__()", flush=True)
        print("=" * 80, flush=True)

        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.lr = lr
        self.steps = int(steps)

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"TRAIN INIT 0: Device = {self.device}", flush=True)
        print(f"TRAIN INIT 0: Total steps = {self.steps}", flush=True)
        print(f"TRAIN INIT 0: Batch size = {batch_size}", flush=True)
        print(f"TRAIN INIT 0: Microbatch = {microbatch}", flush=True)

        # -----------------------------------------------------
        # Optimizer
        # -----------------------------------------------------
        print(
            "TRAIN INIT 1: Creating AdamW optimizer...",
            flush=True
        )

        self.opt = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=weight_decay
        )

        print(
            "TRAIN INIT 2: AdamW optimizer created successfully.",
            flush=True
        )

        self.save_interval = int(save_interval)
        self.log_interval = int(log_interval)

        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

        print(
            f"TRAIN INIT 2A: Output directory = {self.outdir}",
            flush=True
        )

        self.schedule_sampler = schedule_sampler

        self.microbatch = (
            microbatch if microbatch > 0 else None
        )

        self.cond_dropout_prob = float(
            cond_dropout_prob
        )

        # -----------------------------------------------------
        # FP16
        # -----------------------------------------------------
        self.use_fp16 = (
            use_fp16
            and torch.cuda.is_available()
        )

        print(
            f"TRAIN INIT 2B: use_fp16 = {self.use_fp16}",
            flush=True
        )

        self.scaler = GradScaler(
            "cuda",
            enabled=self.use_fp16
        )

        print(
            "TRAIN INIT 2C: GradScaler created.",
            flush=True
        )

        # -----------------------------------------------------
        # EMA model
        # -----------------------------------------------------
        self.ema_rate = float(ema_rate)

        print(
            f"TRAIN INIT 3: EMA rate = {self.ema_rate}",
            flush=True
        )

        print(
            "TRAIN INIT 3: Starting EMA model deepcopy...",
            flush=True
        )

        self.ema_model = deepcopy(
            self.model
        )

        print(
            "TRAIN INIT 4: EMA model deepcopy finished.",
            flush=True
        )

        print(
            "TRAIN INIT 4A: Moving EMA model to device...",
            flush=True
        )

        self.ema_model = (
            self.ema_model.to(self.device)
        )

        print(
            "TRAIN INIT 5: EMA model moved to device.",
            flush=True
        )

        self.ema_model.eval()

        print(
            "SWAPAN1 TRAIN",
            flush=True
        )

        # -----------------------------------------------------
        # Diffusion timestep information
        # -----------------------------------------------------
        self.num_timesteps = getattr(
            self.diffusion,
            "num_timesteps",
            None
        )

        print(
            f"TRAIN INIT 6: diffusion num_timesteps = "
            f"{self.num_timesteps}",
            flush=True
        )

        if (
            self.num_timesteps is None
            and hasattr(
                self.diffusion,
                "use_timesteps"
            )
        ):
            try:
                self.num_timesteps = (
                    int(
                        max(
                            self.diffusion.use_timesteps
                        )
                    )
                    + 1
                )

            except Exception:
                self.num_timesteps = None

        print(
            f"TRAIN INIT 7: Final num_timesteps = "
            f"{self.num_timesteps}",
            flush=True
        )

        print("=" * 80, flush=True)
        print(
            "TrainLoop.__init__() COMPLETE",
            flush=True
        )
        print("=" * 80, flush=True)

    def _is_sde_diffusion(self):

        return hasattr(
            self.diffusion,
            "sde_type"
        )

    def _sample_timesteps_and_weights(
        self,
        batch_size,
        device
    ):

        if self._is_sde_diffusion():

            t = (
                torch.rand(
                    batch_size,
                    device=device
                )
                * 0.999
                + 0.001
            )

            weights = torch.ones(
                batch_size,
                device=device
            )

            return t, weights

        if self.schedule_sampler is not None:

            t, weights = (
                self.schedule_sampler.sample(
                    batch_size,
                    device
                )
            )

            return (
                t.long(),
                weights.float()
            )

        if self.num_timesteps is None:

            raise ValueError(
                "Cannot sample timesteps."
            )

        t = torch.randint(
            0,
            self.num_timesteps,
            (batch_size,),
            device=device
        )

        weights = torch.ones(
            batch_size,
            device=device
        )

        return t, weights

    def _update_ema(self):

        with torch.no_grad():

            for ema_p, p in zip(
                self.ema_model.parameters(),
                self.model.parameters()
            ):

                ema_p.data.mul_(
                    self.ema_rate
                ).add_(
                    p.data,
                    alpha=1 - self.ema_rate
                )

    def run_loop(self):

        print("=" * 80, flush=True)
        print(
            "RUN LOOP 1: Entering run_loop()",
            flush=True
        )
        print("=" * 80, flush=True)

        print(
            "RUN LOOP 2: Moving/checking model on device...",
            flush=True
        )

        self.model.to(
            self.device
        )

        self.model.train()

        print(
            "RUN LOOP 3: Model is ready for training.",
            flush=True
        )

        step = 0

        loader = self.data

        infinite_loader = (
            not hasattr(
                loader,
                "__len__"
            )
        )

        print(
            f"RUN LOOP 4: DataLoader has __len__ = "
            f"{hasattr(loader, '__len__')}",
            flush=True
        )

        if hasattr(loader, "__len__"):

            try:
                print(
                    f"RUN LOOP 4A: Number of batches "
                    f"in DataLoader = {len(loader)}",
                    flush=True
                )

            except Exception as e:

                print(
                    f"RUN LOOP 4A: Could not determine "
                    f"DataLoader length: {e}",
                    flush=True
                )

        print(
            f"RUN LOOP 5: infinite_loader = "
            f"{infinite_loader}",
            flush=True
        )

        print(
            "RUN LOOP 6: About to enter training while-loop.",
            flush=True
        )

        while step < self.steps:

            print(
                f"\nRUN LOOP 7: Starting DataLoader "
                f"iteration at step {step}",
                flush=True
            )

            print(
                "RUN LOOP 8: Waiting for batch from DataLoader...",
                flush=True
            )

            for batch in loader:

                print(
                    f"RUN LOOP 9: Received batch "
                    f"from DataLoader at step {step}",
                    flush=True
                )

                if step >= self.steps:
                    break

                # -------------------------------------------------
                # Decode batch
                # -------------------------------------------------
                if (
                    isinstance(
                        batch,
                        (list, tuple)
                    )
                    and len(batch) >= 2
                ):

                    target_batch = batch[0]
                    cond_batch = batch[1]

                elif (
                    isinstance(
                        batch,
                        dict
                    )
                    and "era5" in batch
                    and "cerra" in batch
                ):

                    target_batch = (
                        batch["cerra"]
                    )

                    cond_batch = (
                        batch["era5"]
                    )

                else:

                    raise ValueError(
                        "Data loader must yield "
                        "(era5, cerra) tuples or dict."
                    )

                if step == 0:

                    print(
                        f"RUN LOOP 10: target_batch "
                        f"CPU shape = {target_batch.shape}",
                        flush=True
                    )

                    print(
                        f"RUN LOOP 10: cond_batch "
                        f"CPU shape = {cond_batch.shape}",
                        flush=True
                    )

                # -------------------------------------------------
                # Move batch to GPU
                # -------------------------------------------------
                print(
                    f"RUN LOOP 11: Moving batch to "
                    f"{self.device} at step {step}",
                    flush=True
                )

                target_batch = (
                    target_batch.to(
                        self.device
                    )
                )

                cond_batch = (
                    cond_batch.to(
                        self.device
                    )
                )

                print(
                    f"RUN LOOP 12: Batch moved to "
                    f"{self.device}",
                    flush=True
                )

                # -------------------------------------------------
                # Check finite data
                # -------------------------------------------------
                print(
                    f"RUN LOOP 13: Checking finite "
                    f"input values at step {step}",
                    flush=True
                )

                if (
                    not torch.isfinite(
                        target_batch
                    ).all()
                    or not torch.isfinite(
                        cond_batch
                    ).all()
                ):

                    print(
                        f"WARNING: Non-finite input "
                        f"detected at step {step}. "
                        f"Skipping batch.",
                        flush=True
                    )

                    continue

                print(
                    f"RUN LOOP 14: Input values "
                    f"are finite.",
                    flush=True
                )

                B = target_batch.shape[0]

                micro = (
                    self.microbatch
                    or B
                )

                if step == 0:

                    print(
                        f"RUN LOOP 15: Batch size B = {B}",
                        flush=True
                    )

                    print(
                        f"RUN LOOP 15: Microbatch = {micro}",
                        flush=True
                    )

                # -------------------------------------------------
                # Zero gradients
                # -------------------------------------------------
                self.opt.zero_grad(
                    set_to_none=True
                )

                total_loss = 0.0
                num_microbatches = 0

                print(
                    f"RUN LOOP 16: Starting microbatch "
                    f"loop at step {step}",
                    flush=True
                )

                # -------------------------------------------------
                # Microbatch loop
                # -------------------------------------------------
                for i in range(
                    0,
                    B,
                    micro
                ):

                    print(
                        f"RUN LOOP 17: Processing "
                        f"microbatch i={i} at step {step}",
                        flush=True
                    )

                    xb = target_batch[
                        i:i + micro
                    ]

                    cond_slice = cond_batch[
                        i:i + micro
                    ]

                    if (
                        cond_slice is not None
                        and self.cond_dropout_prob > 0
                    ):

                        drop_mask = (
                            torch.rand(
                                cond_slice.shape[0],
                                device=cond_slice.device
                            )
                            < self.cond_dropout_prob
                        )

                        if drop_mask.any():

                            cond_slice = (
                                cond_slice.clone()
                            )

                            cond_slice[
                                drop_mask
                            ] = 0.0

                    print(
                        f"RUN LOOP 18: Sampling "
                        f"timesteps at step {step}",
                        flush=True
                    )

                    t, weights = (
                        self._sample_timesteps_and_weights(
                            xb.shape[0],
                            device=self.device
                        )
                    )

                    if step == 0:

                        print(
                            f"RUN LOOP 19: xb shape = "
                            f"{xb.shape}",
                            flush=True
                        )

                        print(
                            f"RUN LOOP 19: cond_slice shape = "
                            f"{cond_slice.shape}",
                            flush=True
                        )

                        print(
                            f"RUN LOOP 19: timestep shape = "
                            f"{t.shape}",
                            flush=True
                        )

                    # -------------------------------------------------
                    # Forward diffusion loss
                    # -------------------------------------------------
                    print(
                        f"RUN LOOP 20: Calling "
                        f"diffusion.training_losses() "
                        f"at step {step}",
                        flush=True
                    )

                    with autocast(
                        "cuda",
                        enabled=self.use_fp16
                    ):

                        losses = (
                            self.diffusion.training_losses(
                                self.model,
                                xb,
                                t,
                                model_kwargs={
                                    "cond": cond_slice
                                }
                            )
                        )

                        print(
                            f"RUN LOOP 21: "
                            f"training_losses() returned "
                            f"at step {step}",
                            flush=True
                        )

                        loss = (
                            losses["loss"]
                            * weights
                        ).mean()

                    print(
                        f"RUN LOOP 22: Loss computed = "
                        f"{loss.item()}",
                        flush=True
                    )

                    if not torch.isfinite(
                        loss
                    ):

                        print(
                            f"WARNING: Non-finite loss "
                            f"at step {step}, "
                            f"microbatch {i}.",
                            flush=True
                        )

                        continue

                    # -------------------------------------------------
                    # Backward
                    # -------------------------------------------------
                    print(
                        f"RUN LOOP 23: Starting backward "
                        f"at step {step}",
                        flush=True
                    )

                    self.scaler.scale(
                        loss
                    ).backward()

                    print(
                        f"RUN LOOP 24: Backward finished "
                        f"at step {step}",
                        flush=True
                    )

                    total_loss += (
                        loss.item()
                    )

                    num_microbatches += 1

                # -------------------------------------------------
                # Validate microbatches
                # -------------------------------------------------
                if num_microbatches == 0:

                    print(
                        f"Warning: no valid microbatches "
                        f"at step {step}, "
                        f"skipping optimizer step",
                        flush=True
                    )

                    self.opt.zero_grad(
                        set_to_none=True
                    )

                    step += 1
                    continue

                avg_loss = (
                    total_loss
                    / num_microbatches
                )

                print(
                    f"RUN LOOP 25: Average loss = "
                    f"{avg_loss}",
                    flush=True
                )

                # -------------------------------------------------
                # Gradient processing
                # -------------------------------------------------
                print(
                    f"RUN LOOP 26: Unscaling gradients "
                    f"at step {step}",
                    flush=True
                )

                self.scaler.unscale_(
                    self.opt
                )

                print(
                    f"RUN LOOP 27: Calculating/clipping "
                    f"gradient norm",
                    flush=True
                )

                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        1.0
                    )
                )

                print(
                    f"RUN LOOP 28: grad_norm = "
                    f"{grad_norm.item()}",
                    flush=True
                )

                # -------------------------------------------------
                # Check gradient norm
                # -------------------------------------------------
                if not torch.isfinite(
                    grad_norm
                ):

                    print(
                        f"Warning: Non-finite grad norm "
                        f"at step {step}, "
                        f"skipping optimizer step",
                        flush=True
                    )

                    self.opt.zero_grad(
                        set_to_none=True
                    )

                    step += 1
                    continue

                # -------------------------------------------------
                # Optimizer
                # -------------------------------------------------
                print(
                    f"RUN LOOP 29: Optimizer step "
                    f"starting at step {step}",
                    flush=True
                )

                self.scaler.step(
                    self.opt
                )

                self.scaler.update()

                print(
                    f"RUN LOOP 30: Optimizer step "
                    f"finished at step {step}",
                    flush=True
                )

                # -------------------------------------------------
                # EMA update
                # -------------------------------------------------
                print(
                    f"RUN LOOP 31: Updating EMA "
                    f"at step {step}",
                    flush=True
                )

                self._update_ema()

                print(
                    f"RUN LOOP 32: EMA update finished",
                    flush=True
                )

                # -------------------------------------------------
                # Logging
                # -------------------------------------------------
                logger.logkv(
                    "loss",
                    avg_loss
                )

                logger.logkv(
                    "grad_norm",
                    (
                        grad_norm.item()
                        if grad_norm is not None
                        else 0.0
                    )
                )

                logger.logkv_mean(
                    "loss_mean",
                    avg_loss
                )

                logger.logkv_mean(
                    "grad_norm_mean",
                    (
                        grad_norm.item()
                        if grad_norm is not None
                        else 0.0
                    )
                )

                if step % 10 == 0:

                    print(
                        f"[step {step}] "
                        f"loss = {avg_loss:.6f}, "
                        f"grad_norm = "
                        f"{grad_norm.item() if grad_norm is not None else 0.0:.6f}",
                        flush=True
                    )

                    logger.logkv(
                        "step",
                        step
                    )

                if (
                    step
                    % self.log_interval
                    == 0
                ):

                    print(
                        f"RUN LOOP 33: Dumping logger "
                        f"values at step {step}",
                        flush=True
                    )

                    logger.dumpkvs()

                # -------------------------------------------------
                # Checkpoint
                # -------------------------------------------------
                if (
                    step
                    % self.save_interval
                    == 0
                ):

                    print(
                        f"RUN LOOP 34: Saving checkpoint "
                        f"at step {step}",
                        flush=True
                    )

                    ckpt_path = os.path.join(
                        self.outdir,
                        f"model{step:06d}.pt"
                    )

                    torch.save(
                        self.model.state_dict(),
                        ckpt_path
                    )

                    print(
                        f"RUN LOOP 35: Main model saved: "
                        f"{ckpt_path}",
                        flush=True
                    )

                    ema_path = os.path.join(
                        self.outdir,
                        f"ema_{step:06d}.pt"
                    )

                    torch.save(
                        self.ema_model.state_dict(),
                        ema_path
                    )

                    print(
                        f"RUN LOOP 36: EMA model saved: "
                        f"{ema_path}",
                        flush=True
                    )

                    print(
                        f"[step {step}] saved checkpoint",
                        flush=True
                    )

                step += 1

                print(
                    f"RUN LOOP 37: Completed training "
                    f"step {step}",
                    flush=True
                )

            if not infinite_loader:

                print(
                    f"RUN LOOP 38: Reached end of "
                    f"DataLoader epoch. Current step = {step}",
                    flush=True
                )

                continue

        # ---------------------------------------------------------
        # Final checkpoint
        # ---------------------------------------------------------
        print(
            "RUN LOOP 39: Training loop finished. "
            "Saving final checkpoint...",
            flush=True
        )

        final_ckpt = os.path.join(
            self.outdir,
            f"model{step:06d}.pt"
        )

        torch.save(
            self.model.state_dict(),
            final_ckpt
        )

        print(
            f"RUN LOOP 40: Final model saved: "
            f"{final_ckpt}",
            flush=True
        )

        ema_final = os.path.join(
            self.outdir,
            f"ema_{step:06d}.pt"
        )

        torch.save(
            self.ema_model.state_dict(),
            ema_final
        )

        print(
            f"RUN LOOP 41: Final EMA model saved: "
            f"{ema_final}",
            flush=True
        )

        print(
            "Training complete. Final checkpoint:",
            final_ckpt,
            flush=True
        )


def parse_resume_step_from_filename(
    filename
):

    try:

        return int(
            filename
            .split("model")[-1]
            .split(".")[0]
        )

    except (
        IndexError,
        ValueError
    ):

        return 0


def get_blob_logdir():

    return os.getenv(
        "DIFFUSION_BLOB_LOGDIR",
        logger.get_dir()
    )


def find_resume_checkpoint():

    logdir = get_blob_logdir()

    if not bf.exists(
        logdir
    ):
        return None

    ckpts = [
        f
        for f in bf.listdir(
            logdir
        )
        if (
            f.startswith("model")
            and f.endswith(".pt")
        )
    ]

    if not ckpts:
        return None

    ckpts.sort(
        key=lambda f:
        parse_resume_step_from_filename(
            f
        )
    )

    latest = ckpts[-1]

    return bf.join(
        logdir,
        latest
    )


def find_ema_checkpoint(
    main_checkpoint,
    step,
    rate
):

    if not main_checkpoint:
        return None

    path = bf.join(
        bf.dirname(
            main_checkpoint
        ),
        f"ema_{rate}_{step:06d}.pt"
    )

    return (
        path
        if bf.exists(path)
        else None
    )


def log_loss_dict(
    diffusion,
    ts,
    losses
):

    for key, values in (
        losses.items()
    ):

        logger.logkv_mean(
            key,
            values.mean().item()
        )

        for (
            t_idx,
            loss_val
        ) in zip(
            ts.cpu().numpy(),
            values.detach()
            .cpu()
            .numpy()
        ):

            quartile = int(
                4
                * t_idx
                / diffusion.num_timesteps
            )

            logger.logkv_mean(
                f"{key}_q{quartile}",
                loss_val
            )
