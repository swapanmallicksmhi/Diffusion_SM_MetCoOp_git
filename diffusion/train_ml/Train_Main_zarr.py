#!/usr/bin/env python3
"""
===========================================================
 Program Information
===========================================================
 Title       : Diffusion Model Training Script
 Description : 
     Main driver program for training diffusion models on 
     zarr datasets with Uncertainty Quantification (UQ) 
     over the MetCoOp (subdomain of CERRA) domain. The script orchestrates the 
     model creation, data loading, and iterative training 
     process, while supporting distributed training setups 
     (via torchrun) and logging of outputs.

 Author      : Swapan Mallick
 Date        : 22 February 2026
 Date        : v2 24 February 2026 for zarr reader
 Framework   : PyTorch
-----------------------------------------------------------
 Key Features:
-----------------------------------------------------------
 1. Argument Parsing
    - Dynamically builds command-line arguments based on 
      model/diffusion defaults and user-specified parameters.
    - Supports flexible overrides for device, learning rate, 
      steps, microbatch size, etc.

 2. Device Setup
    - Automatically configures GPU device(s) if CUDA is 
      available.
    - Supports torchrun-based distributed training via 
      LOCAL_RANK environment variable.

 3. Model & Diffusion Process
    - Uses factory method to create both the model and 
      associated diffusion process with appropriate defaults.
    - Supports Exponential Moving Average (EMA) rate for 
      stabilized training (commented in training loop).

 4. Data Loading
    - Loads zarr datasets with configurable directory, 
      image size, and batch size.
    - Provides backward compatibility for legacy dataset 
      loading signatures.

 5. Training Loop
    - Implements training using `TrainLoop`, which manages:
        * Forward/backward passes
        * Optimizer steps
        * Periodic checkpointing
        * Logging and monitoring
    - Configurable learning rate, step count, save intervals, 
      and mixed precision (fp16) options.

-----------------------------------------------------------
 Command-Line Arguments (examples):
-----------------------------------------------------------
  --data_dir        : Path to dataset directory
  --TRAIN_OUT       : Output directory for logs and checkpoints
  --image_size      : Resolution of training images (default: 64)
  --batch_size      : Batch size for training (default: 1)
  --lr              : Learning rate (default: 1e-4)
  --steps           : Total training steps (default: 20000)
  --save_interval   : Steps between saving checkpoints
  --use_fp16        : Enable mixed precision training
  --device          : Explicit device string (e.g., 'cuda:0')
  --local_rank      : Rank for distributed training (torchrun)

===========================================================
"""

import argparse
import sys
import torch as th
import os
import importlib
import traceback

from src_diffusion.diffusion_dist import create_model_and_diffusion, model_and_diffusion_defaults
from src_diffusion.zarr_load import load_data
from src_diffusion.diffusion_train import TrainLoop
from src_diffusion import logger
from src_diffusion.resample import create_named_schedule_sampler # SWAPAN new line for sampler

# small helpers to add dict defaults into argparse and convert args to dict
def infer_type(value):
    if isinstance(value, bool):
        return lambda s: s.lower() in ("true", "1", "yes")
    if isinstance(value, int):
        return int
    if isinstance(value, float):
        return float
    return str

def add_dict_to_argparser(parser: argparse.ArgumentParser, defaults: dict):
    for k, v in defaults.items():
        argname = f"--{k}"
        t = infer_type(v)
        parser.add_argument(argname, default=v, type=t, help=f"(default: {v})")


def args_to_dict(args: argparse.Namespace, keys):
    return {k: getattr(args, k) for k in keys}


def create_argparser():
    # basic defaults used by the training driver (you can extend these)
    defaults = dict(
        data_dir="",
        TRAIN_OUT="./outputs",
        image_size=64,
        steps=20000,
        schedule_sampler="uniform", # Sampling schedule for time steps
        lr=1e-4,                    # Learning rate
        weight_decay=0.0,           # Optional L2 regularization
        lr_anneal_steps=0,          # Steps for learning rate annealing (0 disables)
        batch_size="",              # Number of samples per batch
        microbatch="",              # Microbatching (-1 disables)
        ema_rate="0.9999",          # EMA decay rate for model parameters
        log_interval=100,           # Logging interval (in training steps)
        save_interval=1000,         # How often to save model checkpoints
        resume_checkpoint="",       # Path to resume training from a checkpoint
        use_fp16=False,             # Use mixed precision training (for speed and memory)
        fp16_scale_growth=1e-3,     # FP16 scaling factor growth
        in_channels=3,              # Add this explicitly
        cond_dropout_prob=0.1,      #new_add
        cfg_guidance_scale=1.0, 
    )

    # Add model and diffusion-specific defaults
    defaults.update(model_and_diffusion_defaults())

    parser = argparse.ArgumentParser(description="Train diffusion model (+24h forecast)")
    add_dict_to_argparser(parser, defaults)

    #new_add
    parser.add_argument(
        "--input_vars",
        nargs="+",
        default=["t2m_era5_mean", "t2m_era5_std"],
        help="Physical conditioning variables loaded from zarr",
    )
    parser.add_argument(
        "--target_vars",
        nargs="+",
        default=["t2m_cerra_mean", "t2m_cerra_std"],
        help="Target variables loaded from zarr",
    )
    parser.add_argument(
       "--add_seasonal_cond",
       action="store_true",
       help="Append seasonal sin/cos channels to conditioning input",
    )
    parser.add_argument(
        "--seasonal_channels",
        type=int,
        default=0,
        help="Number of seasonal channels to append (e.g. 12)",
    )

    parser.add_argument(
        "--forecast_lead_hours",
        type=int,
        default=0,
        help="Forecast lead time in hours for target pairing (e.g. 24)",
    )

    # add any extra args that we expect (explicitly)
    parser.add_argument("--device", type=str, default=None, help="device to use (auto if not set)")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)),
                        help="local rank for distributed training (set by torchrun)")
    return parser


def setup_device(args):
    # if torchrun is used, LOCAL_RANK will be set; set device accordingly.
    if args.device:
        device = th.device(args.device)
    else:
        # prefer CUDA if available
        if th.cuda.is_available():
            # set cuda device according to local_rank (works with torchrun)
            local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank or 0))
            try:
                th.cuda.set_device(local_rank)
                device = th.device(f"cuda:{local_rank}")
            except Exception:
                device = th.device("cuda")
        else:
            device = th.device("cpu")
    return device


def debug_model_and_data(model, loader, device):
    """Debug function to test model and data compatibility"""
    print("\n" + "=" * 80)
    print("DEBUGGING MODEL AND DATA")
    print("=" * 80)
    
    # Test data loader
    print("\n1. Testing Data Loader:")
    try:
        sample_batch = next(iter(loader))
        print(f"   Number of items in batch: {len(sample_batch)}")
        if len(sample_batch) == 2:
            images, cond = sample_batch
            print(f"   Images shape: {images.shape}")
            print(f"   Condition shape: {cond.shape}")
            print(f"   Images dtype: {images.dtype}")
            print(f"   Condition dtype: {cond.dtype}")
        else:
            print(f"   Unexpected batch format: {sample_batch}")
            return False
    except Exception as e:
        print(f"   Error in data loader: {e}")
        traceback.print_exc()
        return False

    # Test model
    print("\n2. Testing Model:")
    print(f"   Model type: {type(model)}")
    print(f"   Model in_channels: {model.in_channels}")
    print(f"   Model out_channels: {model.out_channels}")
    if hasattr(model, 'cond_channels'):
        print(f"   Model cond_channels: {model.cond_channels}")
    print(f"   Model device: {next(model.parameters()).device}")

    # Test forward pass
    print("\n3. Testing Forward Pass:")
    try:
        with th.no_grad():
            model.eval()
            # Move data to device
            images = images.to(device)
            cond = cond.to(device)
            
            # Create random timesteps
            t = th.randint(0, 1000, (images.shape[0],)).to(device)
            
            print(f"   Input shapes - x: {images.shape}, t: {t.shape}, cond: {cond.shape}")
            
            # Try forward pass with cond
            output = model(images, t, cond=cond)
            print(f"   Output shape: {output.shape}")
            print("    Forward pass successful!")
            
            # Try without cond (should also work if cond is optional)
            try:
                output2 = model(images, t)
                print(f"   Output without cond shape: {output2.shape}")
            except Exception as e:
                print(f"   Note: Forward without cond failed (expected if cond is required): {e}")
            
            model.train()
    except Exception as e:
        print(f"   Error in forward pass: {e}")
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("DEBUGGING COMPLETE - ALL CHECKS PASSED")
    print("=" * 80)
    return True


def main():
    parser = create_argparser()
    args = parser.parse_args()

    # Create output dir
    os.makedirs(args.TRAIN_OUT, exist_ok=True)

    # Setup device (handles torchrun local_rank)
    device = setup_device(args)
    print(f"Using device: {device}")

    # Create model and diffusion instances using the factory
    md_defaults = model_and_diffusion_defaults()
    model_diff_kwargs = args_to_dict(args, md_defaults.keys())

    #new_add: modify input_vars in job only
    auto_in_channels = len(args.target_vars)
    auto_cond_channels = len(args.input_vars) + (
        args.seasonal_channels if args.add_seasonal_cond else 0
    )
    model_diff_kwargs["in_channels"] = auto_in_channels
    model_diff_kwargs["cond_channels"] = auto_cond_channels
    
    print(f"Auto in_channels = {auto_in_channels}")
    print(f"Auto cond_channels = {auto_cond_channels}")
    print(f"Physical cond vars = {args.input_vars}")
    print(f"Seasonal channels = {args.seasonal_channels if args.add_seasonal_cond else 0}")
    print(f"Forecast lead hours = {args.forecast_lead_hours}")
    # Ensure in_channels is set correctly
    print(f"Model kwargs: {model_diff_kwargs}")
    print(f"Selected diffusion_type: {model_diff_kwargs.get('diffusion_type', 'ddpm')}") 
    # IMPORTANT: model_and_diffusion_defaults() should expose 'diffusion_type' and 'loss_type'
    # so they become available in model_diff_kwargs if provided via CLI.
    model, diffusion = create_model_and_diffusion(**model_diff_kwargs)
    logger.configure(dir=args.TRAIN_OUT)
    logger.log("Creating model and diffusion process...")
    logger.log(f"Model device: {device}")

    # Move model to device
    model = model.to(device)
    print(f"Model moved to {device}")

    # Create the schedule sampler for selecting timesteps
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # Prepare dataset / loader
    logger.log("Loading dataset...")
    # load_data signature expected: data_dir, batch_size, image_size (older/new versions may accept other kwargs)
    try:
        loader = load_data(
            zarr_path=args.data_dir,
            batch_size=int(args.batch_size),
            image_size=int(args.image_size),
            input_vars=args.input_vars, #new_add
            target_vars=args.target_vars,
            add_seasonal_cond=args.add_seasonal_cond,
            seasonal_channels=args.seasonal_channels,
            forecast_lead_hours=args.forecast_lead_hours,
        )
        #new_add: check channels
        sample_target, sample_cond = next(iter(loader))
        if sample_target.shape[1] != auto_in_channels:
            raise ValueError(
                f"Target channel mismatch: expected {auto_in_channels}, got {sample_target.shape[1]}"
            )
        if sample_cond.shape[1] != auto_cond_channels:
            raise ValueError(
                f"Condition channel mismatch: expected {auto_cond_channels}, got {sample_cond.shape[1]}"
            )
    except Exception as e:
        print("ERROR while calling load_data():", e)
        traceback.print_exc()
        raise

    # Debug model and data compatibility
    if not debug_model_and_data(model, loader, device):
        print("Debugging failed. Please fix the issues before training.")
        return

    # Start training
    logger.log("Starting training loop...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=loader,
        batch_size=args.batch_size,
        microbatch=int(args.microbatch),
        lr=float(args.lr),
        steps=int(args.steps),
        device=device,
        outdir=args.TRAIN_OUT,
        save_interval=args.save_interval,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        cond_dropout_prob=args.cond_dropout_prob, #new_add
    ).run_loop()

if __name__ == "__main__":
    main()
