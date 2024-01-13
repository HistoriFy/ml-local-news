##FINE-TUNING SEAMLESS MODEL CODE
## seamless_communication repo has to be cloned
##git clone https://github.com/facebookresearch/seamless_communication.git
import argparse
import logging
import os
import torch
from pathlib import Path
from fairseq2.models.nllb.tokenizer import NllbTokenizer
from seamless_communication.src.seamless_communication.cli.m4t.finetune import dataloader, dist_utils, trainer
from seamless_communication.src.seamless_communication.models.unity import (
    load_unity_model, load_unity_text_tokenizer, load_unity_unit_tokenizer)

def setup_logging():
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s %(levelname)s: %(message)s")
    logging.getLogger("finetune")

def parse_args():
    parser = argparse.ArgumentParser(description="Finetuning script for M4T models")
    parser.add_argument("--train_dataset", type=Path, required=True, help="Path to train samples")
    parser.add_argument("--eval_dataset", type=Path, required=True, help="Path to eval samples")
    parser.add_argument("--model_name", type=str, default="seamlessM4T_medium", help="Model name")
    parser.add_argument("--save_model_to", type=Path, required=True, help="Path to save model")
    parser.add_argument("--seed", type=int, default=2343, help="Seed value")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument("--patience", type=int, default=3, help="Early termination patience")
    parser.add_argument("--max_epochs", type=int, default=1, help="Max training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-7, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--eval_steps", type=int, default=50, help="Eval steps")
    parser.add_argument("--log_steps", type=int, default=10, help="Log steps")
    parser.add_argument("--mode", type=trainer.FinetuneMode, choices=list(trainer.FinetuneMode), 
                        default=trainer.FinetuneMode.SPEECH_TO_TEXT, help="Finetune mode")
    return parser.parse_args()

def main():
    setup_logging()
    args = parse_args()
    dist_utils.init_distributed()
    device = torch.device("cuda")
    model = load_unity_model(args.model_name, device=device, dtype=torch.float16)

    train_dataloader = dataloader.UnitYDataLoader(load_unity_text_tokenizer(args.model_name),
                                                  load_unity_unit_tokenizer(args.model_name),
                                                  dataloader.BatchingConfig(args.batch_size),
                                                  args.train_dataset)
    eval_dataloader = dataloader.UnitYDataLoader(load_unity_text_tokenizer(args.model_name),
                                                 load_unity_unit_tokenizer(args.model_name),
                                                 dataloader.BatchingConfig(args.batch_size),
                                                 args.eval_dataset)
    finetune_params = trainer.FinetuneParams(args.mode, args.save_model_to, device, args.batch_size,
                                             args.patience, args.max_epochs, args.learning_rate,
                                             args.warmup_steps, args.eval_steps, args.log_steps)
    trainer.UnitYFinetune(model, finetune_params, train_dataloader, eval_dataloader).run()

if __name__ == "__main__":
    main()
