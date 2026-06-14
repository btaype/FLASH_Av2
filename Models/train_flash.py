import argparse
import csv
import os
import time


MODEL_CHOICES = {
    "0": ("Qwen 2.5 1.5B recomendado para 8GB", "Qwen/Qwen2.5-1.5B"),
    "1": ("Qwen 2.5 7B", "Qwen/Qwen2.5-7B"),
    "2": ("TinyLlama 1.1B Chat", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
}

FLASH_CHOICES = {
    "0": ("sin FlashAttention: eager", "eager"),
    "1": ("con FlashAttention 2 oficial", "flash_attention_2"),
    "2": ("sin paquete flash-attn: PyTorch SDPA", "sdpa"),
}


def select_model(args) -> str:
    if args.model_id:
        return args.model_id

    choice = args.model_choice
    if choice is None:
        print("Selecciona el modelo para entrenar:")
        for key, (name, model_id) in MODEL_CHOICES.items():
            print(f"{key}: {name} ({model_id})")
        choice = input("Modelo 0, 1 o 2: ").strip()

    if choice not in MODEL_CHOICES:
        valid = ", ".join(MODEL_CHOICES)
        raise ValueError(f"Modelo invalido: {choice}. Usa uno de: {valid}")

    name, model_id = MODEL_CHOICES[choice]
    print(f"Entrenando un modelo: {name} -> {model_id}")
    return model_id


def select_flash_attention(args) -> str:
    choice = args.flash_choice
    if choice is None:
        print("Selecciona atencion:")
        for key, (name, _) in FLASH_CHOICES.items():
            print(f"{key}: {name}")
        choice = input("FlashAttention 0 o 1: ").strip()

    if choice not in FLASH_CHOICES:
        valid = ", ".join(FLASH_CHOICES)
        raise ValueError(f"Opcion de FlashAttention invalida: {choice}. Usa uno de: {valid}")

    name, enabled = FLASH_CHOICES[choice]
    print(f"Atencion seleccionada: {name}")
    return enabled


def default_output_dir(model_id: str) -> str:
    model_name = model_id.rstrip("/").split("/")[-1].replace(".", "-")
    return f"outputs/{model_name}-openwebtext-flash"


def resolve_max_steps(args) -> int:
    if not args.streaming:
        return -1
    if args.max_steps is not None:
        return args.max_steps
    return max(1, int(args.epochs * args.steps_per_epoch))


def print_training_perf(model, metrics, actual_tokens: int, world_size: int) -> None:
    runtime = metrics.get("train_runtime")
    if not runtime or runtime <= 0 or actual_tokens <= 0:
        print("No se pudo calcular TFLOPS: falta runtime o tokens reales.")
        return

    params = model.num_parameters()
    flops = 6 * params * actual_tokens
    tokens_per_second = actual_tokens / runtime
    model_tflops_total = flops / runtime / 1e12
    model_tflops_per_gpu = model_tflops_total / max(1, world_size)

    print("Resultados de rendimiento aproximados:")
    print("Metodo TFLOPS: model FLOPs ~= 6 * parametros * tokens reales / runtime")
    print(f"Parametros: {params:,}")
    print(f"Tokens reales contados: {actual_tokens:,}")
    print(f"Runtime: {runtime:.2f} s")
    print(f"Tokens/s reales: {tokens_per_second:,.2f}")
    print(f"Model TFLOPS/s total: {model_tflops_total:,.2f}")
    print(f"Model TFLOPS/s por GPU: {model_tflops_per_gpu:,.2f}")


def load_training_model(model_id: str, args, attention_backend: str):
    import torch
    from common_flash import load_flash_model, print_attention_config

    if args.train_mode == "qlora":
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        tokenizer, model = load_flash_model(
            model_id,
            attention_backend=attention_backend,
            device_map={"": 0},
            quantization_config=quant_config,
        )
        print_attention_config(model)
        return tokenizer, model

    tokenizer, model = load_flash_model(model_id, attention_backend=attention_backend, device_map=None)
    print_attention_config(model)
    return tokenizer, model


def apply_lora_if_needed(model, args):
    if args.train_mode == "full":
        return model

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if args.train_mode == "qlora":
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_text_dataset(args):
    from datasets import load_dataset

    if args.dataset_path:
        return load_dataset("json", data_files=args.dataset_path, split="train", streaming=args.streaming)
    return load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.dataset_split,
        streaming=args.streaming,
    )


def tokenize_dataset(tokenizer, args):
    dataset = load_text_dataset(args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.streaming and args.shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    def tokenize(batch):
        if not args.pack:
            return tokenizer(
                batch[args.text_column],
                truncation=True,
                max_length=args.max_length,
                padding=False,
            )

        all_tokens = []
        eos_token_id = tokenizer.eos_token_id
        for text in batch[args.text_column]:
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            all_tokens.extend(token_ids)
            if eos_token_id is not None:
                all_tokens.append(eos_token_id)

        block_size = args.max_length
        chunks = [
            all_tokens[i : i + block_size]
            for i in range(0, len(all_tokens) - block_size + 1, block_size)
        ]
        return {
            "input_ids": chunks,
            "attention_mask": [[1] * len(chunk) for chunk in chunks],
        }

    return dataset.map(tokenize, batched=True, remove_columns=[args.text_column])


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal causal LM training with FlashAttention 2.")
    parser.add_argument("--model-id", help="Hugging Face model id or local path. Overrides --model-choice.")
    parser.add_argument("--model-choice", choices=sorted(MODEL_CHOICES), help="0 = Qwen 1.5B, 1 = Qwen 7B, 2 = TinyLlama 1.1B Chat.")
    parser.add_argument("--flash-choice", choices=sorted(FLASH_CHOICES), help="0 = eager, 1 = FlashAttention 2 oficial, 2 = PyTorch SDPA.")
    parser.add_argument("--train-mode", choices=["full", "lora", "qlora"], default="qlora", help="Use qlora for 8GB GPUs.")
    parser.add_argument("--dataset-path", help="Optional local JSONL file with a text field.")
    parser.add_argument("--dataset-name", default="Skylion007/openwebtext", help="Hugging Face dataset name.")
    parser.add_argument("--dataset-config", default="plain_text", help="Hugging Face dataset config/subset.")
    parser.add_argument("--dataset-split", default="train", help="Dataset split to train on.")
    parser.add_argument("--text-column", default="text", help="Column containing raw text.")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pack", action=argparse.BooleanOptionalAction, default=True, help="Pack texts into full context blocks.")
    parser.add_argument("--output-dir", help="Directory for checkpoints. Defaults to one folder per model.")
    # Contexto del modelo: por defecto 2048 tokens, igual a 2k.
    parser.add_argument("--max-length", type=int, default=2048)
    # Ajuste principal de batch: para 7B normalmente empieza con 1 y compensa con grad-accum.
    parser.add_argument("--batch-size", type=int, default=1)
    # Batch efectivo = batch-size * grad-accum * numero de GPUs.
    parser.add_argument("--grad-accum", type=int, default=16)
    # Bueno para continued pretraining suave; baja a 1e-5 si el loss se vuelve inestable.
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    # Epocas logicas para streaming: epochs * steps-per-epoch = pasos totales.
    parser.add_argument("--epochs", type=float, default=1.0)
    # Tamano de una epoca logica cuando se entrena por web/streaming.
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    # Si lo defines, manda sobre epochs * steps-per-epoch en modo streaming.
    parser.add_argument("--max-steps", type=int)
    # Mezcla ejemplos en modo streaming sin descargar todo el dataset.
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--progress-steps", type=int, default=10, help="Print detailed progress every N optimizer steps.")
    args = parser.parse_args()

    import torch
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainerCallback, TrainingArguments

    model_id = select_model(args)
    attention_backend = select_flash_attention(args)
    if args.output_dir is None:
        args.output_dir = default_output_dir(model_id)

    tokenizer, model = load_training_model(model_id, args, attention_backend)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = apply_lora_if_needed(model, args)
    tokenized = tokenize_dataset(tokenizer, args)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    max_steps = resolve_max_steps(args)
    if args.streaming:
        print(f"Epocas: {args.epochs} | pasos por epoca: {args.steps_per_epoch} | max_steps: {max_steps}")
    print("Configuracion de entrenamiento:")
    print(f"Modelo: {model_id}")
    print(f"Modo: {args.train_mode}")
    print(f"Backend de atencion: {attention_backend}")
    print(f"Contexto: {args.max_length} tokens")
    print(f"Batch efectivo: {args.batch_size * args.grad_accum * max(1, torch.cuda.device_count())}")
    print(f"Salida: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    progress_csv_path = os.path.join(args.output_dir, "progress.csv")
    print(f"CSV progreso: {progress_csv_path}")

    class CountingTrainer(Trainer):
        def __init__(self, *trainer_args, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
            self.actual_tokens = 0

        def training_step(self, model, inputs, *step_args, **step_kwargs):
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                self.actual_tokens += int(attention_mask.sum().item())
            else:
                self.actual_tokens += int(inputs["input_ids"].numel())
            return super().training_step(model, inputs, *step_args, **step_kwargs)

    class ProgressCallback(TrainerCallback):
        def __init__(self):
            self.start_time = None
            self.last_loss = None
            self.csv_file = None
            self.csv_writer = None

        def on_train_begin(self, args_, state, control, **kwargs):
            self.start_time = time.time()
            self.csv_file = open(progress_csv_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.DictWriter(
                self.csv_file,
                fieldnames=[
                    "step",
                    "max_steps",
                    "percent",
                    "logical_epoch",
                    "loss",
                    "learning_rate",
                    "tokens",
                    "tokens_per_second",
                    "model_tflops_per_gpu",
                    "elapsed_seconds",
                ],
            )
            self.csv_writer.writeheader()
            self.csv_file.flush()

        def on_log(self, args_, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                self.last_loss = logs["loss"]

        def on_step_end(self, args_, state, control, **kwargs):
            if state.global_step == 0 or state.global_step % args.progress_steps != 0:
                return
            elapsed = max(1e-9, time.time() - self.start_time)
            tokens = trainer.actual_tokens
            tokens_per_second = tokens / elapsed
            params = model.num_parameters()
            model_tflops = (6 * params * tokens / elapsed / 1e12) / max(1, args_.world_size)
            percent = 100 * state.global_step / max(1, state.max_steps)
            logical_epoch = state.global_step / max(1, args.steps_per_epoch)
            loss_text = f"{self.last_loss:.4f}" if self.last_loss is not None else "n/a"
            lr = state.log_history[-1].get("learning_rate") if state.log_history else None
            lr_text = f"{lr:.2e}" if lr is not None else "n/a"
            self.csv_writer.writerow(
                {
                    "step": state.global_step,
                    "max_steps": state.max_steps,
                    "percent": round(percent, 4),
                    "logical_epoch": round(logical_epoch, 6),
                    "loss": self.last_loss if self.last_loss is not None else "",
                    "learning_rate": lr if lr is not None else "",
                    "tokens": tokens,
                    "tokens_per_second": round(tokens_per_second, 4),
                    "model_tflops_per_gpu": round(model_tflops, 6),
                    "elapsed_seconds": round(elapsed, 4),
                }
            )
            self.csv_file.flush()
            print(
                f"[progreso] paso {state.global_step}/{state.max_steps} ({percent:.1f}%) | "
                f"epoca logica {logical_epoch:.2f}/{args.epochs} | loss {loss_text} | lr {lr_text} | "
                f"tokens {tokens:,} | tokens/s {tokens_per_second:,.1f} | "
                f"model TFLOPS/s GPU {model_tflops:,.2f}"
            )

        def on_train_end(self, args_, state, control, **kwargs):
            if self.csv_file is not None:
                self.csv_file.close()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_steps=max_steps,
        num_train_epochs=args.epochs,
        warmup_ratio=0.03,
        weight_decay=0.01,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if args.train_mode == "qlora" else "adamw_torch",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )

    trainer = CountingTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[ProgressCallback()],
    )
    train_result = trainer.train()
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    print_training_perf(model, train_result.metrics, trainer.actual_tokens, training_args.world_size)


if __name__ == "__main__":
    main()
