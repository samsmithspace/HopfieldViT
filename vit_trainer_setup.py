from transformers import TrainingArguments, Trainer
from vit_attention_analyzer import ViTAttentionAnalyzer  # Assuming this exists for attention analysis
from attention_logger import AttentionLoggerCallback  # Assuming custom callback for logging
import os
import torch

class ViTTrainerSetup:
    def __init__(self, model, processor, prepared_ds, output_dir="./vit-output", test_fraction=0.2):
        self.model = model
        self.processor = processor
        self.prepared_ds = prepared_ds
        self.output_dir = output_dir
        self.test_fraction = test_fraction  # Fraction of the test set to use
        self.training_args = None
        self.trainer = None

    def setup_training_arguments(self):
        """Configures training arguments with checkpoint loading support."""
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=16,
            #evaluation_strategy="steps",
            evaluation_strategy="no",
            num_train_epochs=4,
            save_steps=20,
            #eval_steps=20,
            logging_steps=10,
            learning_rate=2e-4,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to='tensorboard',
            #load_best_model_at_end=True,
            resume_from_checkpoint=True,  # Allows for resuming training
        )

    def debug_label_range_in_batches(self, train_dataloader):
        """Checks that label values in batches are within the correct range."""
        print("Checking label range in training batches...")
        for batch in train_dataloader:
            labels = batch["labels"]
            min_label, max_label = labels.min().item(), labels.max().item()
            print(f"Batch label range: {min_label} - {max_label}")
            assert min_label >= 0 and max_label < self.model.config.num_labels, \
                "Labels out of range!"
            break  # Run on one batch initially; remove or adjust for more batches

    def setup_trainer(self, collate_fn, compute_metrics):
        """
        Sets up the Trainer with model, arguments, data, and collator.

        Args:
            collate_fn (function): Collate function for data loading.
            compute_metrics (function): Metric computation function.
        """
        if not self.training_args:
            raise ValueError("Training arguments are not set up. Call `setup_training_arguments()` first.")

        # Use the latest checkpoint if available
        checkpoint = self.get_last_checkpoint()

        if checkpoint:
            print(f"Resuming training from checkpoint: {checkpoint}")
        else:
            print("Starting training from scratch.")

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            train_dataset=self.prepared_ds["train"],
            eval_dataset=self.prepared_ds["test"],  # Use the sliced eval dataset
            tokenizer=self.processor,
        )

        # Check label ranges in training batches before actual training
        self.debug_label_range_in_batches(self.trainer.get_train_dataloader())

    def start_training(self):
        """Starts training the model, resuming from checkpoint if available."""
        if not self.trainer:
            raise ValueError("Trainer is not set up. Call `setup_trainer()` first.")

        # Retrieve the last checkpoint, if available
        checkpoint = self.get_last_checkpoint()

        # Start training
        if checkpoint:
            print(f"Training will resume from checkpoint: {checkpoint}")
            self.trainer.train(resume_from_checkpoint=checkpoint)
        else:
            print("Training will start from scratch.")
            self.trainer.train()

    def get_last_checkpoint(self):
        """Retrieves the last checkpoint directory if it exists."""
        checkpoints = [f for f in os.listdir(self.output_dir) if
                       os.path.isdir(os.path.join(self.output_dir, f)) and 'checkpoint-' in f]
        if checkpoints:
            # Sort the checkpoints and return the most recent one
            checkpoints.sort(key=lambda x: int(x.split('-')[-1]), reverse=True)
            return os.path.join(self.output_dir, checkpoints[0])
        return None

    def add_callback(self, callback):
        """Adds a custom callback to the Trainer."""
        if not self.trainer:
            raise ValueError("Trainer is not set up. Call `setup_trainer()` first.")
        self.trainer.add_callback(callback)

    def save(self):
        """Saves the model, metrics, and state after training."""
        if not self.trainer:
            raise ValueError("Trainer is not set up. Call `setup_trainer()` first.")

        # Save model and results
        self.trainer.save_model()
        print(f"Model saved to {self.output_dir}.")

        # Optional: save the current state and any pre-existing metrics
        self.trainer.save_state()
        print(f"Training complete. Model and metrics saved to {self.output_dir}.")
