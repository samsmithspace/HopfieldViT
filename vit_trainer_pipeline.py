from datasets import load_dataset
from transformers import ViTForImageClassification
from vit_trainer_setup import ViTTrainerSetup
from vit_model_preparation import ViTModelPreparation
from TinyImageNetLoader import TinyImageNetLoader
from attention_logger import AttentionLoggerCallback
import torch
import sys
import tracemalloc

tracemalloc.start()
import os

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
class ViTTrainerPipeline:
    def __init__(self, config_path='./dataset_config.json', model_name_or_path='google/vit-base-patch16-224-in21k',
                 output_dir="./vit-output", log_dir="./logs", max_steps=200):
        self.config_path = config_path
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.max_steps = max_steps
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize components
        self.dataset_loader = None
        self.dataset_dict = None
        self.model_prep = None
        self.model = None
        self.processor = None
        self.prepared_ds = None
        self.trainer_setup = None
        self.callback = None

    def load_data(self):
        """Loads the dataset using TinyImageNetLoader or another dataset."""

        #self.dataset_dict=load_dataset("dpdl-benchmark/oxford_flowers102")
        #self.dataset_dict = load_dataset("uoft-cs/cifar10")
        #self.dataset_dict =load_dataset("uoft-cs/cifar100")
        #ds = load_dataset("timm/oxford-iiit-pet")
        self.dataset_dict = load_dataset('beans')
        #self.dataset_dict = load_dataset("keremberke/indoor-scene-classification", "full")
        #self.dataset_dict =load_dataset("timm/oxford-iiit-pet")
        #print(self.dataset_dict)  # This should show the train, validation, and test splits
        #self.dataset_loader = TinyImageNetLoader(config_path = self.config_path)
        #self.dataset_dict = self.dataset_loader.create_datasets()
        # Inspect the first few examples
        #print(self.dataset_dict['train'][0])  # Check the first example in the train set
        #print(self.dataset_dict['test'][0])  # Check the first example in the test set
        print("Dataset loaded successfully.")


    def prepare_model_and_data(self):
        """Prepares the model and applies transformations to the dataset."""
        # Initialize the ViT model preparation
        self.model_prep = ViTModelPreparation(self.dataset_dict, self.model_name_or_path)
        self.model = self.model_prep.model

        self.processor = self.model_prep.processor

        # Apply dataset transformation for training
        self.prepared_ds = self.model_prep.prepare_dataset()
        print("Model and data preparation complete.")
        #self.check_memory_usage()

    def setup_trainer(self):
        """Configures the trainer with arguments, collate function, and metrics."""
        self.trainer_setup = ViTTrainerSetup(self.model, self.processor, self.prepared_ds, self.output_dir)

        # Configure training arguments
        self.trainer_setup.setup_training_arguments()

        # Use the collate function defined in the ViTModelPreparation or define it here
        # Using the updated collate function that handles the dataset correctly
        collate_fn = lambda batch: {

            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),

            'labels': torch.tensor([x['labels'] for x in batch])

        }

        compute_metrics = self.model_prep.compute_metrics  # Ensure this points to the correct function

        # Set up the Trainer
        self.trainer_setup.setup_trainer(collate_fn=self.model_prep.collate_fn, compute_metrics=compute_metrics)
        print("Trainer setup complete.")


        #self.check_memory_usage()

    def add_callbacks(self):
        """Adds custom callbacks to the Trainer, such as AttentionLoggerCallback."""
        callback = AttentionLoggerCallback(self.model, self.prepared_ds, log_dir=self.log_dir,
                                                max_steps=self.max_steps)
        self.trainer_setup.add_callback(callback)
        print("Custom callback added.")


    def train_model(self):
        """Trains the model and saves it along with metrics and state."""
        self.trainer_setup.train_and_save()
        print("Training completed and model saved.")
        #self.check_memory_usage()
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        #print("Top memory usage:")
        #for stat in top_stats[:10]:  # Show top 10 memory consumers
        #    print(stat)

    def run_pipeline(self):
        """Executes the entire training pipeline from data loading to training."""
        self.load_data()
        self.prepare_model_and_data()

        self.setup_trainer()
        self.add_callbacks()
        self.trainer_setup.start_training()
        #self.train_model()
        self.trainer_setup.save()
        print("Pipeline run complete.")


# Example Usage
if __name__ == "__main__":
    print(torch.cuda.is_available())

    # Instantiate and run the training pipeline
    pipeline = ViTTrainerPipeline(config_path='./dataset_config.json',
                                  model_name_or_path='google/vit-base-patch16-224-in21k')
    pipeline.run_pipeline()
