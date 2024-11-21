import numpy as np
import torch
from transformers import ViTImageProcessor, ViTConfig, ViTForImageClassification
from evaluate import load
from TinyImageNetLoader import TinyImageNetLoader  # Import the loader class

class ViTModelPreparation:
    def __init__(self, dataset_dict, model_name_or_path='google/vit-base-patch16-224-in21k'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_dict = dataset_dict
        self.model_name_or_path = model_name_or_path
        self.processor = ViTImageProcessor.from_pretrained(model_name_or_path)
        self.metric = load('accuracy')
        self.labels = dataset_dict['train'].features['labels'].names

        # Define configuration based on the dataset

        labels =self.dataset_dict['train'].features['labels'].names

        self.config = ViTConfig(
            num_labels=self.dataset_dict['train'].features['labels'].num_classes,
            image_size=224,
            patch_size=16,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=12,
            intermediate_size=3072
        )
        model = ViTForImageClassification.from_pretrained(
            model_name_or_path,
            num_labels=len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)}
        )
        #self.model = ViTForImageClassification(self.config).to(self.device)
        self.model = model.to(self.device)


    def compute_metrics(self, p):
        """Computes metrics for model evaluation."""
        return self.metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    def transform(self, example_batch):
        """Transforms a batch of examples for the ViT model."""
        images = [x.convert("RGB") for x in example_batch['image']]  # Convert to RGB if necessary
        inputs = self.processor(images, return_tensors='pt')
        inputs['labels'] = example_batch['labels']
        return inputs

    def collate_fn(self, batch):
        """Collates a batch for DataLoader compatibility."""
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x['labels'] for x in batch])
        }

    def prepare_dataset(self):
        """Applies the transform to the dataset."""
        return self.dataset_dict.with_transform(self.transform)
