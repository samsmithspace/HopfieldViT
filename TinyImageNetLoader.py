import os
import json
import torch
from datasets import Dataset, DatasetDict, Features, ClassLabel, Image
from PIL import Image as PILImage, UnidentifiedImageError


class TinyImageNetLoader:
    def __init__(self, config_path, num_classes=179):
        self.train_dir, self.val_dir, self.test_dir, self.val_annotations_path = self._load_dataset_config(config_path)
        self.label_map = {}
        self.reverse_label_map = {}
        self.num_classes = num_classes  # Set number of classes

    def _load_dataset_config(self, config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config.get('train_dir'), config.get('val_dir'), config.get('test_dir'), config.get(
            'val_annotations_path')

    def load_train_images_labels(self):
        images = []
        labels = []
        label_names = sorted(os.listdir(self.train_dir))
        self.label_map = {name: i for i, name in enumerate(label_names)}

        for label_name in label_names:
            class_dir = os.path.join(self.train_dir, label_name, 'images')
            if not os.path.isdir(class_dir):
                continue
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                if os.path.isfile(img_path):
                    try:
                        images.append(PILImage.open(img_path).convert("RGB"))
                        labels.append(self.label_map[label_name])
                    except (UnidentifiedImageError, OSError) as e:
                        print(f"Skipping error image: {img_path} - {e}")

        if len(images) != len(labels):
            raise ValueError("Mismatch between number of images and labels in training data.")

        print("Train labels range:", min(labels), "-", max(labels))
        return images, labels

    def load_val_images_labels(self):
        images = []
        labels = []
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

        try:
            with open(self.val_annotations_path, 'r') as f:
                annotations = {}
                for line in f:
                    parts = line.strip().split('\t')
                    img_file, class_label = parts[0], parts[1]

                    if class_label not in self.label_map:
                        self.label_map[class_label] = len(self.label_map)
                        self.reverse_label_map[self.label_map[class_label]] = class_label

                    annotations[img_file] = self.label_map[class_label]
        except FileNotFoundError:
            print(f"Annotations file not found: {self.val_annotations_path}")
            return images, labels

        for subdir in os.listdir(self.val_dir):
            path = os.path.join(self.val_dir, subdir)
            if not os.path.isdir(path):
                continue

            for img_file in os.listdir(path):
                img_path = os.path.join(path, img_file)
                if os.path.isfile(img_path):
                    try:
                        label = annotations[img_file]
                        if label < self.num_classes:  # Only include labels within the valid range
                            images.append(PILImage.open(img_path).convert("RGB"))
                            labels.append(label)
                        else:
                            print(f"Skipping out-of-range label {label} for {img_file}")
                    except (UnidentifiedImageError, OSError, KeyError) as e:
                        print(f"Skipping error image: {img_path} - {e}")

        if len(images) != len(labels):
            raise ValueError("Mismatch between number of images and labels in validation data.")

        print("Validation labels range:", min(labels), "-", max(labels))
        return images, labels

    def load_test_images(self):
        images = []
        labels = []
        for files in os.listdir(self.test_dir):
            path = os.path.join(self.test_dir, files)

            for img_file in os.listdir(path):
                img_path = os.path.join(path, img_file)
                if os.path.isfile(img_path):
                    try:
                        images.append(PILImage.open(img_path).convert("RGB"))
                        labels.append(-1)  # Dummy label for test set
                    except (UnidentifiedImageError, OSError) as e:
                        print(f"Skipping error image: {img_path} - {e}")

        if len(images) != len(labels):
            raise ValueError("Mismatch between number of images and labels in test data.")

        print("Test dataset loaded successfully with", len(images), "images.")
        return images, labels

    def inspect_labels(self, labels, dataset_name):
        """Check if all labels are within the valid range and print statistics."""
        min_label, max_label = min(labels), max(labels)
        print(f"{dataset_name} labels range: {min_label} to {max_label}")
        if min_label < 0 or max_label >= self.num_classes:
            print(f"Warning: {dataset_name} labels out of range!")
        else:
            print(f"All {dataset_name} labels are within the valid range.")

    def create_datasets(self):
        train_images, train_labels = self.load_train_images_labels()
        val_images, val_labels = self.load_val_images_labels()
        test_images, test_labels = self.load_test_images()

        # Inspect labels to verify range
        self.inspect_labels(train_labels, "Train")
        self.inspect_labels(val_labels, "Validation")

        features = Features({
            'image': Image(),
            'labels': ClassLabel(names=list(self.reverse_label_map.values()))
        })

        train_dataset = Dataset.from_dict({'image': train_images, 'labels': train_labels}, features=features)
        val_dataset = Dataset.from_dict({'image': val_images, 'labels': val_labels}, features=features)
        test_dataset = Dataset.from_dict({'image': test_images, 'labels': test_labels}, features=features)

        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })

        return dataset_dict


# Add a debugging function in the training script to check label ranges in batches


# Example usage of debug code in training script
# trainer = Trainer(...)
# debug_training_loop(trainer, trainer.get_train_dataloader())
