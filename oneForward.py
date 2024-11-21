from datasets import load_dataset

from forward_operation import ForwardOperation
from TinyImageNetLoader import TinyImageNetLoader  # Ensure this loader is defined as required
from vit_model_preparation import ViTModelPreparation
# Initialize your TinyImageNet dataset
model_name_or_path = './vit-output'

model_prep = ViTModelPreparation(load_dataset('beans'), model_name_or_path)
prepared_ds = model_prep.prepare_dataset()
# Instantiate and run the forward operation

forward_op = ForwardOperation(model_name_or_path, prepared_ds)
forward_op.run_forward_pass()
