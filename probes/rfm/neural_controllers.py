"""
Neural Controllers for LLM Steering

This module provides the NeuralController class which orchestrates:
- Computing steering directions from labeled data
- Applying steering to model generations via forward hooks
- Detecting concept presence in inputs

Supported control methods: RFM, Linear, Logistic, Mean Difference, PCA
"""

import torch
import random
import numpy as np

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

from probes.rfm import direction_utils, generation_utils
from probes.rfm.control_toolkits import (
    RFMToolkit, LinearProbeToolkit, LogisticRegressionToolkit,
    MeanDifferenceToolkit, PCAToolkit
)
from probes.rfm.utils import LLMType

import os
import pickle
from tqdm import tqdm
import shutil

TOOLKITS = {
    'rfm': RFMToolkit,
    'linear': LinearProbeToolkit,
    'logistic': LogisticRegressionToolkit,
    'mean_difference': MeanDifferenceToolkit,
    'pca': PCAToolkit
}


class NeuralController:
    """
    Controller for computing and applying steering directions to LLMs.

    This class handles the full pipeline of:
    1. Computing steering directions from labeled training data
    2. Saving/loading directions to/from disk
    3. Applying directions during generation via forward hooks
    4. Detecting concept presence in input prompts

    Attributes:
        llm: The language model wrapper (namedtuple with model, tokenizer, etc.)
        model: The underlying PyTorch model
        directions: Dictionary mapping layer indices to direction vectors
        hidden_layers: List of layer indices to target for steering
        control_method: Algorithm used for direction computation
    """
    def __init__(self, llm, tokenizer, control_method='rfm', n_components=5, 
                 rfm_iters=8, batch_size=16):
        self.llm = llm
        self.model = llm.language_model.eval()
        if llm.model_type == LLMType.MULTIMODAL:
            self.language_model = llm.language_model.language_model.model
        elif llm.model_type == LLMType.MULTIMODAL_DEEPSEEK:
            self.language_model = llm.language_model.language_model.model
        else:
            self.language_model = self.model
            
        self.tokenizer = tokenizer
        self.processor = llm.processor
        self.control_method = control_method
        self.llm_type = llm.model_type
        self.name = None

        hparams = {
            'control_method' : control_method,
            'rfm_iters' : rfm_iters,
            'forward_batch_size' : batch_size,
            'M_batch_size' : 2048,
            'n_components' : n_components,
        }
        self.hyperparams = hparams
        
        # if 'concat' in control_method:
        #     self.hidden_layers = ['concat']
        # else:
        #     self.hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
        if 'concat' in control_method:
            self.hidden_layers = ['concat']
        elif llm.model_type == LLMType.MULTIMODAL:
            self.hidden_layers = get_non_cross_attention_layer_indices_after_first_cross(self.language_model)
        elif llm.model_type == LLMType.MULTIMODAL_DEEPSEEK:
            self.hidden_layers = list(range(-1, -self.language_model.config.num_hidden_layers, -1))
        else:
            self.hidden_layers = list(range(-1, -self.model.config.num_hidden_layers, -1))

        self.toolkit = TOOLKITS[control_method]()
        self.signs = None
        self.detector_coefs = None

        print('Hidden layers:', self.hidden_layers)
        print("\nController hyperparameters:")
        for n_, v_ in self.hyperparams.items():
            print(f"{n_:<20} : {v_}")
        print()

    def describe(self):
        def print_in_dashed_box(lines):
            # Determine the longest line for box width
            terminal_width = shutil.get_terminal_size().columns
            max_length = max(len(line) for line in lines)
            box_width = min(terminal_width, max_length + 4)

            # Print top border
            print('-' * box_width)

            # Print each line with padding and add dashed separator between lines
            for i, line in enumerate(lines):
                print(f"{line.ljust(box_width)}")
                if i < len(lines) - 1:  # Only add separator between lines, not after the last line
                    print('-' * box_width)

            # Print bottom border
            print('-' * box_width)

        lines = ['Controller Description:']
        for name, module in self.model.named_modules():
            lines.append(f"Model: {module}")
            break
        lines.append(f'Control method: {self.control_method}')
        lines.append(f'Tracked layers: {self.hidden_layers}')
            
        print_in_dashed_box(lines)


        
    def compute_directions(self, data, labels, hidden_layers=None, **kwargs):
        if hidden_layers is None:
            hidden_layers = self.hidden_layers
        self.hidden_layers = hidden_layers
        
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels).reshape(-1,1)
        self.directions, self.signs, self.detector_coefs, _, _ = self.toolkit._compute_directions(data, 
                                                           labels,
                                                           self.llm,
                                                           self.model, 
                                                           self.tokenizer,                                                    
                                                           self.hidden_layers, 
                                                           self.hyperparams,
                                                           **kwargs
                                                          )
        
    def compute_directions_and_accs(self, 
                                    train_data, train_labels, 
                                    test_data, test_labels, 
                                    hidden_layers=None, **kwargs):
        
        if hidden_layers is None:
            hidden_layers = self.hidden_layers
        self.hidden_layers = hidden_layers
        
        if not isinstance(train_labels, torch.Tensor):
            train_labels = torch.tensor(train_labels).reshape(-1,1)
        if not isinstance(test_labels, torch.Tensor):
            test_labels = torch.tensor(test_labels).reshape(-1,1)
            
        
        self.directions, self.signs, self.detector_coefs, direction_accs, predictor_accs = self.toolkit._compute_directions(
                                                           train_data, 
                                                           train_labels, 
                                                           self.model, 
                                                           self.tokenizer, 
                                                           self.hidden_layers, 
                                                           self.hyperparams,
                                                           test_data,
                                                           test_labels,
                                                           **kwargs
                                                          )
        
        return direction_accs, predictor_accs
    
    def evaluate_directions(self,
                            val_data, val_labels,
                            test_data, test_labels,
                            hidden_layers=None, 
                            n_components=1,
                            agg_positions=False,
                            use_logistic=False,
                            use_rfm=False,
                            unsupervised=False
                           ):
        
        if hidden_layers is None:
            hidden_layers = self.hidden_layers
        self.hidden_layers = hidden_layers

        if not isinstance(val_labels, torch.Tensor):
            val_labels = torch.tensor(val_labels).reshape(-1,1)
        if not isinstance(test_labels, torch.Tensor):
            test_labels = torch.tensor(test_labels).reshape(-1,1)

        if len(val_labels.shape) == 1:
            val_labels = val_labels.reshape(-1,1)
        if len(test_labels.shape) == 1:
            test_labels = test_labels.reshape(-1,1)
        
        val_y = val_labels.to(self.model.device).float()
        test_y = test_labels.to(self.model.device).float()
        assert(val_y.shape[1]==test_y.shape[1])

        val_hidden_states = direction_utils.get_hidden_states(val_data, 
                                                              self.model, 
                                                              self.tokenizer, 
                                                              hidden_layers, 
                                                              self.hyperparams['forward_batch_size'],
                                                              all_positions=agg_positions
                                                             )
        
        test_hidden_states = direction_utils.get_hidden_states(test_data, 
                                                              self.model, 
                                                              self.tokenizer, 
                                                              hidden_layers, 
                                                              self.hyperparams['forward_batch_size'],
                                                              all_positions=agg_positions
                                                             )
        
        projections = {
                        'val' : [],
                        'test' : []
                    }
        val_metrics = {}
        test_metrics = {}
        detector_coefs = {}
        
        for layer_to_eval in tqdm(hidden_layers):
            direction = self.directions[layer_to_eval]
            if isinstance(direction, np.ndarray):
                direction = torch.from_numpy(direction)
            direction = direction.to(self.model.device).float()[:n_components]
            direction = direction.T
            
            val_X = val_hidden_states[layer_to_eval].cuda().float()
            projected_val = val_X@direction
            
            test_X = test_hidden_states[layer_to_eval].cuda().float()
            projected_test = test_X@direction
            
            if agg_positions:
                projected_val = torch.mean(projected_val, dim=1) # mean projection
                projected_test = torch.mean(projected_test, dim=1) # mean projection
                            
            if use_logistic:
                beta, b = direction_utils.logistic_solve(projected_val, val_y)
            else:
                beta, b = direction_utils.linear_solve(projected_val, val_y)
            print("Learned beta:", beta, "Learned intercept", b)
            
            detector_coefs[layer_to_eval] = [beta, b]
     
            if unsupervised: # evaluate sign on test data
                projected_test_preds = projected_test
                projected_test_preds = torch.where(projected_test_preds>0, 1, 0)
                
                projected_val_preds = projected_val
                projected_val_preds = torch.where(projected_val_preds>0, 1, 0)
            else: # evaluate slope, intercept on test data
                projected_val_preds = projected_val@beta + b
                projected_test_preds = projected_test@beta + b
       
            assert(projected_test_preds.shape==test_y.shape)
            
            val_metrics_on_layer = direction_utils.compute_classification_metrics(projected_val_preds, val_y)
            val_metrics[layer_to_eval] = val_metrics_on_layer
            
            test_metrics_on_layer = direction_utils.compute_classification_metrics(projected_test_preds, test_y)
            test_metrics[layer_to_eval] = test_metrics_on_layer
            
            projections['val'].append(projected_val.reshape(-1, n_components))
            projections['test'].append(projected_test.reshape(-1, n_components))
        
        # print("Aggregating predictions over layers using linear stacking")
        agg_metrics, agg_beta, agg_bias = direction_utils.aggregate_layers(projections, val_y, test_y, use_logistic, use_rfm)
        test_metrics['linear_agg'] = agg_metrics
            
        detector_coefs['agg'] = [agg_beta, agg_bias]
        return val_metrics, test_metrics, detector_coefs
    
    
    def detect(self, prompts, rep_layer=-15, use_rep_layer=False, use_avg_projection=False):
        hidden_states = direction_utils.get_hidden_states(
                            prompts, 
                            self.model, 
                            self.tokenizer, 
                            self.hidden_layers, 
                            self.hyperparams['forward_batch_size'],
                            all_positions=True
                         )
        
        projections = direction_utils.project_hidden_states(hidden_states, self.directions, self.hyperparams['n_components'])
        
        if use_avg_projection:
            scores = 0
            num_layers = 0
            for layer, h in projections.items():
                if layer!='agg' and layer>-21:
                    scores += h 
                    num_layers+=1
            
            preds = 0.5 + scores / num_layers # bias to mean 0.5
            
        elif 'agg' in self.detector_coefs and not use_rep_layer:
            preds = direction_utils.aggregate_projections_on_coefs(projections, self.detector_coefs['agg'])
            
        else:
            beta, b = self.detector_coefs[rep_layer]
            x = projections[rep_layer]
            preds = x@beta + b
            
        return preds.squeeze()
    
    def get_composite_directions(self,
                            val_data, val_labels,
                            n_components,
                            hidden_layers=None, 
                            agg_positions=False,
                            use_logistic=False
                           ):
        
        if hidden_layers is None:
            hidden_layers = self.hidden_layers
        self.hidden_layers = hidden_layers
        
        val_y = torch.tensor(val_labels).to(self.model.device).float().reshape(-1,1)
        val_hidden_states = direction_utils.get_hidden_states(val_data, 
                                                              self.model, 
                                                              self.tokenizer, 
                                                              hidden_layers, 
                                                              self.hyperparams['forward_batch_size'],
                                                              all_positions=agg_positions
                                                             )
        
        composite_directions = {}
        
        for layer_to_eval in tqdm(hidden_layers):
            direction = self.directions[layer_to_eval]
            if isinstance(direction, np.ndarray):
                direction = torch.from_numpy(direction)
            direction = direction.to(self.model.device).float()[:n_components]
            direction = direction.T
            
            val_X = val_hidden_states[layer_to_eval].cuda().float()
            projected_val = val_X@direction
            
            beta = direction_utils.linear_solve(projected_val, val_y, use_bias=False)
            print("Learned beta:", beta)
            
            composite_vec = direction@beta
            composite_vec = composite_vec.reshape(1,-1)
            composite_directions[layer_to_eval] = composite_vec / composite_vec.norm()
                
        return composite_directions
        
    def save(self, concept, model_name, path='./', composite=False):
        if composite:
            filename = os.path.join(path, f'{self.control_method}_composite_{concept}_{model_name}.pkl')
        else:
            filename = os.path.join(path, f'{self.control_method}_{concept}_{model_name}.pkl')
            
        with open(filename, 'wb') as f:
            pickle.dump(self.directions, f)

        if self.detector_coefs is not None:
            detector_path = os.path.join(path, f'{self.control_method}_{concept}_{model_name}_detector.pkl')
            with open(detector_path, 'wb') as f:
                pickle.dump(self.detector_coefs, f)
            
    def load(self, concept, model_name, path='./', composite=False):
        if composite:
            filename = os.path.join(path, f'{self.control_method}_composite_{concept}_{model_name}.pkl')
        else:
            filename = os.path.join(path, f'{self.control_method}_{concept}_{model_name}.pkl')
        with open(filename, 'rb') as f:
            self.directions = pickle.load(f)
            self.hidden_layers = self.directions.keys()
        
        detector_path = os.path.join(path, f'{self.control_method}_{concept}_{model_name}_detector.pkl')
        if os.path.exists(detector_path):
            print("Detector found")
            with open(detector_path, 'rb') as f:
                self.detector_coefs = pickle.load(f)
        
    def format_prompt(self, prompt, role='user'):
        if self.name == 'toxicchat-t5-large':
            new_prompt = f"ToxicChat: {prompt}"
            return new_prompt
        
        if self.llm.model_type == LLMType.TEXT:
            chat = [{"role": "user", "content": prompt}]
        elif self.llm.model_type == LLMType.GEMMA_TEXT:
            chat = [{"role": "user", "content": [{"type": "text", "text": prompt},]}]
        out = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True).strip()
        return out

    def generate(self, plaintext_prompt, image=None, layers_to_control=[], control_coef=0.4, **kwargs):
        """
        Generate text with optional steering applied.

        Args:
            plaintext_prompt: The user prompt (will be formatted with chat template)
            image: Optional image for multimodal models
            layers_to_control: List of layer indices to apply steering to
            control_coef: Steering coefficient (higher = stronger steering)
            **kwargs: Additional arguments passed to generation (max_new_tokens, etc.)

        Returns:
            Generated text string
        """
        if image is not None and self.llm_type == LLMType.MULTIMODAL:
            if len(layers_to_control) == 0:
                return generation_utils.generate_on_image_and_text(self.model, self.processor, plaintext_prompt, image, **kwargs)
            else:
                return self._controlled_multimodal_generate(plaintext_prompt, image, layers_to_control, control_coef, **kwargs)
        elif image is not None and self.llm_type == LLMType.MULTIMODAL_DEEPSEEK:
            if len(layers_to_control) == 0:
                return generation_utils.generate_on_image_and_text_deepseek(self.llm, self.processor, plaintext_prompt, image, **kwargs)
            else:
                return self._controlled_multimodal_generate(plaintext_prompt, image, layers_to_control, control_coef, **kwargs)
        else:
            prompt = self.format_prompt(plaintext_prompt)
            if len(layers_to_control) == 0:
                return generation_utils.generate_on_text(self.model, self.tokenizer, prompt, **kwargs)
            else:
                return self._controlled_generate(prompt, layers_to_control, control_coef, **kwargs)


    def _controlled_generate(self, prompt, layers_to_control, control_coef, **kwargs):
        ## define hooks
        hooks = generation_utils.hook_model(self.model, self.directions, layers_to_control, control_coef)

        ## do forward pass
        out = generation_utils.generate_on_text(self.model, self.tokenizer, prompt, **kwargs)

        ## clear hooks
        generation_utils.clear_hooks(hooks)
        return out

    def _controlled_multimodal_generate(self, plaintext_prompt, image, layers_to_control, control_coef, **kwargs):
        ## define hooks
        hooks = generation_utils.hook_model(self.model, self.directions, layers_to_control, control_coef)

        ## do forward pass
        if self.llm.model_type == LLMType.MULTIMODAL_DEEPSEEK:
            out = generation_utils.generate_on_image_and_text_deepseek(self.llm, self.processor, plaintext_prompt, image, **kwargs)
        else:
            out = generation_utils.generate_on_image_and_text(self.model, self.processor, plaintext_prompt, image, **kwargs)

        ## clear hooks
        generation_utils.clear_hooks(hooks)
        return out


def get_non_cross_attention_layer_indices_after_first_cross(model):
    """
    Returns a list of indices of all layers in the language model (model.language_model.model.layers)
    that are NOT cross-attention layers and that come after the first cross-attention layer.
    
    Assumes:
      - The layers are stored in model.language_model.model.layers.
      - Cross-attention layers have "CrossAttention" in their class name.
    """
    layers = model.layers
    first_cross_idx = None
    
    # Find the index of the first cross-attention layer.
    for idx, layer in enumerate(layers):
        if "CrossAttention" in layer.__class__.__name__:
            first_cross_idx = idx
            break
            
    if first_cross_idx is None:
        print("No cross-attention layers found in the language model.")
        return []
    
    # Collect indices for all layers after the first cross-attention layer that are NOT cross-attention.
    non_cross_indices = []
    for idx in range(first_cross_idx + 1, len(layers) - 1):
        if "CrossAttention" not in layers[idx].__class__.__name__:
            non_cross_indices.append(idx)
    
    return non_cross_indices

