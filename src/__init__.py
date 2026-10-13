"""VG research utilities with optional SAE dependencies loaded on demand.

Public names retain their original identities. Importing the core VG model or
NumPy density analysis does not import the optional SAELens training stack.
"""
from importlib import import_module

_EXPORTS = {
    'BaselineSAELossTerms': ('.sae_loss', 'BaselineSAELossTerms'),
    'ActivationCache': ('.saelens_data', 'ActivationCache'),
    'ActivationScale': ('.saelens_data', 'ActivationScale'),
    'ActivationSplits': ('.saelens_data', 'ActivationSplits'),
    'BatchTopKSAE': ('.sae_model', 'BatchTopKSAE'),
    'BatchTopKSAEConfig': ('.sae_model', 'BatchTopKSAEConfig'),
    'EvaluationResult': ('.evaluate', 'EvaluationResult'),
    'RegressionTensors': ('.data', 'RegressionTensors'),
    'SyntheticConfig': ('.data', 'SyntheticConfig'),
    'SyntheticSparseCodingConfig': ('.sae_data', 'SyntheticSparseCodingConfig'),
    'GatedSAE': ('.sae_model', 'GatedSAE'),
    'GatedSAEConfig': ('.sae_model', 'GatedSAEConfig'),
    'JumpReLU': ('.sae_model', 'JumpReLU'),
    'JumpReLUSAE': ('.sae_model', 'JumpReLUSAE'),
    'JumpReLUSAEConfig': ('.sae_model', 'JumpReLUSAEConfig'),
    'StandardSAE': ('.sae_model', 'StandardSAE'),
    'StandardSAEConfig': ('.sae_model', 'StandardSAEConfig'),
    'SAETrainResult': ('.sae_train', 'SAETrainResult'),
    'SAELensVGSAE': ('.saelens_vg', 'VGSAE'),
    'SAELensVGSAEConfig': ('.saelens_vg', 'VGSAEConfig'),
    'SplitIndices': ('.saelens_data', 'SplitIndices'),
    'Step': ('.sae_model', 'Step'),
    'TopKSAE': ('.sae_model', 'TopKSAE'),
    'TopKSAEConfig': ('.sae_model', 'TopKSAEConfig'),
    'TensorActivationProvider': ('.saelens_data', 'TensorActivationProvider'),
    'TrainHistory': ('.train', 'TrainHistory'),
    'TrainResult': ('.train', 'TrainResult'),
    'VGConfig': ('.model', 'VGConfig'),
    'VGSAEConfig': ('.sae_model', 'VGSAEConfig'),
    'VGSAELossTerms': ('.sae_loss', 'VGSAELossTerms'),
    'VGSAETrainer': ('.saelens_vg', 'VGSAETrainer'),
    'VGTrainingSAE': ('.saelens_vg', 'VGTrainingSAE'),
    'VGTrainingSAEConfig': ('.saelens_vg', 'VGTrainingSAEConfig'),
    'VGLossTerms': ('.loss', 'VGLossTerms'),
    'VariationalGarrote': ('.model', 'VariationalGarrote'),
    'VariationalGarroteSAE': ('.sae_model', 'VariationalGarroteSAE'),
    'bernoulli_kl_from_lambda': ('.sae_loss', 'bernoulli_kl_from_lambda'),
    'build_sae': ('.sae_train', 'build_sae'),
    'as_activation_tensor': ('.saelens_data', 'as_activation_tensor'),
    'energy_term': ('.loss', 'energy_term'),
    'fit_sae': ('.sae_train', 'fit_sae'),
    'free_energy_loss': ('.loss', 'free_energy_loss'),
    'generalization_error': ('.evaluate', 'generalization_error'),
    'generalization_error_numpy': ('.evaluate', 'generalization_error_numpy'),
    'infer_data_sparsity': ('.evaluate', 'infer_data_sparsity'),
    'make_synthetic_regression': ('.data', 'make_synthetic_regression'),
    'make_synthetic_sparse_coding': ('.sae_data', 'make_synthetic_sparse_coding'),
    'make_synthetic_train_test': ('.data', 'make_synthetic_train_test'),
    'make_split_indices': ('.saelens_data', 'make_split_indices'),
    'load_saelens_activation_cache': ('.saelens_data', 'load_activation_cache'),
    'register_vg_saes': ('.saelens_vg', 'register_vg_saes'),
    'run_from_config': ('.train', 'run_from_config'),
    'sae_loss_terms': ('.sae_loss', 'sae_loss_terms'),
    'sample_spike_and_slab': ('.data', 'sample_spike_and_slab'),
    'selection_error': ('.evaluate', 'selection_error'),
    'selection_uncertainty': ('.evaluate', 'selection_uncertainty'),
    'split_activations': ('.saelens_data', 'split_activations'),
    'sweep_gamma': ('.train', 'sweep_gamma'),
    'theoretical_e_sel': ('.evaluate', 'theoretical_e_sel'),
    'theoretical_sigma_sel': ('.evaluate', 'theoretical_sigma_sel'),
    'train_vg': ('.train', 'train_vg'),
    'to_inference_sae': ('.sae_baselines', 'to_inference_sae'),
    'vg_free_energy': ('.loss', 'vg_free_energy'),
    'vg_loss_terms': ('.loss', 'vg_loss_terms'),
    'vg_sae_loss_terms': ('.sae_loss', 'vg_sae_loss_terms'),
}

__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value

def __dir__():
    return sorted(set(globals()) | set(__all__))
