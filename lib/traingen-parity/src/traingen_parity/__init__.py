"""Shared training parity helpers for generator packages.

Import helpers from their submodules: ``traingen_parity.compare``,
``traingen_parity.determinism``, ``traingen_parity.trace``, and
``traingen_parity.tensorflow_compat``. The package root re-exports nothing, so
importing ``traingen_parity.tensorflow_compat`` in a TensorFlow reference
environment does not import PyTorch.
"""
