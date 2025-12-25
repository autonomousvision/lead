Documentation
=============

This documentation is organized into three sections.
**Basic:** Quickstart guides to get you started.
**SLURM:** Infrastructure for large-scale experiments.
**In-depth:** Details you do not need now, but may want to know later.

The basic section is sufficient to get started with the local data collection, training and evaluation loop.

.. image:: ../assets/pipeline.png
   :alt: LEAD Pipeline
   :align: center

|

.. toctree::
   :maxdepth: 2
   :caption: Basic

   data_collection
   carla_training
   evaluation
   jupyter_notebooks

.. toctree::
   :maxdepth: 2
   :caption: SLURM

   slurm_overview
   slurm_data_collection
   slurm_training
   slurm_evaluation

.. toctree::
   :maxdepth: 1
   :caption: In-Depth

   inspecting_pickle_files
   cross_dataset_training
   config_system
   debug_expert

.. toctree::
   :maxdepth: 1
   :caption: Miscellaneous

   faq
   glossary
   known_issues
