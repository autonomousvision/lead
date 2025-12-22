# Glossary

### Route
**Synonym:** Path, spatial path, checkpoints
**Definition:** Spatial path with 1m distance between each point.

### Waypoints
**Synonym:** Trajectory, future positions
**Definition:** Spatio-temporal trajectory, describes future positions.

### Goal point
**Synonym:** Target point
**Definition:** Point in ego-coordinate system depicts where to drive to.

### Target speed
**Synonym:** Desired speed, reference speed
**Definition:** Desired velocity the vehicle should achieve.

### Command
**Synonym:** Navigation command, high-level command
**Definition:** High-level navigation instruction (left, right, straight, lane follow, etc.).

### Occupancy map
**Definition:** Rasterized image with dynamic actors displayed.

### HD-Map
**Definition:** Rasterized image with road structure displayed.

### BEV Semantic Segmentation
**Definition:** HD-Map + Occupancy map.

### Persistent Cache
**Synonym:** HDD-Cache
**Definition:** Training data cache that is stored on disk and persists after training. Should be built only once.

### Training Session Cache
**Definition:** Mostly used in SLURM context where each SLURM job gets its own fast SSD assigned.

### Perturbation
**Synonym:** Sensor perturbation, sensor augmentation.
**Definition:** Shift and rotate sensor rig to simulate deviation from route and teach model to recover.

### Ensemble
**Synonym:** Averaging.
**Definition:** Ensemble multiple seeds for more robust prediction.

### Buckets
**Related:** Bucket collection
**Definition:** A bucket = a subset of the dataset. A bucket collection = a list of buckets. Should be built only once.
