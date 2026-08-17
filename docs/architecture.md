# Architecture

LEAD is a dataset, and policies that are swappable behind two contracts.

```mermaid
%%{init: {"fontFamily": "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"}}%%
flowchart TB
    subgraph entry[" "]
        direction LR
        train["<b>Training</b><br/><code>lead.training</code>"]
        eval["<b>Closed-loop evaluation</b><br/><code>lead.evaluation</code>"]
    end

    subgraph api["The contracts: <code>lead.api</code>"]
        direction LR
        ap["<b>AbstractPolicy</b><br/>Builds its own dataset,<br/>forward, loss, visualizers"]
        ad["<b>AbstractDrivingAgent</b><br/>Drives CARLA with a policy"]
    end

    train --> ap
    eval --> ad
    ad --> ap

    api -->|"<b>LeadConfig.policy.target</b><br/>names the class to load"| pols

    subgraph pols["Per policy: one package with its dataloader, one driving agent under the same name"]
        direction TB
        tf["<b>Transfuser</b><br/><code>lead.policy.transfuser</code><br/><code>lead.evaluation.agents.transfuser</code>"]
        es["<b>EgoStatus</b><br/><code>lead.policy.ego_status</code><br/><code>lead.evaluation.agents.ego_status</code>"]
        yours["<b>YourPolicy</b><br/><code>lead.policy.your_policy</code><br/><code>lead.evaluation.agents.your_policy</code>"]
    end

    classDef default fill:transparent,stroke:#888888
    style entry fill:transparent,stroke:#88888855
    style api fill:transparent,stroke:#88888855
    style pols fill:transparent,stroke:#88888855
```

Every policy brings its own dataloader and its own driving agent, and one config field decides which
one training builds and which one evaluation drives. Because every module talks through those
contracts instead of to a policy directly, the dependency graph only ever points one way —
machine-checked by the `layered architecture` import contract in `pyproject.toml`:

```mermaid
%%{init: {"fontFamily": "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"}}%%
flowchart LR
    subgraph stack["Application modules"]
        direction TB
        tr["<code>lead.training</code><br/>Runs the training loop"] -->|"Only knows <code>AbstractPolicy</code>"| po["<code>lead.policy</code><br/>The policies and their dataloaders"]
        ev["<code>lead.evaluation</code><br/>Closed-loop evaluation of a policy in CARLA"] --> po
        ev --> dl["<code>lead.log_reader</code><br/>Reads the Py123D logs and returns Py123D objects"]
        po --> dl
        ex["<code>lead.expert</code><br/>Data generator: generates demonstrations and Py123D logs"]
    end

    stack -.->|"Imports"| base["<code>lead.api</code><br/><code>lead.cache</code><br/><code>lead.common</code><br/><code>lead.config</code>"]

    classDef default fill:transparent,stroke:#888888
    style stack fill:transparent,stroke:#88888855
```

## Disk to tensor

One `__getitem__`: read the log, derive one policy's arrays, stack them.

```mermaid
%%{init: {"fontFamily": "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"}}%%
flowchart TB
    subgraph disk["Two sensor views of one 123D log on disk"]
        direction LR
        nv[("<b>normal_view</b><br/>Normal rig")]
        pv[("<b>perturbated_view</b><br/>Shifted rig")]
    end

    subgraph worker["A <code>torch DataLoader</code> worker"]
        idx["<b>AbstractPolicyDataset.__getitem__</b><br/><code>lead.api.abstract_dataset</code><br/>One sample index"]
        idx --> dispatch
        dispatch["<b>SamplePart</b><br/><code>lead.api.training_sample</code><br/>Its outputs already cached?"]
        dispatch -->|Yes| store[("<b>CacheStoreReader</b><br/><code>lead.cache</code>")]
        dispatch -->|No| spec["<b>SamplePart.reads</b> → <b>SceneLoadingSpec</b><br/><code>lead.api.scene_loading_spec</code><br/>Which modalities it needs"]

        subgraph generic[" "]
            spec --> loader["<b>SceneLoader.read()</b><br/><code>lead.log_reader.scene_loader</code><br/>Loads the requested modalities,<br/>plus ego pose, boxes and meta"]
            loader --> sd(["<b>SceneData</b><br/><code>lead.api.scene_data</code><br/>Py123D objects"])
        end

        subgraph featurize["One <code>SamplePart.builds</code> per part"]
            sd --> feat["<b>build_camera_features()</b>, <b>build_lidar_raster()</b>,<br/><b>build_radar_features()</b><br/><code>lead.policy.transfuser.dataloader.features</code><br/>Model inputs: camera, lidar raster, radar"]
            sd --> lab["<b>build_labels()</b>, <b>build_depth_target()</b><br/><code>lead.policy.transfuser.dataloader.label_builders</code><br/>Labels: detections, segmentation, depth"]
            sd --> own["<b>_build_planning_targets()</b>, <b>_build_meta_features()</b><br/><code>lead.policy.transfuser.dataloader.dataset</code><br/>Waypoints, route, driving-meta lifts"]
            feat --> outs
            lab --> outs
            own --> outs
            outs(["<b>Merged part outputs</b><br/><code>dict[str, numpy array]</code>"])
        end
        store --> outs
        outs --> aug["<b>postprocess_outputs()</b>: <b>apply_color_augmentation()</b><br/><code>lead.policy.transfuser.dataloader.augmentation</code><br/>Pixel augmentation on the camera image"]
        aug --> smp(["<b>TransfuserTrainingSample</b><br/><code>lead.policy.transfuser.dataloader.sample</code><br/>Inputs and labels, numpy, no batch axis"])
    end
    disk --> loader

    smp -->|"<b>PolicyTrainingSample.collate</b><br/><code>lead.api.training_sample</code><br/>Stacks samples into one batch"| batch

    subgraph step["Model step: batched, on the GPU"]
        batch(["<b>TransfuserForwardBatch</b><br/><code>dict[str, torch.Tensor]</code><br/>Inputs and labels, with a batch axis"])
        batch --> fwd["<b>Transfuser.forward()</b>"]
    end

    classDef default fill:transparent,stroke:#888888
    style disk fill:transparent,stroke:#88888855
    style worker fill:transparent,stroke:#88888855
    style generic fill:transparent,stroke:#88888855
    style featurize fill:transparent,stroke:#88888855
    style step fill:transparent,stroke:#88888855
```
