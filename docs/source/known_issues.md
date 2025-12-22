# Known Issues

We report some issues that are well-known for our pipeline.

## Bugs on Town13 Evaluation

As the Town13 videos show on the [project's official website](https://ln2697.github.io/lead), our evaluation pipeline is currently not bug-free yielding to some problems with target points on Town13.

This issue degrades policy's performance so the numbers on Town13 does not mirror the policy's true capability.

## Multi-GPU Training Can Slightly Degrade Policy Performance

In some settings, training on 4 GPUs yields marginally lower closed-loop performance compared to single-GPU training. While the effect is small and does not change qualitative conclusions, it is consistently observable in certain runs.

## CARLA Waypoint PID Controller Is Not Well-Tuned

A well-tuned controller (MPC) can improve the performance quite significantly. In preliminary experiments, we observed improvements of approximately 5–7 DS on Bench2Drive; however, these numbers should be interpreted cautiously as controller tuning was not the focus of this work.

## CARLA 0.9.16 Is Not Working Properly Right Now

At the time of writing, CARLA 0.9.16 exhibits issues in the goal-point pipeline, leading to degraded policy behavior. As a result, we do not recommend evaluating the current models on this version.

## Expert Performance on Town13

The current expert performs reliably on short and medium routes but exhibits a notable performance drop on Town13. The underlying causes are under investigation.

## Expert Optimality

The provided expert is intentionally designed to be simple and extensible rather than optimal. We explicitly encourage future work to improve the expert and to explore alternative designs within the LEAD framework.
