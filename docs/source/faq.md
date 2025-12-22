# FAQ

## I reproduced the results with the provided checkpoints but get different results

CARLA evaluation can be volatile, and results may differ between runs. Based on our empirical observations, typical variations are around 1-2 DS on Bench2Drive, 5-7 DS on Longest6 v2, and 1.0 DS on Town13. These are rough estimates from our experience and not strict guarantees—actual variation depends on many factors including randomness.

## Why do we have so many version of `leaderboard` and `scenario_runner`?

Each benchmark has its own evaluation protocol and needs its own forks of those two repositories. Expert data collector also needs its own fork.

## How do I create more routes?

See [carla_route_generator](https://github.com/autonomousvision/carla_route_generator). Also, see Section 5 of [LEAD's supplemental](https://ln2697.github.io/assets/pdf/Nguyen2026LEADSUPP.pdf).

## Can I see a list of modifications applied to `leaderboard` and `scenario_runner`?

We maintain custom forks of CARLA evaluation tools with our modifications:
* [scenario_runner_autopilot](https://github.com/ln2697/scenario_runner_autopilot), [leaderboard_autopilot](https://github.com/ln2697/leaderboard_autopilot), [Bench2Drive](https://github.com/ln2697/Bench2Drive), [scenario_runner](https://github.com/ln2697/scenario_runner), [leaderboard](https://github.com/ln2697/leaderboard)

## Which TransFuser versions are there?

See this [list](https://github.com/autonomousvision/carla_garage/blob/leaderboard_2/docs/history.md).

## How often does CARLA crash or fail to start?

In our experience, roughly 10% of CARLA launch attempts may fail, though this varies by system. Common issues include startup hangs, port conflicts, or GPU initialization problems. This is normal behavior with CARLA.

**What to do:**
- Use `bash scripts/clean_carla.sh` to clean up zombie processes
- Restart CARLA with `bash scripts/start_carla.sh`
- Check that ports 2000-2002 aren't in use
- For Docker: `docker compose restart carla`

## How to add custom scenarios to CARLA?

See [this](https://github.com/autonomousvision/lead/tree/main/3rd_party/scenario_runner_autopilot/srunner/scenarios).

## How does expert access to scenario's specific data?

See [this](https://github.com/autonomousvision/lead/blob/main/3rd_party/scenario_runner_autopilot/srunner/scenariomanager/carla_data_provider.py).
