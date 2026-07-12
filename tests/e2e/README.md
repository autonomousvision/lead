# End-to-end tests

Tests in this directory drive the full stack (expert data collection,
evaluation agents) against a **running CARLA server** and are therefore not
part of the regular CI unit test run (`pytest tests/unittests`).

Requirements:

- A CARLA 0.9.16 server reachable on `localhost:2000` (start one with
  `scripts/bin/start_carla`).

Run with:

```bash
pytest tests/e2e
```
