# Config profiles

Yaml profiles for the config sections listed in `PROFILE_SECTIONS`
(`src/lead/config/lead_config.py`), currently `expert/` and `agent/`.
A profile is a **delta**: it only lists the keys it changes from the Python
defaults in `src/lead/config/<section>/`. An empty file (e.g.
`expert/default.yaml`) means "use the defaults as-is".

## Selecting a profile

Set `<section>.config_profile=<file stem>` via any override source:

```bash
# CLI dotlist
python scripts/... expert.config_profile=leaderboard2_3cameras agent.config_profile=transfuser

# or the LEAD_CONFIG environment variable
LEAD_CONFIG="expert.config_profile=leaderboard2_3cameras" python scripts/...
```

Profiles are applied first, so every other source (loaded config file,
`LEAD_CONFIG`, CLI) can still override individual keys on top.

## Adding a profile

Drop a `<name>.yaml` into the section directory with only the keys you want
to change (nested, matching the config tree, e.g. `transfuser.image_architecture`).
It becomes selectable as `<section>.config_profile=<name>` — no registration
needed. Unknown keys raise at load time.
