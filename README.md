# Project Zeus 0.3

Project Zeus is a small, transparent decision-learning sandbox. It evaluates
candidate actions in a toy grid-world, explains its recommendations, stores
experience, and makes bounded updates from feedback.

## Features

- Transparent weighted scoring for candidate actions.
- Similar-experience adjustment weighted by feature similarity.
- Bounded feedback learning with validation of non-finite inputs.
- Relative confidence heuristic and human-readable explanations.
- Append-only JSON Lines memory with corruption reporting.
- Performance summaries for reward and outcome counts.
- A terminal-state-aware grid-world simulator.
- Standard-library-only implementation; no third-party dependencies.

## Requirements

- Python 3.9 or newer.
- No external packages required.

## Run from the project folder

```bash
python -m zeus.demo
python -m unittest discover -s tests -v
```

The demo writes its local experience log to `zeus_memory.jsonl`. That generated
runtime file is ignored by Git and should not be committed.

## Project layout

```text
project-zeus/
├── README.md
├── .gitignore
├── zeus/
│   ├── __init__.py
│   ├── brain.py
│   ├── memory.py
│   ├── simulator.py
│   └── demo.py
└── tests/
    └── test_zeus.py
```

## Important limitations

Zeus 0.3 is a learning prototype, not a general intelligence system or a
validated machine-learning model. Its confidence is a relative score-gap
heuristic, not a calibrated probability. Its memory matching is intentionally
simple and can become slower as the log grows. It does not train a neural
network, connect to external services, or control physical devices. Treat its
outputs as experimental recommendations, review changes, and test only in the
included simulation.

## Safety scope

This repository is for transparent decision-making experiments and harmless
toy simulations. It is not designed for weapon control, targeting, guidance,
attack planning, or autonomous harmful actions.

## Development checks

```bash
python -m unittest discover -s tests -v
```
