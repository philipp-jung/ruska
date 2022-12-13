# Ruska - Philipp's Phd Helper

Ruska contains a bundle of utilities used to carry out data cleaning
experiments.

## Installation

Clone the repository from git and set up virtualenv in the root dir of the package:

```
python3 -m venv venv
```

Install the package from local sources:

```
./venv/bin/pip install -e .
```

To receive Telegram notifications, specify your bot's secrets in a `.env` file
as follows:
```
TELEGRAM_BOT_TOKEN=<token>
TELEGRAM_CHAT_ID=<chat_id>
```

## Running Ruska

To run ruska, you need three things:
- A function, called `experiments`, which accepts parameters called
- `config`. This is a plain dict, following the syntax `{"parameter": "value"}`,
- and `ranges`. That is another dict, following the syntax
`{"parameter": ["list", "of", "parameter", "values", "to", "iterate", "over"],
...}.`

You create an instance of `Ruska` like so:

```python
from ruska import Ruska

rsk = Ruska(name='Measurement Name',
            description='What is the measurement about?',
            commit='hashValueOfTheGitCommit',
            config={"parameter": "value",
            ranges={"parameter": ["list", "of", "params"]},
            runs=3,
            save_path='/path/to/store/result/at',
            )
```

Following this initialization, results will be stored in a plain text file at
`f'{save_path}/{name}.txt`.
A measurement is started by running

```python
rsk.run(experiment, parallel=False)
```

## Quickstart Example

```python
import ruska
import time

def experiment(run):
    time.sleep(2)

rsk = ruska.Ruska(name='test', description='test', commit='', config={'run':0}, ranges={}, runs=3, save_path='/Users/philipp/code/ruska',)

rsk.run(experiment, parallel=False)
```
