import random
import ruska

if __name__ == "__main__":
    from time import sleep

    config = {"dataset": "letter", "strategy": "paint", "model": "dtc"}
    ranges = {"dataset": ["paper", "letter", "restaurant"]}
    ruska = ruska.Ruska(
        "test",
        "Eine Test-Messung, die ich ausführe, um zu testen.",
        "f12fadsdfads",
        config,
        ranges,
        10,
        save_path="/Users/philipp/code/ruska/",
    )

    def fun(config):
        t = random.randint(0,3)
        sleep(t)
        return {"result": None, "config": config}

    ruska.run(fun, parallel=True)
