import asyncio
from dt.pipeline import Pipeline

if __name__ == "__main__":
    pl = Pipeline("config/config.yaml")
    asyncio.run(pl.run())
