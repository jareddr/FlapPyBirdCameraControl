import asyncio
import logging

from src.flappy import Flappy

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(name)s: %(message)s",
    )
    asyncio.run(Flappy().start())
