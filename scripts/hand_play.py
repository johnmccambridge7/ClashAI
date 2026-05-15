from __future__ import annotations

from coc_env.layouts import default_layout
from coc_env.simulator import Simulator
from coc_env.render_pygame import play


def sim_factory() -> Simulator:
    return Simulator(layout=default_layout(), army_size=10)


SCHEDULE: list[tuple[int, int, int]] = [
    (i * 3, 12 + (i % 20), 43) for i in range(10)
]


if __name__ == "__main__":
    play(sim_factory, fps=30, autoplay=True, deploy_schedule=SCHEDULE)
