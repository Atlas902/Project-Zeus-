"""Small deterministic grid-world for safe decision-logic experiments."""
from dataclasses import dataclass


@dataclass
class GridWorld:
    width: int = 5
    height: int = 5
    position: tuple = (0, 0)
    goal: tuple = (4, 4)
    hazard: tuple = (2, 2)

    def __post_init__(self):
        if isinstance(self.width, bool) or isinstance(self.height, bool):
            raise ValueError("grid dimensions must be positive integers")
        if not isinstance(self.width, int) or not isinstance(self.height, int):
            raise ValueError("grid dimensions must be positive integers")
        if self.width < 1 or self.height < 1:
            raise ValueError("grid dimensions must be positive integers")
        for label, point in (("position", self.position), ("goal", self.goal), ("hazard", self.hazard)):
            if (not isinstance(point, tuple) or len(point) != 2 or
                    any(isinstance(v, bool) or not isinstance(v, int) for v in point)):
                raise ValueError(f"{label} must be an (x, y) integer tuple")
            if not self._inside(point):
                raise ValueError(f"{label} must be inside the grid")
        if self.goal == self.hazard:
            raise ValueError("goal and hazard must be different cells")

    def _inside(self, point):
        return 0 <= point[0] < self.width and 0 <= point[1] < self.height

    @property
    def terminal(self):
        return self.position in (self.goal, self.hazard)

    def options(self):
        if self.terminal:
            return {}
        x, y = self.position
        candidates = {
            "north": (x, y - 1), "south": (x, y + 1),
            "west": (x - 1, y), "east": (x + 1, y),
            "wait": (x, y),
        }
        return {name: pos for name, pos in candidates.items() if self._inside(pos)}

    def step(self, action: str):
        if self.terminal:
            raise ValueError("simulation has ended; reset or create a new world")
        options = self.options()
        if action not in options:
            raise ValueError("Action is not available from current position")
        self.position = options[action]
        if self.position == self.hazard:
            return -1.0, "hazard reached"
        if self.position == self.goal:
            return 1.0, "goal reached"
        if action == "wait":
            return -0.05, "waited"
        return 0.1, "safe move"
