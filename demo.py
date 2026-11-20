"""Run a supervised toy grid-world demonstration."""
from .brain import ZeusBrain
from .memory import ExperienceMemory
from .simulator import GridWorld


def make_features(world, target, action):
    gx, gy = world.goal
    x, y = target
    current_x, current_y = world.position
    current_distance = abs(gx - current_x) + abs(gy - current_y)
    next_distance = abs(gx - x) + abs(gy - y)
    # Keep feature magnitudes comparable; explicitly penalize entering the hazard.
    hazard_safety = -1.0 if target == world.hazard else 1.0
    return {"move_toward_goal": float(current_distance - next_distance),
            "avoid_hazard": hazard_safety,
            "wait": 1.0 if action == "wait" else 0.0}


def main():
    world = GridWorld()
    brain = ZeusBrain()
    memory = ExperienceMemory()
    print("PROJECT ZEUS 0.3 — audited decision-learning simulation")
    print("Starting at", world.position, "goal:", world.goal, "hazard:", world.hazard)

    for turn in range(30):
        available = world.options()
        if not available:
            break
        features = {action: make_features(world, target, action)
                    for action, target in available.items()}
        decision = brain.decide(features, memory.all())
        if decision is None:
            break
        chosen_features = features[decision.action]
        reward, outcome = world.step(decision.action)
        memory.record({"position": world.position}, decision.action, reward,
                      outcome, chosen_features)
        brain.learn(decision.action, reward, chosen_features)
        print(f"Turn {turn + 1}: {decision.action:5} -> {world.position}; {outcome}; "
              f"relative confidence {decision.confidence:.0%}")
        print("  Why:", decision.reason)
        if world.terminal:
            break

    print("\nExperience report:", memory.summary())
    print("Learning status:", brain.status())


if __name__ == "__main__":
    main()
