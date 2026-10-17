from qlearning import GridWorld, QLearningAgent

MAZE = [
    "S..#.....",
    ".#.#.###.",
    ".#...#X..",
    ".####.#.#",
    "......#.G",
]


def main():
    world = GridWorld(MAZE)
    agent = QLearningAgent()
    rewards = agent.train(world, episodes=600)

    print("Recompensa medie pe episod (in timpul antrenarii):")
    for start in range(0, len(rewards), 100):
        chunk = rewards[start:start + 100]
        print(f"  episoadele {start + 1:>3}-{start + len(chunk):>3}: {sum(chunk) / len(chunk):6.1f}")

    print("\nPolitica invatata (S start, G tinta, X capcana, # zid):")
    print(agent.policy_map(world))

    path = agent.greedy_path(world)
    print(f"\nDrumul agentului: {len(path) - 1} pasi, de la {path[0]} la {path[-1]}")


if __name__ == "__main__":
    main()
