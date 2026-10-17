import unittest

from qlearning import GridWorld, QLearningAgent

CORRIDOR = ["S..G"]

TRAP = [
    "S.X",
    "...",
    "..G",
]


class GridWorldTest(unittest.TestCase):
    def test_walls_and_edges_block_movement(self):
        world = GridWorld(["S#", ".."])
        self.assertEqual(world.step((0, 0), ">")[0], (0, 0))
        self.assertEqual(world.step((0, 0), "^")[0], (0, 0))
        self.assertEqual(world.step((0, 0), "v")[0], (1, 0))

    def test_rewards(self):
        world = GridWorld(TRAP)
        self.assertEqual(world.step((0, 1), ">"), ((0, 2), -10, True))
        self.assertEqual(world.step((1, 2), "v"), ((2, 2), 10, True))
        self.assertEqual(world.step((0, 0), "v"), ((1, 0), -1, False))


class AgentTest(unittest.TestCase):
    def test_learns_shortest_path_in_corridor(self):
        world = GridWorld(CORRIDOR)
        agent = QLearningAgent()
        agent.train(world, episodes=100)
        self.assertEqual(agent.greedy_path(world), [(0, 0), (0, 1), (0, 2), (0, 3)])

    def test_avoids_trap(self):
        world = GridWorld(TRAP)
        agent = QLearningAgent()
        agent.train(world, episodes=300)
        path = agent.greedy_path(world)
        self.assertEqual(path[-1], (2, 2))
        self.assertNotIn((0, 2), path)
        self.assertEqual(len(path) - 1, 4)


if __name__ == "__main__":
    unittest.main()
