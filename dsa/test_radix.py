import unittest
from radix import RadixTree, RadixTreeNode


class TestRadixTreeNode(unittest.TestCase):
    def setUp(self):
        self.node = RadixTreeNode()

    def test_simple_insert(self):
        self.node.insert("hello")

        self.assertEqual(list(self.node.children.keys()), ["h"])
        self.assertEqual(self.node.children["h"][0], "hello")

    def test_insert_with_duplicate(self):
        self.node.insert("hello")
        self.node.insert("hello")
        self.assertEqual(list(self.node.children.keys()), ["h"])
        self.assertEqual(self.node.children["h"][0], "hello")

    def test_insert_with_overlap(self):
        self.node.insert("overwhelmed")
        self.node.insert("overreached")
        self.assertEqual(list(self.node.children.keys()), ["o"])
        child_word, child_node = self.node.children["o"]
        self.assertEqual(child_word, "over")
        self.assertIsInstance(child_node, RadixTreeNode)

        self.assertEqual(list(sorted(child_node.children.keys())), ["r", "w"])

        child_a_word, child_a_node = child_node.children["r"]
        self.assertEqual(child_a_word, "reached")
        self.assertIsInstance(child_a_node, RadixTreeNode)

        child_h_word, child_h_node = child_node.children["w"]
        self.assertEqual(child_h_word, "whelmed")
        self.assertIsInstance(child_h_node, RadixTreeNode)

    def test_insert_with_overlap_and_duplicate(self):
        self.node.insert("overwhelmed")
        self.node.insert("overreached")
        self.node.insert("overwhelmed")  # Insert duplicate

        self.assertEqual(list(self.node.children.keys()), ["o"])
        child_word, child_node = self.node.children["o"]
        self.assertEqual(child_word, "over")
        self.assertIsInstance(child_node, RadixTreeNode)

        self.assertEqual(list(sorted(child_node.children.keys())), ["r", "w"])

        child_a_word, child_a_node = child_node.children["r"]
        self.assertEqual(child_a_word, "reached")
        self.assertIsInstance(child_a_node, RadixTreeNode)

        child_h_word, child_h_node = child_node.children["w"]
        self.assertEqual(child_h_word, "whelmed")
        self.assertIsInstance(child_h_node, RadixTreeNode)


if __name__ == "__main__":
    unittest.main()
