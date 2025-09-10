import unittest
from radix import RadixTree, RadixTreeNode
from unittest.mock import patch, call


class TestRadixTreeNode(unittest.TestCase):
    def setUp(self):
        self.node = RadixTreeNode()

    def _compare_node_dict(self, node, d):
        self.assertEqual(
            sorted(node.children.keys()), sorted(map(lambda k: k[0], d.keys()))
        )
        # TODO: check the is_end_of_word flag
        for key in d.keys():
            self.assertEqual(node.children[key[0]][0], key)
            self._compare_node_dict(node.children[key[0]][1], d[key])

    def test_simple_insert(self):
        self.node.insert("hello")

        self._compare_node_dict(self.node, {"hello": {}})
        self.assertTrue(self.node.children["h"][1].is_end_of_word)

    def test_insert_with_duplicate(self):
        self.node.insert("hello")
        self.node.insert("hello")
        self._compare_node_dict(self.node, {"hello": {}})
        self.assertTrue(self.node.children["h"][1].is_end_of_word)

    def test_insert_with_overlap(self):
        self.node.insert("overwhelmed")
        self.node.insert("overreached")
        self.assertEqual(list(self.node.children.keys()), ["o"])
        child_word, child_node = self.node.children["o"]
        self.assertEqual(child_word, "over")
        self.assertIsInstance(child_node, RadixTreeNode)
        self.assertFalse(child_node.is_end_of_word)

        self.assertEqual(list(sorted(child_node.children.keys())), ["r", "w"])
        child_a_word, child_a_node = child_node.children["r"]
        self.assertEqual(child_a_word, "reached")
        self.assertIsInstance(child_a_node, RadixTreeNode)
        self.assertTrue(child_a_node.is_end_of_word)
        child_h_word, child_h_node = child_node.children["w"]
        self.assertEqual(child_h_word, "whelmed")
        self.assertIsInstance(child_h_node, RadixTreeNode)
        self.assertTrue(child_h_node.is_end_of_word)

        # TODO: replace check above
        self._compare_node_dict(self.node, {"over": {"reached": {}, "whelmed": {}}})

    def test_insert_with_overlap_and_duplicate(self):
        self.node.insert("overwhelmed")
        self.node.insert("overreached")
        self.node.insert("overwhelmed")  # Insert duplicate

        self.assertEqual(list(self.node.children.keys()), ["o"])
        child_word, child_node = self.node.children["o"]
        self.assertEqual(child_word, "over")
        self.assertIsInstance(child_node, RadixTreeNode)
        self.assertFalse(child_node.is_end_of_word)

        self.assertEqual(list(sorted(child_node.children.keys())), ["r", "w"])
        child_a_word, child_a_node = child_node.children["r"]
        self.assertEqual(child_a_word, "reached")
        self.assertIsInstance(child_a_node, RadixTreeNode)
        self.assertTrue(child_a_node.is_end_of_word)
        child_h_word, child_h_node = child_node.children["w"]
        self.assertEqual(child_h_word, "whelmed")
        self.assertIsInstance(child_h_node, RadixTreeNode)
        self.assertTrue(child_h_node.is_end_of_word)

        self._compare_node_dict(self.node, {"over": {"reached": {}, "whelmed": {}}})

    def test_find_last_full_match_node(self):
        self.node.insert("overwhelmed")
        found_node, prefix_overlap = self.node.find_last_full_match_node("overreached")
        self.assertEqual(prefix_overlap, 4)
        self.assertEqual(list(found_node.children.keys()), ["o"])

        self._compare_node_dict(self.node, {"overwhelmed": {}})


class TestRadixTree(unittest.TestCase):
    def setUp(self):
        self.tree = RadixTree()

    @patch.object(RadixTreeNode, "insert")
    def test_simple_insert(self, mock_insert):
        self.tree.insert("hello")
        mock_insert.assert_called_once_with("hello")

    @patch.object(RadixTreeNode, "insert")
    def test_insert_duplicate(self, mock_insert):
        self.tree.insert("hello")
        self.tree.insert("hello")
        self.assertEqual(mock_insert.call_count, 2)
        mock_insert.assert_has_calls([call("hello"), call("hello")])


if __name__ == "__main__":
    unittest.main()
