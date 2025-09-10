import unittest
from radix import RadixTree, RadixTreeNode
from unittest.mock import patch, call

END_KEY = 1


class TestRadixTreeNode(unittest.TestCase):
    def setUp(self):
        self.node = RadixTreeNode()

    def _compare_node_dict(self, node, d):
        self.assertEqual(node.is_end_of_word, d.get(END_KEY, False))
        d = {k: v for k, v in d.items() if k != END_KEY}
        self.assertEqual(
            sorted(node.children.keys()), sorted(map(lambda k: k[0], d.keys()))
        )
        for key in d.keys():
            self.assertEqual(node.children[key[0]][0], key)
            self._compare_node_dict(node.children[key[0]][1], d[key])

    def test_simple_insert(self):
        self.node.insert("hello")
        self._compare_node_dict(self.node, {"hello": {END_KEY: True}})

    def test_insert_with_duplicate(self):
        self.node.insert("hello")
        self.node.insert("hello")
        self._compare_node_dict(self.node, {"hello": {END_KEY: True}})

    def test_insert_with_overlap(self):
        self.node.insert("overwhelmed")
        self.node.insert("overreached")
        self._compare_node_dict(
            self.node,
            {"over": {"reached": {END_KEY: True}, "whelmed": {END_KEY: True}}},
        )

        self.node.insert("over")
        self._compare_node_dict(
            self.node,
            {
                "over": {
                    END_KEY: True,
                    "reached": {END_KEY: True},
                    "whelmed": {END_KEY: True},
                }
            },
        )

    def test_insert_partial_word(self):
        self.node.insert("overwhelmed")
        self._compare_node_dict(
            self.node,
            {"overwhelmed": {END_KEY: True}},
        )

        self.node.insert("over")
        self._compare_node_dict(
            self.node,
            {"over": {END_KEY: True, "whelmed": {END_KEY: True}}},
        )

    def test_insert_with_overlap_and_duplicate(self):
        self.node.insert("overwhelmed")
        self.node.insert("overreached")
        self.node.insert("overwhelmed")  # Insert duplicate

        self._compare_node_dict(
            self.node,
            {"over": {"reached": {END_KEY: True}, "whelmed": {END_KEY: True}}},
        )

    def test_find_last_full_match_node(self):
        self.node.insert("overwhelmed")

        found_node, prefix_overlap = self.node.find_last_full_match_node("overreached")
        self.assertEqual(prefix_overlap, 4)
        self._compare_node_dict(found_node, {"overwhelmed": {END_KEY: True}})

        found_node, prefix_overlap = self.node.find_last_full_match_node("overwhelmed")
        self._compare_node_dict(found_node, {END_KEY: True})

        self.node.insert("overreached")
        found_node, prefix_overlap = self.node.find_last_full_match_node("overcooked")
        self.assertEqual(prefix_overlap, 4)
        self._compare_node_dict(
            found_node, {"whelmed": {END_KEY: True}, "reached": {END_KEY: True}}
        )


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
