import unittest
from radix import RadixTree, RadixTreeNode
from unittest.mock import patch, call
from radix import END_KEY

def compare_node_dict(test_cls, node, d):
    test_cls.assertEqual(node.is_end_of_word, d.get(END_KEY, False))
    d = {k: v for k, v in d.items() if k != END_KEY}
    test_cls.assertEqual(
        sorted(node.children.keys()), sorted(map(lambda k: k[0], d.keys()))
    )
    for key in d.keys():
        test_cls.assertEqual(node.children[key[0]][0], key)
        compare_node_dict(test_cls, node.children[key[0]][1], d[key])

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
            compare_node_dict(self, node.children[key[0]][1], d[key])

    def test_simple_insert(self):
        self.node.insert("hello")
        compare_node_dict(self, self.node, {"hello": {END_KEY: True}})

    def test_insert_with_duplicate(self):
        self.node.insert("hello")
        self.node.insert("hello")
        compare_node_dict(self, self.node, {"hello": {END_KEY: True}})

    def test_insert_with_overlap(self):
        self.node.insert("overwhelmed")
        self.node.insert("overreached")
        compare_node_dict(self, self.node, {"over": {"reached": {END_KEY: True}, "whelmed": {END_KEY: True}}})

        self.node.insert("over")
        compare_node_dict(self, self.node, {"over": {END_KEY: True, "reached": {END_KEY: True}, "whelmed": {END_KEY: True}}})

    def test_insert_partial_word(self):
        self.node.insert("overwhelmed")
        compare_node_dict(self, self.node, {"overwhelmed": {END_KEY: True}})

        self.node.insert("over")
        compare_node_dict(self, self.node, {"over": {END_KEY: True, "whelmed": {END_KEY: True}}})

    def test_insert_with_overlap_and_duplicate(self):
        self.node.insert("overwhelmed")
        self.node.insert("overreached")
        self.node.insert("overwhelmed")  # Insert duplicate

        compare_node_dict(self, self.node, {"over": {"reached": {END_KEY: True}, "whelmed": {END_KEY: True}}})

    def test_find_last_full_match_node(self):
        self.node.insert("overwhelmed")

        found_node, prefix_overlap = self.node.find_last_full_match_node("overreached")
        self.assertEqual(prefix_overlap, 4)
        compare_node_dict(self, found_node, {"overwhelmed": {END_KEY: True}})

        found_node, prefix_overlap = self.node.find_last_full_match_node("overwhelmed")
        compare_node_dict(self, found_node, {END_KEY: True})

        self.node.insert("overreached")
        found_node, prefix_overlap = self.node.find_last_full_match_node("overcooked")
        self.assertEqual(prefix_overlap, 4)
        compare_node_dict(self, found_node, {"whelmed": {END_KEY: True}, "reached": {END_KEY: True}})


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

    def test_exist(self):
        self.tree.insert("hello")
        self.assertTrue(self.tree.exist("hello"))
        self.assertFalse(self.tree.exist("hell"))
        self.assertFalse(self.tree.exist("world"))
        self.assertFalse(self.tree.exist("hello world"))
        self.assertFalse(self.tree.exist("world hello"))

        self.tree.insert("overwhelmed")
        self.tree.insert("overreached")
        self.assertTrue(self.tree.exist("overwhelmed"))
        self.assertTrue(self.tree.exist("overreached"))
        self.assertFalse(self.tree.exist(""))
        self.assertFalse(self.tree.exist("over"))
        self.assertFalse(self.tree.exist("overcooked"))
        self.assertFalse(self.tree.exist("overwhelmeded"))
        self.assertFalse(self.tree.exist("overreacheded"))
        self.assertFalse(self.tree.exist("overcookeded"))
        self.assertFalse(self.tree.exist("overwhelmededed"))

    def test_delete(self):
        self.tree.insert("overwhelmed")
        self.tree.insert("overreached")
        self.tree.insert("over")
        compare_node_dict(self, self.tree.root, {"over": {END_KEY: True, "reached": {END_KEY: True}, "whelmed": {END_KEY: True}}})

        # "whelmed" stays until async cleanup.
        self.tree.delete("overwhelmed")
        compare_node_dict(self, self.tree.root, {"over": {END_KEY: True, "reached": {END_KEY: True}, "whelmed": {}}})

        self.tree.insert("overcooked")
        compare_node_dict(self, self.tree.root, {"over": {END_KEY: True, "reached": {END_KEY: True}, "cooked": {END_KEY: True}, "whelmed": {}}})
        
        self.tree.delete("over")
        compare_node_dict(self, self.tree.root, {"over": {"reached": {END_KEY: True}, "cooked": {END_KEY: True}, "whelmed": {}}})



if __name__ == "__main__":
    unittest.main()
