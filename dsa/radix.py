class RadixTreeNode:
    def __init__(self, is_end_of_word=False):
        self.children = {}
        self.is_end_of_word = is_end_of_word

    def insert(self, word):
        assert len(word) > 0
        start_c = word[0]
        if start_c not in self.children:
            self.children[start_c] = (word, RadixTreeNode(True))
        else:
            child_word, child_node = self.children[start_c]
            prefix_overlap = self._prefix_overlap(word, child_word)
            if prefix_overlap == len(child_word):
                if len(word) == prefix_overlap:
                    child_node.is_end_of_word = True
                else:
                    child_node.insert(word[prefix_overlap:])
            else:
                existing_word_suffix = child_word[prefix_overlap:]
                new_word_suffix = word[prefix_overlap:]
                common_prefix = child_word[:prefix_overlap]
                new_intermediate_node = RadixTreeNode()
                self.children[start_c] = (common_prefix, new_intermediate_node)
                new_intermediate_node.children[existing_word_suffix[0]] = (
                    existing_word_suffix,
                    child_node,
                )
                new_intermediate_node.insert(new_word_suffix)

    def find_matching_node(self, word) -> tuple["RadixTreeNode", int]:
        start_c = word[0]
        if start_c not in self.children:
            return None, 0
        child_word, child_node = self.children[start_c]
        prefix_overlap = self._prefix_overlap(word, child_word)
        return child_node, prefix_overlap

    def _prefix_overlap(self, str1, str2) -> int:
        l = min(len(str1), len(str2))
        for i in range(l):
            if str1[i] != str2[i]:
                return i
        return l


class RadixTree:
    def __init__(self):
        self.root = RadixTreeNode()

    def exist(self, word) -> bool:
        node = self.root
        while True:
            child_node, prefix_overlap = node.find_matching_node(word)

    def insert(self, word):
        node = self.root

    def delete(self, word):
        raise NotImplementedError("Not implemented")
