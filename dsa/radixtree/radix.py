import json

END_KEY = 1


class RadixTreeNode:
    def __init__(self, is_end_of_word=False):
        self.children = {}
        self.is_end_of_word = is_end_of_word

    class Stats:
        def __init__(self, node_count=0, char_count=0, str_count=0, full_char_count=0):
            self.node_count = node_count
            self.char_count = char_count
            self.str_count = str_count
            self.full_char_count = full_char_count

        def __add__(self, other):
            return self.__class__(
                self.node_count + other.node_count,
                self.char_count + other.char_count,
                self.str_count + other.str_count,
                self.full_char_count + other.full_char_count,
            )

        def __str__(self):
            return f"Stats(node_count={self.node_count}, char_count={self.char_count}, str_count={self.str_count}, full_char_count={self.full_char_count})"

    def stats(self) -> Stats:
        s = self.Stats(node_count=1)
        if self.is_end_of_word:
            s.str_count += 1
        for c, (word, node) in self.children.items():
            child_stats = node.stats()
            s += child_stats
            s.char_count += len(word)
            s.full_char_count += len(word) * child_stats.str_count
        return s

    def insert(self, word):
        if len(word) == 0:
            self.is_end_of_word = True
            return
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

    def find_last_full_match_node(self, word) -> tuple["RadixTreeNode", int]:
        if len(word) == 0:
            return self, 0
        start_c = word[0]
        if start_c not in self.children:
            return self, 0
        child_word, child_node = self.children[start_c]
        prefix_overlap = self._prefix_overlap(word, child_word)
        if prefix_overlap != len(child_word):
            return self, prefix_overlap
        else:
            tmp_node, tmp_prefix_overlap = child_node.find_last_full_match_node(
                word[prefix_overlap:]
            )
            return tmp_node, tmp_prefix_overlap + prefix_overlap

    def to_dict(self) -> dict:
        d = {}
        if self.is_end_of_word:
            d[END_KEY] = True
        for c, (word, node) in self.children.items():
            d[word] = node.to_dict()
        return d

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
        node, prefix_overlap = self.root.find_last_full_match_node(word)
        if prefix_overlap == len(word) and node.is_end_of_word:
            return True
        return False

    def insert(self, word):
        self.root.insert(word)

    def _dumb_removal(self, word):
        node, prefix_overlap = self.root.find_last_full_match_node(word[:-1])
        assert prefix_overlap == len(word) - 1
        for c, (word, child) in list(node.children.items()):
            if len(child.children) == 0 and not child.is_end_of_word:
                del node.children[c]

        """
        TODO: there's 2 optimizations we need to to.
        1. get word node's parent without another search / traversal
            - child keep pointer to parent
        2. in the parent, locate the child without for loop
            - find a way to cache the edge to child and reuse it
        """

    def delete(self, word) -> bool:
        node, prefix_overlap = self.root.find_last_full_match_node(word)
        if prefix_overlap != len(word):
            return False
        if not node.is_end_of_word:
            return False
        node.is_end_of_word = False
        # TODO: either do async period cleanup, or impl online node removal
        self._dumb_removal(word)
        return True


if __name__ == "__main__":
    tree = RadixTree()
    while True:
        word = input("Enter a word: ")
        tree.insert(word)
        print(json.dumps(tree.root.to_dict(), indent=4))
        print(tree.root.stats())
