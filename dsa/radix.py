END_KEY = 1

class RadixTreeNode:
    def __init__(self, is_end_of_word=False):
        self.children = {}
        self.is_end_of_word = is_end_of_word

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

if __name__ == "__main__":
    tree = RadixTree()
    while True:
        word = input("Enter a word: ")
        tree.insert(word)
        import json
        print(json.dumps(tree.root.to_dict(), indent=4))
