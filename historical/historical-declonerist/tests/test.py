import os
import shutil
import random
import unittest
from decl import *

CONTENT_STRING = 'Will iPad mini with retina display be released in WWDC 2013?'


ALPHABET = 'abcdefghijklmnopqrstuvwxyz'


def random_string(l):
    return ''.join([random.choice(ALPHABET) for i in range(l)])


class TestDetect(unittest.TestCase):

    def setUp(self):
        os.mkdir('./files')

    def tearDown(self):
        shutil.rmtree('./files')

    def test_empty_folder(self):
        ret = detect('files')
        self.assertEqual(len(ret), 0)

    def test_duplicate(self):
        with open('./files/a', 'w') as fa:
            with open('./files/b', 'w') as fb:
                fa.write(CONTENT_STRING)
                fb.write(CONTENT_STRING)

        ret = detect('./files')
        self.assertEqual(len(ret), 1)

    def test_same_size_different_content(self):
        with open('./files/a', 'w') as fa:
            with open('./files/b', 'w') as fb:
                fa.write(CONTENT_STRING + 'a' * 42)
                fb.write(CONTENT_STRING + 'b' * 42)

        ret = detect('./files')
        self.assertEqual(len(ret), 0)


if __name__ == '__main__':
    unittest.main()
