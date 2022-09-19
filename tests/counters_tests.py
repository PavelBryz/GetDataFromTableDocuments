import unittest
from classes.сounter import Counter


class MyTestCase(unittest.TestCase):
    def test_init(self):
        box = [[5, 5],
               [15, 4],
               [17, 21],
               [6, 22]]
        count = Counter(box)
        self.assertEqual(count.top_left, (5, 5))
        self.assertEqual(count.top_right, (15, 4))
        self.assertEqual(count.down_left, (6, 22))
        self.assertEqual(count.down_right, (17, 21))

    def test_get_wight(self):
        box = [[5, 5],
               [15, 4],
               [17, 21],
               [6, 22]]
        count = Counter(box)
        self.assertEqual(count.get_wight(), 12)

    def test_get_height(self):
        box = [[5, 5],
               [15, 4],
               [17, 21],
               [6, 22]]
        count = Counter(box)
        self.assertEqual(count.get_height(), 18)


if __name__ == '__main__':
    unittest.main()
