import collections
import itertools
from difflib import SequenceMatcher


class TextProcessor:
    def __init__(self):
        self.count_empties = 0
        self.not_recognized = 0
        self.text = []

    def add_with_filter(self, val: str):
        if val == 'Не распознал':
            self.not_recognized += 1
        elif val == 'Пусто':
            self.count_empties += 1
        else:
            self.text.append(val)

    def get_result(self):
        count_values = self.not_recognized + self.count_empties + len(self.text)
        if self.not_recognized == count_values:
            return 'Не распознал'
        elif self.count_empties == count_values:
            return 'Пусто'
        if len(self.text) == 0: raise ValueError("Can't be empty. Check preprocessing")

        if len(self.text) == 1: return self.text[0]

        results = []

        for comparable_element, compare_with in itertools.combinations(self.text, 2):
            subresults = []
            c = SequenceMatcher(None, a=comparable_element, b=compare_with)
            for tag, i1, i2, j1, j2 in c.get_opcodes():
                if tag == 'equal':
                    if comparable_element[i1:i2] != ' ':
                        subresults.append(comparable_element[i1:i2])
            if len(subresults) != 0:
                results.append(''.join(subresults))
        test_2 = collections.Counter(results).most_common(1)
        if len(test_2) == 0:
            return ' '
        else:
            return test_2[0][0]