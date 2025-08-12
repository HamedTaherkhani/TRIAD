

import unittest

class TestSearchSingleVector(unittest.TestCase):
    def setUp(self):
        self.i = 5
        self.P = [[1, 2, 3]]

    def test_single_vector_padding(self):
        result = search(self.i, self.P)
        self.assertEqual(result, [[1, 2, 3, 0, 0]])

import unittest

class TestSearchExactFit(unittest.TestCase):
    def setUp(self):
        self.i = 3
        self.P = [[1, 2, 3], [4, 5, 6]]

    def test_exact_fit_no_padding(self):
        result = search(self.i, self.P)
        self.assertEqual(result, [[1, 2, 3], [4, 5, 6]])

import unittest

class TestSearchWithSeparators(unittest.TestCase):
    def setUp(self):
        self.i = 6
        self.P = [[1, 2, 3], [4, 5, 6], [7], [8]]

    def test_separation_and_packing(self):
        result = search(self.i, self.P)
        expected = [
            [1, 2, 3, 0, 7, 0],
            [4, 5, 6, 0, 8, 0]
        ]
        self.assertEqual(result, expected)

import unittest

class TestSearchVectorLargerThanC(unittest.TestCase):
    def setUp(self):
        self.i = 4
        self.P = [[1, 2, 3, 4, 5]]

    def test_vector_too_large_returns_empty(self):
        result = search(self.i, self.P)
        self.assertEqual(result, [])

import unittest

class TestSearchMultipleSmallVectors(unittest.TestCase):
    def setUp(self):
        self.i = 5
        self.P = [[1], [2], [3], [4], [5]]

    def test_all_fit_in_one_row(self):
        result = search(self.i, self.P)
        self.assertEqual(result, [[1, 0, 2, 0, 3], [4, 0, 5, 0, 0]])

import unittest

class TestSearchEmptyList(unittest.TestCase):
    def setUp(self):
        self.i = 3
        self.P = []

    def test_empty_list_returns_empty(self):
        result = search(self.i, self.P)
        self.assertEqual(result, [])

import unittest

class TestSearchExactFillLastRow(unittest.TestCase):
    def setUp(self):
        self.i = 6
        self.P = [[1, 2], [3, 4, 5], [6]]

    def test_last_row_exact_fill(self):
        result = search(self.i, self.P)
        expected = [
            [1, 2, 0, 3, 4, 5],
            [6, 0, 0, 0, 0, 0]
        ]
        self.assertEqual(result, expected)

import unittest

class TestSearchLexicographicalPreference(unittest.TestCase):
    def setUp(self):
        self.i = 6
        self.P = [[1, 2, 3], [4], [5, 6], [7]]

    def test_prefers_earlier_placement(self):
        result = search(self.i, self.P)
        expected = [
            [1, 2, 3, 0, 4, 0],
            [5, 6, 0, 7, 0, 0]
        ]
        self.assertEqual(result, expected)
