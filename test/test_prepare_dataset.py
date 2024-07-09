import unittest
from rag_math_solver.data.prepare_dataset import extract_dataset, transform_dataset, load_dataset

class TestPrepareDataset(unittest.TestCase):

    def test_extract_dataset(self):
        dataset = extract_dataset()
        self.assertTrue(len(dataset) > 0, "Dataset should contain examples")

    def test_transform_dataset(self):
        dummy_dataset = [{'question': '  Solve for x: 3x + 5 = 14  ', 'answer': ' 3 '}]
        transformed_dataset = transform_dataset(dummy_dataset)
        self.assertEqual(transformed_dataset[0]['question'], 'Solve for x: 3x + 5 = 14', "Question should be cleaned")
        self.assertEqual(transformed_dataset[0]['answer'], '3', "Answer should be cleaned")

    def test_load_dataset(self):
        train_dataset, eval_dataset = load_dataset(test_size=0.2, seed=42)
        self.assertTrue(len(train_dataset) > 0, "Train dataset should not be empty")
        self.assertTrue(len(eval_dataset) > 0, "Evaluation dataset should not be empty")

if __name__ == '__main__':
    unittest.main()
