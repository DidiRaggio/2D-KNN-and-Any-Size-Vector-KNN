import unittest
from KNN_2D import KNN
from AnySizeVectorKNN import AnySizeVectorKNN

class Test2DKNN(unittest.TestCase):

	#CREATE METHODS TO INITIALIZE TESTING EXAMPLES, BEFORE EACH TEST CASE
	def setUp(self):
		self.data_set = [(-2,4),(0,-2),(-1,0),(3,5),(-2,-3),(3,2)]
		self.k = 3
		self.ref_point = (0,3)
		self.KNN = KNN()


	def test_initial_knn(self):
		knn = self.KNN
		self.assertIsInstance(knn, KNN)
		self.assertIsInstance(knn.k, int)
		self.assertIsInstance(knn.ref_point, tuple)
		self.assertIsInstance(knn.data_set, list)

		# default KNN state
		self.assertEqual(knn.k, 2)
		self.assertEqual(knn.ref_point, (0,0))
		self.assertEqual(knn.data_set, [])
		

		# updating KNN State
		knn.k = self.k
		knn.ref_point = self.ref_point
		knn.data_set = self.data_set
		self.assertEqual(knn.k, 3)
		self.assertEqual(knn.ref_point, (0,3))
		self.assertEqual(knn.data_set, [(-2,4),(0,-2),(-1,0),(3,5),(-2,-3),(3,2)])




	def test_knn_get_distance(self):
		self.assertEqual(self.KNN.getDistance((0,1)), 1)
		self.assertEqual(self.KNN.getDistance((2,4)), 4.47213595499958)
		self.KNN.ref_point=(2,4)
		self.assertEqual(self.KNN.getDistance((2,4)), 0)


	def test_knn_format_data(self):
		initial_formatted_data = self.KNN.formatData()
		self.assertIsInstance(initial_formatted_data, list)
		self.assertEqual(len(initial_formatted_data), 0)
		self.KNN.data_set = self.data_set
		new_formatted_data = self.KNN.formatData()
		self.assertIsInstance(new_formatted_data, list)
		self.assertIsInstance(new_formatted_data[0], dict)
		self.assertEqual(len(new_formatted_data), 6)
		self.assertEqual(new_formatted_data[0], {'coordinates': (-2, 4), 'distance': 4.47213595499958})

		
	def test_knn_solve(self):
		initial_solve = self.KNN.solve()
		self.assertIsInstance(initial_solve, list)
		self.assertEqual(len(initial_solve), 0)
		self.KNN.data_set = self.data_set
		new_solve = self.KNN.solve()
		self.assertIsInstance(new_solve, list)
		self.assertIsInstance(new_solve[0], dict)
		self.KNN.k = self.k
		new_k_solve = self.KNN.solve()
		self.assertEqual(len(new_k_solve), 3)
		self.assertLessEqual(new_k_solve[0]['distance'], new_k_solve[-1]['distance'])




class TestAnySizeVectorKNN(unittest.TestCase):

	#CREATE METHODS TO INITIALIZE TESTING EXAMPLES, BEFORE EACH TEST CASE
	def setUp(self):
		self.data_set = [(-2,4,67,11),(0,-2,9),(20,5,6,11,-1,0),(1,0,0,2,0),(-2,-3),(3), (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1), (1,2,3,4,5,6,7,8,9)]
		self.k = 4
		self.ref_point = (0,0,0,0,0,0,0,0)
		self.KNN = AnySizeVectorKNN()


	def test_any_size_vector_initial_knn(self):
		knn = self.KNN
		self.assertIsInstance(knn, AnySizeVectorKNN)
		self.assertIsInstance(knn.k, int)
		self.assertIsInstance(knn.ref_point, tuple)
		self.assertIsInstance(knn.data_set, list)

		# default KNN state
		self.assertEqual(knn.k, 2)
		self.assertEqual(knn.ref_point, (0,0))
		self.assertEqual(knn.data_set, [])
		

		# updating KNN State
		knn.k = self.k
		knn.ref_point = self.ref_point
		knn.data_set = self.data_set
		self.assertEqual(knn.k, 4)
		self.assertEqual(knn.ref_point, (0,0,0,0,0,0,0,0))
		self.assertEqual(knn.data_set, self.data_set)

	def test_any_size_vector_knn_get_distance(self):
		self.assertEqual(self.KNN.getDistance((0,1,0,0,0)), 1)
		self.assertEqual(self.KNN.getDistance((2,4,1,4,2,1,0,2,9)), 11.269427669584644)
		self.KNN.ref_point=(2,4,0,0,0,0,0)
		self.assertEqual(self.KNN.getDistance((2,4)), 0)


	def test_any_size_vector_knn_format_data(self):
		initial_formatted_data = self.KNN.formatData()
		self.assertIsInstance(initial_formatted_data, list)
		self.assertEqual(len(initial_formatted_data), 0)
		self.KNN.data_set = self.data_set
		new_formatted_data = self.KNN.formatData()
		self.assertIsInstance(new_formatted_data, list)
		self.assertIsInstance(new_formatted_data[0], dict)
		self.assertEqual(len(new_formatted_data), 8)
		self.assertEqual(new_formatted_data[0], {'point': (-2, 4, 67, 11), 'distance': 68.044103344816})

		
	def test_any_size_vector_knn_solve(self):
		initial_solve = self.KNN.solve()
		self.assertIsInstance(initial_solve, list)
		self.assertEqual(len(initial_solve), 0)
		self.KNN.data_set = self.data_set
		new_solve = self.KNN.solve()
		self.assertIsInstance(new_solve, list)
		self.assertIsInstance(new_solve[0], dict)
		self.KNN.k = self.k
		new_k_solve = self.KNN.solve()
		self.assertEqual(len(new_k_solve), 4)
		self.assertLessEqual(new_k_solve[0]['distance'], new_k_solve[-1]['distance'])




if __name__ == '__main__':
	unittest.main()
