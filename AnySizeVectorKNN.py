import math
import heapq

class AnySizeVectorKNN(object):
	"""Generalized KNN for vectors of any size"""

	#initalize KNN
	def __init__(self, data_set:list = [], k = 2, ref_point = (0,0)):
		super(AnySizeVectorKNN, self).__init__()
		self.ref_point = ref_point
		self.k = k
		self.data_set = data_set

	# calculate eucledian distance from reference point to data point
	def getDistance(self, data_point):
		
		ref_point_len = len(self.ref_point)
		vector_len = ( 1 if isinstance(data_point, int) else len(data_point))
		max_len = (vector_len if vector_len >= ref_point_len else ref_point_len)
		squared_differences = []

		for i in range(max_len):
			try:
				ref_point_value = self.ref_point[i]
			except:
				ref_point_value = 0
			try:
				data_point_value = (data_point[i] if vector_len != 1 else (data_point if i < 1 else 0))
			except:
				data_point_value = 0

			squared_differences.append((ref_point_value - data_point_value)**2)
		distance = math.sqrt(sum(squared_differences))
		return distance

	# format the data set and save into dictionary
	def formatData(self):

		formatted_data = []

		for data_point in self.data_set:
			point_dict = {"point": data_point, "distance": self.getDistance(data_point)}
			formatted_data.append(point_dict)

		return formatted_data

	# solve KNN
	def solve(self):
		formatted_data = self.formatData()
		k_nearest_neighbors = heapq.nsmallest(self.k, formatted_data, key = lambda x: x['distance'])
		return k_nearest_neighbors


if __name__ == "__main__":
	data_set = [(-2,4,67,11),(0,-2,9),(20,5,6,11,-1,0),(1,0,0,2,0),(-2,-3),(3), (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1), (1,2,3,4,5,6,7,8,9)]
	knn = AnySizeVectorKNN(data_set, 4, (0,0,0,0,0,0,0,0))
	print(knn.solve())