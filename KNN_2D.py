import math
import heapq

class KNN(object):
	"""Simple 2D KNN"""

	# initalize KNN
	def __init__(self, data_set:list = [], k = 2, ref_point = (0,0)):
		super(KNN, self).__init__()
		self.ref_point = ref_point
		self.k = k
		self.data_set = data_set

	# calculate eucledian distance from reference point to data point
	def getDistance(self, data_point):
		distance = math.sqrt((self.ref_point[0]-data_point[0])**2 + (self.ref_point[1]-data_point[1])**2)
		return distance

	# format the data set and save into dictionary
	def formatData(self):

		formatted_data = []

		for data_point in self.data_set:
			point_dict = {"coordinates": data_point, "distance": self.getDistance(data_point)}
			formatted_data.append(point_dict)

		return formatted_data

	# solve KNN
	def solve(self):
		formatted_data = self.formatData()
		k_nearest_neighbors = heapq.nsmallest(self.k, formatted_data, key = lambda x: x['distance'])
		return k_nearest_neighbors

if __name__ == "__main__":
	data_set = [(-2,4),(0,-2),(-1,0),(3,5),(-2,-3),(3,2)]
	knn = KNN(data_set, 3, (0,0))
	print(knn.solve())

