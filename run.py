from KNN_2D import KNN 
from AnySizeVectorKNN import AnySizeVectorKNN
import ast

if __name__ == "__main__":

	KNN_type_not_valid = True
	while KNN_type_not_valid:
		type_of_KNN = input("Would you like to use the 2D KNN or the Any Size Vector KNN (type '2D' or 'Any size vector'): ")
		
		if type_of_KNN == 'Any size vector':
			knn = AnySizeVectorKNN()
			KNN_type_not_valid = False
		elif type_of_KNN == '2D':
			knn = KNN()
			KNN_type_not_valid = False
		else:
			print("You did not select a valid KNN type")

	ref_point = ast.literal_eval(input("Please input Reference Point (for example: (0,0) ): ").strip())
	k = int(input("Please input K value (default=2): "))
	data_set = ast.literal_eval(input("Please input a data_set (must be a list of tuples, for example [(1,-2),(0,9)]: ").strip())
	knn.data_set = data_set
	knn.k = k
	knn.ref_point = ref_point
	print(knn.solve())

