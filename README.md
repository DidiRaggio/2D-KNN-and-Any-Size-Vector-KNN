K Nearest Neighbors algorithm

python_version = 3.6
dependencies = math, heapq, unittest

Both KNN_2D.py and AnySizeKNN.py instanciate a class for their respective solvers with 3 arguments:
	- reference point argument (ref_point) that defaults to (0,0). With which all euclidan distances will be calculated from.
	- K argument (k) that defaults to 2, that defines the amount of nearest neighbors you'd like to return.
	- data set argument (data_set), which is all the data points to be analyzed.
	Note: for KNN_2D.py all datapoints must be of 2 dimensions (example: (3,2)), for AnySizeKNN.py the data points can be any size you want, from 1 to infinity (I wouldn't recomend infinity though, lol)

How to Run:
	1)Open command line.
	2)Run command for script you'd like to run (I recommend the GUI)
		GUI: $python run.py
		2D KNN: $python KNN_2D.py
		Any Size Vector KNN: $python AnySizeVectorKNN.py
		tests: $python test_KNN.py


