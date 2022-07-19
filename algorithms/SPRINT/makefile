predict_interactions_parallel: Src/predict_interactions/predict_interactions.cpp
	g++ -O3 -fopenmp -Wall Src/predict_interactions/predict_interactions.cpp -o bin/predict_interactions -DPARAL
compute_HSPs_parallel: Src/compute_HSPs/compute_hsps.cpp
	g++ -O3 -fopenmp -Wall Src/compute_HSPs/compute_hsps.cpp -o bin/compute_HSPs -DPARAL
predict_interactions_serial: Src/predict_interactions/predict_interactions.cpp
	g++ -O3 -Wall Src/predict_interactions/predict_interactions.cpp -o bin/predict_interactions
compute_HSPs_serial: Src/compute_HSPs/compute_hsps.cpp
	g++ -O3 -Wall Src/compute_HSPs/compute_hsps.cpp -o bin/compute_HSPs
clean:
	rm bin/*

