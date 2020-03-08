CC = g++

all: run

run: run_graph.cpp
	$(CC) -O3 run_graph.cpp -o run_graph -lpoplar && ./run_graph

clean:
	rm -f run_graph
