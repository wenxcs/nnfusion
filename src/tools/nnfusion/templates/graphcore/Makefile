CC = g++

all: run

run: run_graph.cpp
	$(CC) run_graph.cpp -o run_graph -lpoplar -lpoplin -lpopnn -lpopops && ./run_graph

clean:
	rm -f run_graph
