# See README.txt.

.PHONY: all python clean

python: protoc_middleman

clean:
	rm -f graph_def_pb2.py node_def_pb2.py attr_value_pb2.py tensor_shape_pb2.py
	rm -f *.pyc

protoc_middleman: ../../core/graph/graph_def.proto ../../core/graph/node_def.proto ../../core/graph/tensor_shape.proto ../../core/graph/attr_value.proto
	protoc $$PROTO_PATH --python_out=. --proto_path ../../core/graph graph_def.proto 
	protoc $$PROTO_PATH --python_out=. --proto_path ../../core/graph node_def.proto
	protoc $$PROTO_PATH --python_out=. --proto_path ../../core/graph tensor_shape.proto
	protoc $$PROTO_PATH --python_out=. --proto_path ../../core/graph attr_value.proto
