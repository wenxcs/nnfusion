// tool to generate optimized code for a input model with given backend.
// compile and run with:
// g++ ./nnfusion.cpp -std=c++11 -I$HOME/ngraph_dist/include -L$HOME/ngraph_dist/lib -lngraph -o nnfusion
// env LD_LIBRARY_PATH=$HOME/ngraph_dist/lib ./nnfusion

#include <fstream>
#include <iomanip>

#include "ngraph/except.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/frontend/onnx_import/onnx.hpp"
#include "ngraph/frontend/tensorflow_import/tensorflow.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"

#include "nnfusion/core/graph/graph.hpp"

using namespace std;
using namespace ngraph;

element::Type get_op_element_type(const Node& op)
{
    element::Type type;
    if (op.description() == "Convert")
    {
        type = op.get_input_element_type(0);
    }
    else if (op.description() == "Equal" || op.description() == "Greater" ||
             op.description() == "GreaterEq" || op.description() == "Less" ||
             op.description() == "LessEq" || op.description() == "NotEqual")
    {
        // Get the type of the second input, not the first
        // All BinaryElementwiseComparision ops have the same type for inputs
        // Select has bool for first input and the type we are interested in for the second
        type = op.get_input_element_type(1);
    }
    else
    {
        type = op.get_outputs().at(0).get_element_type();
    }
    return type;
}

void display_help()
{
    cout << R"###(
DESCRIPTION
    Generate optimized code for ngraph json model with given backend.

SYNOPSIS
        nnfusion <filename> [-f <format>] [-b <backend>] [-m <function>] [-s] [-v]

OPTIONS
        -f|--format               Model file format (ngraph(default) or tensorflow)
        -b|--backend              Backend to use (CPU(default), GPU, CUDA_CODEGEN[:naive_graphtest])
        -m|--model_format         Import tensorflow model as function(default), graph
        -s|--statistics           Display op stastics
        -v|--visualize            Visualize a model (WARNING: requires GraphViz installed)
)###";
}

int main(int argc, char** argv)
{
    string model;
    string format = "ngraph";
    string backend = "CPU";
    string model_format = "function";
    bool failed = false;
    bool statistics = false;
    bool visualize = false;

    if (argc > 1)
    {
        model = argv[1];
    }
    else
    {
        display_help();
        return 1;
    }
    for (size_t i = 2; i < argc; i++)
    {
        string arg = argv[i];
        if (arg == "-f" || arg == "--format")
        {
            format = argv[++i];
        }
        else if (arg == "-b" || arg == "--backend")
        {
            backend = argv[++i];
        }
        else if (arg == "-m" || arg == "--model_format")
        {
            model_format = argv[++i];
        }
        else if (arg == "-s" || arg == "--statistics")
        {
            statistics = true;
        }
        else if (arg == "-v" || arg == "--visualize")
        {
            visualize = true;
        }
        else
        {
            cout << "Unknown option: " << arg << endl;
            failed = true;
        }
    }
    if (!model.empty() && !file_util::exists(model))
    {
        cout << "File " << model << " not found\n";
        failed = true;
    }

    if (failed)
    {
        display_help();
        return 1;
    }

    cout << "\n";
    cout << "============================================================================\n";
    cout << "---- Processing '" << model << "'\n";
    cout << "============================================================================\n";
    try
    {
        shared_ptr<Function> f = nullptr;
        shared_ptr<nnfusion::graph::Graph> graph = nullptr;
        if (format == "ngraph")
        {
            f = deserialize(model);
        }
        else if (format == "tensorflow")
        {
            std::vector<std::shared_ptr<Function>> functions =
                ngraph::frontend::load_tensorflow_model(model);
            // TODO(jxue): currently we only use the first output function, need to support compile
            // multiple output functions in the future
            f = functions.front();

            // load tensorlfow model as graph
            graph = ngraph::frontend::load_tensorflow_model_as_graph(model);
        }
        else if (format == "onnx")
        {
            f = onnx_import::import_onnx_function(model);
        }
        else
        {
            throw ngraph_error("Unsupported model format '" + format + "' in NNFusion");
        }

        if (visualize)
        {
            auto model_file_name = ngraph::file_util::get_file_name(model) + std::string(".") +
                                   pass::VisualizeTree::get_file_ext();

            pass::Manager pass_manager;
            pass_manager.register_pass<pass::VisualizeTree>(model_file_name);
            pass_manager.run_passes(f);
        }

        if (statistics)
        {
            cout << "\n---- Source Graph Statistics ----\n";
            cout << "Total nodes: " << f->get_ops().size() << endl;
            size_t total_constant_bytes = 0;
            unordered_map<string, size_t> op_list;
            set<string> type_list;
            for (shared_ptr<Node> node : f->get_ordered_ops())
            {
                string name = node->get_name();
                string op_name = name.substr(0, name.find('_'));
                string shape_name = "{" + join(node->get_outputs()[0].get_shape()) + "}";
                op_list[op_name + shape_name]++;
                auto et = get_op_element_type(*node);
                string type_string = et.c_type_string();
                type_list.insert(type_string);

                if (op_name == "Constant")
                {
                    const Shape& shape = node->get_outputs()[0].get_shape();
                    size_t const_size = node->get_outputs()[0].get_element_type().size();
                    if (shape.size() == 0)
                    {
                        total_constant_bytes += const_size;
                    }
                    else
                    {
                        total_constant_bytes +=
                            (const_size * shape_size(node->get_outputs()[0].get_shape()));
                    }
                }
            }
            cout << "--\n";
            cout << "Total Constant size: " << total_constant_bytes << " bytes\n";
            cout << "--\n";
            cout << "Types used:\n";
            for (const string& type : type_list)
            {
                cout << "    " << type << "\n";
            }
            cout << "--\n";
            for (const pair<string, size_t>& op_info : op_list)
            {
                cout << op_info.first << ": " << op_info.second << " ops" << endl;
            }
        }

        if (!backend.empty())
        {
            auto runtime = runtime::Backend::create(backend);
            if (model_format == "function")
            {
                runtime->codegen(f);
            }
            else if (model_format == "graph")
            {
                runtime->codegen(graph);
            }
        }
    }
    catch (ngraph::unsupported_op& ue)
    {
        cout << "Unsupported op '" << ue.what() << "' in model " << model << endl;
    }
    catch (exception& e)
    {
        cout << "Exception caught on '" << model << "'\n" << e.what() << endl;
    }

    return 0;
}
