import tensorflow as tf
from google.protobuf import text_format
from argparse import ArgumentParser

def convert_pbtxt_to_graphdef(filename):
  """Returns a `tf.GraphDef` proto representing the data in the given pbtxt file.

  Args:
    filename: The name of a file containing a GraphDef pbtxt (text-formatted
      `tf.GraphDef` protocol buffer data).

  Returns:
    A `tf.GraphDef` protocol buffer.
  """
  with tf.gfile.FastGFile(filename, 'r') as f:
    graph_def = tf.GraphDef()

    file_content = f.read()

    # Merges the human-readable string in `file_content` into `graph_def`.
    text_format.Merge(file_content, graph_def)
  return graph_def

def replace_extension(file, new_ext):
  """
  Replaces extension with 'new_ext"
  """

  #drop ext
  dropped = '.'.join(file.split('.')[0:-1])
  
  return dropped + new_ext

def write_graphdef(proto_graph, output_file):
  """
  Writes graphdef file to binary file
  """
  with open(output_file, "wb") as f:
    f.write(proto_graph.SerializeToString())

def convert_pbtxt_to_pb(input_graph):
  """
  converts human readable pbtxt to pb binary
  """
  proto_graph = convert_pbtxt_to_graphdef(input_graph)
  out_file = replace_extension(input_graph, '.pb')
  write_graphdef(proto_graph, out_file)

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument(
    "--input_graph",
    type=str,
    default="",
    help="TensorFlow .pbtxt file to convert."
  )
  input_graph = parser.parse_args().input_graph

  convert_pbtxt_to_pb(input_graph)