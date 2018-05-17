import tfcoreml as tf_converter
from argparse import ArgumentParser

def replace_extension(file, new_ext):
  """
  Replaces extension with 'new_ext"
  """

  #drop ext
  dropped = '.'.join(file.split('.')[0:-1])
  
  return dropped + new_ext


def main():
  parser = ArgumentParser()
  parser.add_argument(
    "--input_graph",
    type=str,
    default="",
    help="TensorFlow .pbtxt file to convert."
  )
  input_graph = parser.parse_args().input_graph
  tf_converter.convert(tf_model_path = input_graph,
      mlmodel_path = replace_extension(input_graph, '.mlmodel'),
      output_feature_names = ['bbox/trimming/bbox','probability/score','probability/class_idx'])


if __name__ == '__main__':
  main()