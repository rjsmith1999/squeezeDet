
Freeze graph:
```python2 ./freeze_graph.py --input_graph ./squeezeDet.pb --input_checkpoint ../data/model_checkpoints/squeezeDet/model.ckpt-87000 --input_meta_graph ../data/model_checkpoints/squeezeDet/model.ckpt-87000.meta  --output_graph=./frozen_graph.pb --output_node_names=bbox/trimming/bbox,probability/score,probability/class_idx --input_binary true```

After converting output .pb to binary with 
```python2 ./convert_to_binary.py --input_graph ./squeezeDet.pbtxt```

view in tensorboard with
```python2 import_pb_to_tensorboard.py --model_dir ./squeezeDet.pb  --log_dir ./logs/```

Freeze graph drops all the extra training data stuff in addition to setting stuff to constants

Get .pbtxt with
```python2 ./src/write_graph.py --out_dir ./coreml/```