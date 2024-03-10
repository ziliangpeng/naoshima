from python import Python

fn use_array() raises:
    let t = Python.import_module("tinygrad")

try:
    use_array()
except:
    print(1)


