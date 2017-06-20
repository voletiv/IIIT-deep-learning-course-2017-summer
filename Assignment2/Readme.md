## When trying to run vae_CelebFaces_VGGFaceSmall.py in node10
```
>>> # Fit
... history = vae.fit(trainImages, trainImages, batch_size=minibatchSize, epochs=nEpochs)
Epoch 1/10
2017-06-20 20:25:42.825757: W tensorflow/core/common_runtime/bfc_allocator.cc:273] Allocator (GPU_0_bfc) ran out of memory trying to allocate 286.00MiB.  Current allocation summary follows.
2017-06-20 20:25:42.825808: I tensorflow/core/common_runtime/bfc_allocator.cc:643] Bin (256):   Total Chunks: 0, Chunks in use: 0 0B allocated for chunks. 0B client-requested for chunks. 0B in use in bin. 0B client-requested in use in bin.
2017-06-20 20:25:42.825822: I tensorflow/core/common_runtime/bfc_allocator.cc:643] Bin (512):   Total Chunks: 0, Chunks in use: 0 0B allocated for chunks. 0B client-requested for chunks. 0B in use in bin. 0B client-requested in use in bin.
2017-06-20 20:25:42.825834: I tensorflow/core/common_runtime/bfc_allocator.cc:643] Bin (1024):  Total Chunks: 0, Chunks in use: 0 0B allocated for chunks. 0B client-requested for chunks. 0B in use in bin. 0B client-requested in use in bin.
2017-06-20 20:25:42.825845: I tensorflow/core/common_runtime/bfc_allocator.cc:643] Bin (2048):  Total Chunks: 0, Chunks in use: 0 0B allocated for chunks. 0B client-requested for chunks. 0B in use in bin. 0B client-requested in use in bin.
2017-06-20 20:25:42.825856: I tensorflow/core/common_runtime/bfc_allocator.cc:643] Bin (4096):  Total Chunks: 0, Chunks in use: 0 0B allocated for chunks. 0B client-requested for chunks. 0B in use in bin. 0B client-requested in use in bin.
2017-06-20 20:25:42.825868: I tensorflow/core/common_runtime/bfc_allocator.cc:643] Bin (8192):  Total Chunks: 0, Chunks in use: 0 0B allocated for chunks. 0B client-requested for chunks. 0B in use in bin. 0B client-requested in use in bin.
2017-06-20 20:25:42.825880: I tensorflow/core/common_runtime/bfc_allocator.cc:643] Bin (16384):     Total Chunks: 0, Chunks in use: 0 0B allocated for chunks. 0B client-requested for chunks. 0B in use in bin. 0B client-requested in use in bin.
2017-06-20 20:25:42.825891: I tensorflow/core/common_runtime/bfc_allocator.cc:643] Bin (32768):     Total Chunks: 0, Chunks in use: 0 0B allocated for chunks. 0B client-requested for chunks. 0B in use in bin. 0B client-requested in use in bin.
2017-06-20 20:25:42.825903: I tensorflow/core/common_runtime/bfc_allocator.cc:643] Bin (65536):     Total Chunks: 0, Chunks in use: 0 0B allocated for chunks. 0B client-requested for chunks. 0B in use in bin. 0B client-requested in use in bin.
2017-06-20 20:25:42.825914: I tensorflow/core/common_runtime/bfc_allocator.cc:643] Bin (131072):    Total Chunks: 0, Chunks in use: 0 0B allocated for chunks. 0B client-requested for chunks. 0B in use in bin. 0B client-requested in use in bin.
2017-06-20 20:25:42.825926: I tensorflow/core/common_runtime/bfc_allocator.cc:643] Bin (262144):    Total Chunks: 0, Chunks in use: 0 0B allocated for chunks. 0B client-requested for chunks. 0B in use in bin. 0B client-requested in use in bin.
2017-06-20 20:25:42.825938: I tensorflow/core/common_runtime/bfc_allocator.cc:643] Bin (524288):    Total Chunks: 0, Chunks in use: 0 0B allocated for chunks. 0B client-requested for chunks. 0B in use in bin. 0B client-requested in use in bin.
2017-06-20 20:25:42.825949: I tensorflow/core/common_runtime/bfc_allocator.cc:643] Bin (1048576):   Total Chunks: 0, Chunks in use: 0 0B allocated for chunks. 0B client-requested for chunks. 0B in use in bin. 0B client-requested in use in bin.
2017-06-20 20:25:42.825960: I tensorflow/core/common_runtime/bfc_allocator.cc:643] Bin (2097152):   Total Chunks: 0, Chunks in use: 0 0B allocated for chunks. 0B client-requested for chunks. 0B in use in bin. 0B client-requested in use in bin.
2017-06-20 20:25:42.825971: I tensorflow/core/common_runtime/bfc_allocator.cc:643] Bin (4194304):   Total Chunks: 0, Chunks in use: 0 0B allocated for chunks. 0B client-requested for chunks. 0B in use in bin. 0B client-requested in use in bin.
2017-06-20 20:25:42.825982: I tensorflow/core/common_runtime/bfc_allocator.cc:643] Bin (8388608):   Total Chunks: 0, Chunks in use: 0 0B allocated for chunks. 0B client-requested for chunks. 0B in use in bin. 0B client-requested in use in bin.
2017-06-20 20:25:42.825993: I tensorflow/core/common_runtime/bfc_allocator.cc:643] Bin (16777216):  Total Chunks: 0, Chunks in use: 0 0B allocated for chunks. 0B client-requested for chunks. 0B in use in bin. 0B client-requested in use in bin.
2017-06-20 20:25:42.826004: I tensorflow/core/common_runtime/bfc_allocator.cc:643] Bin (33554432):  Total Chunks: 0, Chunks in use: 0 0B allocated for chunks. 0B client-requested for chunks. 0B in use in bin. 0B client-requested in use in bin.
2017-06-20 20:25:42.826015: I tensorflow/core/common_runtime/bfc_allocator.cc:643] Bin (67108864):  Total Chunks: 0, Chunks in use: 0 0B allocated for chunks. 0B client-requested for chunks. 0B in use in bin. 0B client-requested in use in bin.
2017-06-20 20:25:42.826029: I tensorflow/core/common_runtime/bfc_allocator.cc:643] Bin (134217728):     Total Chunks: 1, Chunks in use: 0 132.16MiB allocated for chunks. 2.0KiB client-requested for chunks. 0B in use in bin. 0B client-requested in use in bin.
2017-06-20 20:25:42.826040: I tensorflow/core/common_runtime/bfc_allocator.cc:643] Bin (268435456):     Total Chunks: 0, Chunks in use: 0 0B allocated for chunks. 0B client-requested for chunks. 0B in use in bin. 0B client-requested in use in bin.
2017-06-20 20:25:42.826052: I tensorflow/core/common_runtime/bfc_allocator.cc:660] Bin for 286.00MiB was 256.00MiB, Chunk State: 
2017-06-20 20:25:42.826064: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e40000 of size 1280
2017-06-20 20:25:42.826073: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e40500 of size 256
2017-06-20 20:25:42.826082: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e40600 of size 256
2017-06-20 20:25:42.826091: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e40700 of size 256
2017-06-20 20:25:42.826099: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e40800 of size 256
2017-06-20 20:25:42.826108: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e40900 of size 256
2017-06-20 20:25:42.826116: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e40a00 of size 256
2017-06-20 20:25:42.826124: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e40b00 of size 256
2017-06-20 20:25:42.826132: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e40c00 of size 256
2017-06-20 20:25:42.826140: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e40d00 of size 256
2017-06-20 20:25:42.826149: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e40e00 of size 256
2017-06-20 20:25:42.826157: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e40f00 of size 256
2017-06-20 20:25:42.826165: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e41000 of size 256
2017-06-20 20:25:42.826173: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e41100 of size 256
2017-06-20 20:25:42.826181: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e41200 of size 256
2017-06-20 20:25:42.826190: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e41300 of size 2048
2017-06-20 20:25:42.826198: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e41b00 of size 256
2017-06-20 20:25:42.826206: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e41c00 of size 4096
2017-06-20 20:25:42.826214: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e42c00 of size 256
2017-06-20 20:25:42.826223: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e42d00 of size 256
2017-06-20 20:25:42.826231: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e42e00 of size 256
2017-06-20 20:25:42.826239: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e42f00 of size 256
2017-06-20 20:25:42.826248: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e43000 of size 256
2017-06-20 20:25:42.826257: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e43100 of size 1024
2017-06-20 20:25:42.826265: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e43500 of size 256
2017-06-20 20:25:42.826274: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e43600 of size 512
2017-06-20 20:25:42.826283: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e43800 of size 256
2017-06-20 20:25:42.826292: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e43900 of size 256
2017-06-20 20:25:42.826300: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e43a00 of size 256
2017-06-20 20:25:42.826308: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e43b00 of size 256
2017-06-20 20:25:42.826316: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e43c00 of size 256
2017-06-20 20:25:42.826324: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e43d00 of size 256
2017-06-20 20:25:42.826332: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e43e00 of size 256
2017-06-20 20:25:42.826340: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e43f00 of size 256
2017-06-20 20:25:42.826348: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e44000 of size 6912
2017-06-20 20:25:42.826357: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e45b00 of size 294912
2017-06-20 20:25:42.826365: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05e8db00 of size 1179648
2017-06-20 20:25:42.826373: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb05fadb00 of size 4718592
2017-06-20 20:25:42.826396: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb0642db00 of size 9437184
2017-06-20 20:25:42.826406: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb06d2db00 of size 4718592
2017-06-20 20:25:42.826414: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb071adb00 of size 1179648
2017-06-20 20:25:42.826423: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb072cdb00 of size 294912
2017-06-20 20:25:42.826431: I tensorflow/core/common_runtime/bfc_allocator.cc:678] Chunk at 0xb07315b00 of size 6912
2017-06-20 20:25:42.826440: I tensorflow/core/common_runtime/bfc_allocator.cc:687] Free at 0xb07317600 of size 138578432
2017-06-20 20:25:42.826447: I tensorflow/core/common_runtime/bfc_allocator.cc:693]      Summary of in-use Chunks by size: 
2017-06-20 20:25:42.826458: I tensorflow/core/common_runtime/bfc_allocator.cc:696] 29 Chunks of size 256 totalling 7.2KiB
2017-06-20 20:25:42.826467: I tensorflow/core/common_runtime/bfc_allocator.cc:696] 1 Chunks of size 512 totalling 512B
2017-06-20 20:25:42.826477: I tensorflow/core/common_runtime/bfc_allocator.cc:696] 1 Chunks of size 1024 totalling 1.0KiB
2017-06-20 20:25:42.826488: I tensorflow/core/common_runtime/bfc_allocator.cc:696] 1 Chunks of size 1280 totalling 1.2KiB
2017-06-20 20:25:42.826497: I tensorflow/core/common_runtime/bfc_allocator.cc:696] 1 Chunks of size 2048 totalling 2.0KiB
2017-06-20 20:25:42.826506: I tensorflow/core/common_runtime/bfc_allocator.cc:696] 1 Chunks of size 4096 totalling 4.0KiB
2017-06-20 20:25:42.826516: I tensorflow/core/common_runtime/bfc_allocator.cc:696] 2 Chunks of size 6912 totalling 13.5KiB
2017-06-20 20:25:42.826526: I tensorflow/core/common_runtime/bfc_allocator.cc:696] 2 Chunks of size 294912 totalling 576.0KiB
2017-06-20 20:25:42.826535: I tensorflow/core/common_runtime/bfc_allocator.cc:696] 2 Chunks of size 1179648 totalling 2.25MiB
2017-06-20 20:25:42.826544: I tensorflow/core/common_runtime/bfc_allocator.cc:696] 2 Chunks of size 4718592 totalling 9.00MiB
2017-06-20 20:25:42.826554: I tensorflow/core/common_runtime/bfc_allocator.cc:696] 1 Chunks of size 9437184 totalling 9.00MiB
2017-06-20 20:25:42.826563: I tensorflow/core/common_runtime/bfc_allocator.cc:700] Sum Total of in-use chunks: 20.84MiB
2017-06-20 20:25:42.826575: I tensorflow/core/common_runtime/bfc_allocator.cc:702] Stats: 
Limit:                   160432128
InUse:                    21853696
MaxInUse:                 51227136
NumAllocs:                     116
MaxAllocSize:             23282432

2017-06-20 20:25:42.826589: W tensorflow/core/common_runtime/bfc_allocator.cc:277] **************______________________________________________________________________________________
2017-06-20 20:25:42.826612: W tensorflow/core/framework/op_kernel.cc:1142] Internal: Dst tensor is not initialized.
2017-06-20 20:25:42.944985: E tensorflow/core/common_runtime/executor.cc:644] Executor failed to create kernel. Internal: Dst tensor is not initialized.
     [[Node: Const_291 = Const[dtype=DT_FLOAT, value=Tensor<type: float shape: [73216,1024] values: [0 0 0]...>, _device="/job:localhost/replica:0/task:0/gpu:0"]()]]
Traceback (most recent call last):
  File "/users/voleti.vikram/.local/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 1039, in _do_call
    return fn(*args)
  File "/users/voleti.vikram/.local/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 1021, in _run_fn
    status, run_metadata)
  File "/usr/lib/python3.5/contextlib.py", line 66, in __exit__
    next(self.gen)
  File "/users/voleti.vikram/.local/lib/python3.5/site-packages/tensorflow/python/framework/errors_impl.py", line 466, in raise_exception_on_not_ok_status
    pywrap_tensorflow.TF_GetCode(status))
tensorflow.python.framework.errors_impl.InternalError: Dst tensor is not initialized.
     [[Node: Const_291 = Const[dtype=DT_FLOAT, value=Tensor<type: float shape: [73216,1024] values: [0 0 0]...>, _device="/job:localhost/replica:0/task:0/gpu:0"]()]]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
  File "/users/voleti.vikram/.local/lib/python3.5/site-packages/keras/engine/training.py", line 1498, in fit
    initial_epoch=initial_epoch)
  File "/users/voleti.vikram/.local/lib/python3.5/site-packages/keras/engine/training.py", line 1152, in _fit_loop
    outs = f(ins_batch)
  File "/users/voleti.vikram/.local/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py", line 2227, in __call__
    session = get_session()
  File "/users/voleti.vikram/.local/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py", line 164, in get_session
    _initialize_variables()
  File "/users/voleti.vikram/.local/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py", line 337, in _initialize_variables
    sess.run(tf.variables_initializer(uninitialized_variables))
  File "/users/voleti.vikram/.local/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 778, in run
    run_metadata_ptr)
  File "/users/voleti.vikram/.local/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 982, in _run
    feed_dict_string, options, run_metadata)
  File "/users/voleti.vikram/.local/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 1032, in _do_run
    target_list, options, run_metadata)
  File "/users/voleti.vikram/.local/lib/python3.5/site-packages/tensorflow/python/client/session.py", line 1052, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.InternalError: Dst tensor is not initialized.
     [[Node: Const_291 = Const[dtype=DT_FLOAT, value=Tensor<type: float shape: [73216,1024] values: [0 0 0]...>, _device="/job:localhost/replica:0/task:0/gpu:0"]()]]

Caused by op 'Const_291', defined at:
  File "<stdin>", line 2, in <module>
  File "/users/voleti.vikram/.local/lib/python3.5/site-packages/keras/engine/training.py", line 1481, in fit
    self._make_train_function()
  File "/users/voleti.vikram/.local/lib/python3.5/site-packages/keras/engine/training.py", line 1013, in _make_train_function
    self.total_loss)
  File "/users/voleti.vikram/.local/lib/python3.5/site-packages/keras/optimizers.py", line 393, in get_updates
    ms = [K.zeros(shape) for shape in shapes]
  File "/users/voleti.vikram/.local/lib/python3.5/site-packages/keras/optimizers.py", line 393, in <listcomp>
    ms = [K.zeros(shape) for shape in shapes]
  File "/users/voleti.vikram/.local/lib/python3.5/site-packages/keras/backend/tensorflow_backend.py", line 561, in zeros
    return variable(tf.constant_initializer(0., dtype=tf_dtype)(shape),
  File "/users/voleti.vikram/.local/lib/python3.5/site-packages/tensorflow/python/ops/init_ops.py", line 162, in __call__
    verify_shape=self.verify_shape)
  File "/users/voleti.vikram/.local/lib/python3.5/site-packages/tensorflow/python/framework/constant_op.py", line 106, in constant
    attrs={"value": tensor_value, "dtype": dtype_value}, name=name).outputs[0]
  File "/users/voleti.vikram/.local/lib/python3.5/site-packages/tensorflow/python/framework/ops.py", line 2336, in create_op
    original_op=self._default_original_op, op_def=op_def)
  File "/users/voleti.vikram/.local/lib/python3.5/site-packages/tensorflow/python/framework/ops.py", line 1228, in __init__
    self._traceback = _extract_stack()

InternalError (see above for traceback): Dst tensor is not initialized.
     [[Node: Const_291 = Const[dtype=DT_FLOAT, value=Tensor<type: float shape: [73216,1024] values: [0 0 0]...>, _device="/job:localhost/replica:0/task:0/gpu:0"]()]]
```
What's the problem???
