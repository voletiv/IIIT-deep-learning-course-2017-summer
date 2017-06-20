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
```
What's the problem???
