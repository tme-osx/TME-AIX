The Model Here is trained with Data Set:
  >>Training Dataset size: 800000
  >>Evaluation Dataset size: 200000

Token Vector:
  >> Map: 100%|██████████| 800000/800000 [10:55<00:00, 1220.84 examples/s]
  >> Map: 100%|██████████| 200000/200000 [02:48<00:00, 1188.67 examples/s]

Training:
  {'loss': 0.3141, 'grad_norm': 0.018248245120048523, 'learning_rate': 4.816244027930908e-05, 'epoch': 0.18}
    7%|▋         | 1000/13605 [1:49:30<22:57:14,  6.56s/it]
  {'loss': 0.0152, 'grad_norm': 0.006048723589628935, 'learning_rate': 4.6324880558618154e-05, 'epoch': 0.37}
   11%|█         | 1500/13605 [2:44:08<22:02:06,  6.55s/it]
  {'loss': 0.0147, 'grad_norm': 0.005775829777121544, 'learning_rate': 4.448732083792723e-05, 'epoch': 0.55}
   15%|█▍        | 2000/13605 [3:38:47<21:07:24,  6.55s/it]
  {'loss': 0.0145, 'grad_norm': 0.004081073682755232, 'learning_rate': 4.264976111723631e-05, 'epoch': 0.73}                                                       
  ...
  ...                                                 
   88%|████████▊ | 12000/13605 [27:54:35<3:02:12,  6.81s/it]
  {'eval_loss': nan, 'eval_runtime': 3468.047, 'eval_samples_per_second': 57.669, 'eval_steps_per_second': 7.209, 'epoch': 4.41}
  /Users/fenar/Library/Python/3.9/lib/python/site-packages/huggingface_hub/file_download.py:1132: 
   92%|█████████▏| 12500/13605 [28:51:24<2:06:45,  6.88s/it]    
  {'loss': 0.0, 'grad_norm': nan, 'learning_rate': 4.061006982726939e-06, 'epoch': 4.59}
   96%|█████████▌| 13000/13605 [29:48:11<1:08:41,  6.81s/it]
  {'loss': 0.0, 'grad_norm': nan, 'learning_rate': 2.223447262036016e-06, 'epoch': 4.78}
   99%|█████████▉| 13500/13605 [30:44:59<11:56,  6.82s/it]  
  {'loss': 0.0, 'grad_norm': nan, 'learning_rate': 3.858875413450937e-07, 'epoch': 4.96}
  100%|██████████| 13605/13605 [30:56:55<00:00,  8.19s/it]
  {'train_runtime': 111415.7272, 'train_samples_per_second': 35.902, 'train_steps_per_second': 0.122, 'train_loss': 0.02188537514240415, 'epoch': 5.0}

Evaluation:
  100%|██████████| 25000/25000 [58:03<00:00,  7.18it/s]  
  Evaluation Results: {'eval_loss': 0.013824737630784512, 'eval_runtime': 3483.8647, 'eval_samples_per_second': 57.408, 'eval_steps_per_second': 7.176, 'epoch': 4.999737505249895}
