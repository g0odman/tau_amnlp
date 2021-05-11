Student 1:
* Name: Uri Goodman 
* ID: 315554907
* Username: urigoodman

Now, for each log file that you need to submit, you will need to write its last 3 lines. For example, this is what we got for `baseline_gen.log`:
```txt
2021-04-24 17:20:46 | INFO | fairseq_cli.generate | NOTE: hypothesis and token scores are output in base 2
2021-04-24 17:20:46 | INFO | fairseq_cli.generate | Translated 7,283 sentences (165,025 tokens) in 18.5s (394.00 sentences/s, 8927.61 tokens/s)
Generate valid with beam=5: BLEU4 = 33.39, 69.1/42.8/28.5/19.4 (BP=0.934, ratio=0.937, syslen=138824, reflen=148229)
```

3 last lines from the baseline_train.log file: 
```txt
2021-05-07 16:11:09 | INFO | fairseq_cli.train | end of epoch 50 (average epoch stats below)
2021-05-07 16:11:09 | INFO | train | epoch 050 | loss 3.942 | nll_loss 2.512 | ppl 5.71 | wps 34422 | ups 3.3 | wpb 10419.8 | bsz 422.8 | num_updates 18950 | lr 0.000229718 | gnorm 0.619 | train_wall 66 | gb_free 8.9 | wall 5740
2021-05-07 16:11:09 | INFO | fairseq_cli.train | done training in 5739.2 seconds
```

3 last lines from the baseline_gen.log file: 
```txt
2021-05-07 16:12:41 | INFO | fairseq_cli.generate | NOTE: hypothesis and token scores are output in base 2
2021-05-07 16:12:41 | INFO | fairseq_cli.generate | Translated 7,283 sentences (164,445 tokens) in 26.7s (273.28 sentences/s, 6170.46 tokens/s)
Generate valid with beam=5: BLEU4 = 33.27, 69.1/42.8/28.4/19.3 (BP=0.932, ratio=0.934, syslen=138408, reflen=148229)
```

3 last lines from the baseline_mask.log file: 
```txt
2021-05-09 22:58:52 | INFO | fairseq_cli.generate | NOTE: hypothesis and token scores are output in base 2
2021-05-09 22:58:52 | INFO | fairseq_cli.generate | Translated 7,283 sentences (169,791 tokens) in 24.0s (303.43 sentences/s, 7074.04 tokens/s)
Generate valid with beam=5: BLEU4 = 31.77, 65.2/39.4/25.6/17.1 (BP=0.977, ratio=0.977, syslen=144845, reflen=148229)
```

25 last lines from the check_all_masking_options.log file: 
```txt
2021-05-09 23:44:58 | INFO | fairseq.tasks.translation | data-bin/iwslt14.tokenized.de-en valid de-en 7283 examples
2021-05-09 23:45:52 | INFO | fairseq_cli.generate | NOTE: hypothesis and token scores are output in base 2
2021-05-09 23:45:52 | INFO | fairseq_cli.generate | Translated 7,283 sentences (164,950 tokens) in 21.4s (340.38 sentences/s, 7709.17 tokens/s)
Generate valid with beam=5: BLEU4 = 33.18, 68.9/42.6/28.2/19.2 (BP=0.935, ratio=0.937, syslen=138914, reflen=148229)
table of score with masking enc-enc attention head
rows are transformer layer number and columns are head number
       0      1      2      3
0  33.07  33.07  33.02  33.11
1  33.14  33.12  33.07  33.05
2  32.77  32.76  32.80  32.76
3  32.90  32.97  32.89  32.83
table of score with masking enc-dec attention head
rows are transformer layer number and columns are head number
       0      1      2      3
0  31.91  31.78  31.76  31.87
1  32.76  32.80  32.78  32.86
2  32.77  32.86  32.66  32.83
3  31.77  32.03  31.71  32.01
table of score with masking dec-dec attention head
rows are transformer layer number and columns are head number
       0      1      2      3
0  33.22  33.14  33.17  33.13
1  33.18  33.14  33.12  33.21
2  33.23  33.30  33.28  33.17
3  33.22  33.21  33.20  33.18
```

3 last lines from the sandwich_train.log file: 
```txt
2021-05-10 23:01:43 | INFO | fairseq_cli.train | end of epoch 50 (average epoch stats below)
2021-05-10 23:01:43 | INFO | train | epoch 050 | loss 3.969 | nll_loss 2.543 | ppl 5.83 | wps 35471.3 | ups 3.4 | wpb 10419.8 | bsz 422.8 | num_updates 18950 | lr 0.000229718 | gnorm 0.623 | train_wall 65 | gb_free 8.9 | wall 5666
2021-05-10 23:01:43 | INFO | fairseq_cli.train | done training in 5665.3 seconds
```

3 last lines from the sandwich_gen.log file: 
```txt
2021-05-10 23:16:14 | INFO | fairseq_cli.generate | NOTE: hypothesis and token scores are output in base 2
2021-05-10 23:16:14 | INFO | fairseq_cli.generate | Translated 7,283 sentences (167,628 tokens) in 19.5s (373.51 sentences/s, 8596.82 tokens/s)
Generate valid with beam=5: BLEU4 = 33.15, 68.2/41.9/27.7/18.8 (BP=0.950, ratio=0.951, syslen=140940, reflen=148229)
```