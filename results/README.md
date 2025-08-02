这里包含了Factorized、Hyperprior、Joint_autogressive的训练结果与compressai中结果的比较。
BD-RATE(训练，论文)性能如下：
```bash
(F)Factorized -18.06
(F1)Factorized1 1.92    #修改了g_a,g_s
(H)Hyperprior -8.38
(J)Joint      -3.79
```
模型之间的比较如下：
```bash
模型      训练      论文
(F1,H)    -17.80    -26.52
(H,J)     -15.16    -10.57
```
