# Cok-Seed Kalman Ablasyon Sonuclari

Seed'ler: [42, 7, 123] | CMA seed (sabit): 42

## Tum Kosular

| config   |   seed |   avg_f1 |   avg_acc |   total_profit |   avg_calmar |
|:---------|-------:|---------:|----------:|---------------:|-------------:|
| kalman   |     42 |    18.69 |     54.60 |       19388.40 |       200.76 |
| kalman   |      7 |    28.33 |     54.35 |       24244.47 |       350.97 |
| kalman   |    123 |    26.17 |     54.21 |       19347.68 |       379.06 |
| nokalman |     42 |    31.02 |     53.12 |       17440.17 |       377.27 |
| nokalman |      7 |    20.00 |     54.40 |       31924.59 |       738.64 |
| nokalman |    123 |    25.49 |     53.83 |       30325.74 |       485.51 |

## Seed'ler Arasi Ozet (ortalama +/- std)

| config   |   f1_mean |   f1_std |   acc_mean |   profit_mean |   profit_std |   calmar_mean |   calmar_std |
|:---------|----------:|---------:|-----------:|--------------:|-------------:|--------------:|-------------:|
| kalman   |     24.40 |     5.06 |      54.39 |      20993.52 |      2815.48 |        310.26 |        95.87 |
| nokalman |     25.50 |     5.51 |      53.79 |      26563.50 |      7941.38 |        533.81 |       185.46 |
