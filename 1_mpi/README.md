### Smith Waterman Algorithm - Message Passage Interface Version

Here documents the executing time of the serial and parallel codes on the CSE Lab machines.

|                                  | `sample.in`   | `1k.in`     | `20k.in`   |
| -------------------------------- | ------------- | ----------- | ---------- |
| serial                           | 2.146e-5 s    | 0.066572 s  | 10.73292 s |
| parallel `n=1`                   | 3.70974e-5 s  | 0.038292 s  | 15.0195 s  |
| parallel `n=2`                   | 0.000210163 s | 0.0216833 s | 7.87725 s  |
| parallel `n=4`                   | 0.000145245 s | 0.0242907 s | 6.30157 s  |
| parallel `n=8` (across machines) | 0.000503306 s | 0.456867 s  | 32.9077 s  |