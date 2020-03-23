### Smith Waterman Algorithm - Message Passage Interface Version

Here documents the executing time of the serial and parallel codes on the CSE Lab machines.

|                                  | `sample.in`   | `1k.in`     | `20k.in`   |
| -------------------------------- | ------------- | ----------- | ---------- |
| serial                           | 2.146e-5 s    | 0.066572 s  | 10.73292 s |
| parallel `n=1`                   | 3.5065e-5 s   | 0.0351034 s | 13.7656 s  |
| parallel `n=2`                   | 5.6683e-5 s   | 0.0192342 s | 7.16812 s  |
| parallel `n=4`                   | 0.000215836 s | 0.0249722 s | 5.80054 s  |
| parallel `n=8` (across machines) | 0.000824453 s | 0.456867 s  | 32.9077 s  |
