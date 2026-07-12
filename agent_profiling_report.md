# Post-Hoc Agent Execution Profiling Report

This report profiles the execution time, step complexity, tool usage, and estimated token consumption for each of the AI optimization sub-agents spawned during this project.

| Iteration | Conversation ID | Runtime | Steps | Tool Calls | Est. Input Tokens | Est. Output Tokens | Total Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`iter_1`** | `1115269d...` | 4m 35s | 145 | 68 | 2,169 | 37,931 | 40,100 |
| **`iter_1_100k`** | `8b6a7690...` | 10m 26s | 60 | 21 | 2,836 | 13,581 | 16,417 |
| **`iter_2`** | `bc76cd92...` | 2m 47s | 118 | 57 | 987 | 29,126 | 30,113 |
| **`iter_2_100k`** | `43c1bd2a...` | 35m 17s | 126 | 47 | 6,895 | 31,307 | 38,202 |
| **`iter_3`** | `1ddb068a...` | 3m 28s | 124 | 60 | 1,887 | 38,315 | 40,202 |
| **`iter_3_100k`** | `e0b76acf...` | 2m 52s | 387 | 155 | 7,339 | 56,586 | 63,925 |
| **`iter_4`** | `81fa6dd3...` | 2m 30s | 85 | 41 | 984 | 28,846 | 29,830 |
| **`iter_4_100k`** | `f67d0471...` | 37m 13s | 93 | 35 | 5,851 | 27,815 | 33,666 |
| **`iter_5`** | `a25ad35f...` | 3m 22s | 157 | 76 | 1,504 | 44,649 | 46,153 |
| **`iter_5_100k`** | `cc9e77a3...` | 8m 57s | 146 | 54 | 8,210 | 41,957 | 50,167 |
| **`iter_6`** | `d0d65228...` | 2m 39s | 115 | 56 | 1,116 | 35,750 | 36,866 |
| **`iter_blank_1`** | `c26187a1...` | 2m 23s | 77 | 37 | 1,583 | 37,701 | 39,284 |
| **`iter_blank_1_100k`** | `9cb8b7f6...` | 17m 49s | 90 | 40 | 3,646 | 37,760 | 41,406 |
| **`iter_variant_1`** | `d20bf807...` | 4m 36s | 131 | 63 | 1,882 | 43,871 | 45,753 |
| **`iter_variant_2`** | `1d546a96...` | 3m 38s | 114 | 53 | 2,415 | 55,682 | 58,097 |
| **`iter_variant_3`** | `b760a6f2...` | 2m 16s | 87 | 42 | 781 | 25,577 | 26,358 |
| **`parent_orchestrator`** | `d1dab082...` | 442m 38s | 253 | 95 | 10,682 | 60,361 | 71,043 |