| Model | Test, 200 shared questions | Paraphrase, 108 questions | Valid SQL (paraphrase) | Test, all 1,172 |
|---|---|---|---|---|
| From scratch (852k parameters, trained from zero) | 98.5% (95.7%-99.5%) | 38.0% (29.4%-47.4%) | 97.2% | 98.0% (97.0%-98.6%) |
| LoRA Qwen2.5-0.5B (8.8M trained, 494M total) | 97.5% (94.3%-98.9%) | 85.2% (77.3%-90.7%) | 100.0% | 97.6% (96.6%-98.3%) |
| API gpt-oss-120b (120B, 5 examples in the prompt, no training) | 92.5% (88.0%-95.4%) | 93.5% (87.2%-96.8%) | 100.0% | not run |

Pairs compared on identical questions (McNemar exact test):

- From scratch vs LoRA Qwen2.5-0.5B on test (200): only the first right 5, only the second right 3, p = 0.727 (no clear difference)
- From scratch vs LoRA Qwen2.5-0.5B on paraphrase (108): only the first right 1, only the second right 52, p = 1.2e-14 (clear difference)
- From scratch vs API gpt-oss-120b on test (200): only the first right 13, only the second right 1, p = 0.00183 (clear difference)
- From scratch vs API gpt-oss-120b on paraphrase (108): only the first right 0, only the second right 60, p = 1.73e-18 (clear difference)
- LoRA Qwen2.5-0.5B vs API gpt-oss-120b on test (200): only the first right 15, only the second right 5, p = 0.0414 (clear difference)
- LoRA Qwen2.5-0.5B vs API gpt-oss-120b on paraphrase (108): only the first right 6, only the second right 15, p = 0.0784 (no clear difference)
