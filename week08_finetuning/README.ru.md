## Неделя 8. Дообучение (PEFT и RLHF)

### Материалы
- [Слайды лекции](./lecture.pdf).
- **Видео на русском:** [лекция](https://disk.yandex.ru/i/Axjf2u-hWDLAMw), [семинар](https://disk.yandex.ru/i/I_rAisZlK22FLA).
- Материалы других курсов (английский): [семинар EMNLP по PEFT (3,5 ч)](https://www.youtube.com/watch?v=KoOlcX3XLd4), [короткая версия MunichNLP](https://www.youtube.com/watch?v=StdrAJZsmw4), [урок Hugging Face по RLHF](https://www.youtube.com/watch?v=2MBJOuVq380).
- По желанию: [лекция о чат-ботах, ориентированных на выполнение задач (русский)](https://disk.yandex.ru/i/4e_vqRDqwVGiFA), [лекция о диалоговых системах (английский)](https://disk.yandex.ru/i/XR1-8CghVOIK7A).

### Практика
Два задания по 5 баллов каждое; можно получить дополнительные баллы.
- Семинар: [`seminar.ipynb`](./seminar.ipynb) · [Colab](https://colab.research.google.com/github/yandexdataschool/nlp_course/blob/2026/week08_finetuning/seminar.ipynb).
- Домашнее задание: [`homework.ipynb`](./homework.ipynb). В конце есть необязательное сложное задание по дообучению с подкреплением (RL).

### Дополнительные материалы: дообучение с подкреплением и alignment
- [CarperAI/trlx](https://github.com/CarperAI/trlx) — альтернатива `trl` для более крупных моделей.
- Подробный разбор алгоритмов RL, применяемых в RLHF: [часть 1](https://github.com/yandexdataschool/Practical_RL/tree/240c989fb1cca52effc289c4033438fa67af1e20/week06_policy_based) и [часть 2](https://github.com/yandexdataschool/Practical_RL/tree/240c989fb1cca52effc289c4033438fa67af1e20/week09_policy_II).
- [Constitutional AI от Anthropic](https://arxiv.org/abs/2212.08073).
- Ранние исследования RL для генерации естественного языка: [диалоговая система для решения задач](https://arxiv.org/abs/1703.07055), [генерация диалогов](https://arxiv.org/abs/1606.01541), [SeqGAN](https://arxiv.org/abs/1609.05473).
- [Обзор методов машинного перевода с обсуждением RL и его неудач](https://arxiv.org/abs/1609.08144).
- Дополнительные ссылки есть непосредственно в слайдах лекции.

### Дополнительные материалы: архитектура моделей
- [Colin Raffel: Building ML models like we build open-source software](https://www.youtube.com/watch?v=0oGxT_i7nk8).
- [Разбор Rotary Position Embeddings от EleutherAI](https://blog.eleuther.ai/rotary-embeddings/).
- [Grouped-Query Attention для сокращения потребления памяти при инференсе](https://arxiv.org/abs/2305.13245v2).
- [Gated-активации для улучшения трансформеров](https://arxiv.org/abs/2002.05202).
- Другие ссылки приведены в слайдах лекции.
