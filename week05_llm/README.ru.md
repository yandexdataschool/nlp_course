## Неделя 5. Большие языковые модели (LLM)

### Материалы
- [Слайды лекции](./lecture_llm.pdf) ([Google Slides](https://docs.google.com/presentation/d/1Gklql_r2DsArtx2GePqd0FX3bt75GMOj/edit?usp=sharing&ouid=114346045969351679204&rtpof=true&sd=true)).
- **Видео на русском:** [лекция и практика](https://disk.yandex.ru/i/_7cKg_NyNBqnsg).

### Практика
- Семинар: [`seminar.ipynb`](./seminar.ipynb) · [Colab](https://colab.research.google.com/github/yandexdataschool/nlp_course/blob/2026/week05_llm/seminar.ipynb).
- Домашнее задание: инструкции в [`homework/README.md`](./homework/README.md).

### Открытые модели, упомянутые в лекции
- [Qwen](https://huggingface.co/Qwen).
- [LLaMA](https://huggingface.co/meta-llama/).
- [Falcon](https://huggingface.co/tiiuae/falcon-180B).
- [BLOOM](https://huggingface.co/bigscience/bloom).

Для доступа к некоторым моделям нужно подать заявку, рассмотрение которой занимает время. Пока заявка обрабатывается, можно поискать на Hugging Face альтернативные версии той же модели, например квантованные, доступные без заявки. В оригинальном курсе приведён пример [4-битной LLaMA-2-70B](https://huggingface.co/TheBloke/Llama-2-70B-Chat-GPTQ), которую можно загрузить через `transformers.AutoModelForCausalLM.from_pretrained`. **Ограничения лицензии исходной модели могут сохраняться**, даже если скачивание альтернативной версии не требует отдельной заявки.

### Дополнительные материалы
- [Glitch-токены, включая SolidGoldMagikarp](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation).
- [Sparks of AGI — дискуссионная, но влиятельная статья о возможностях LLM](https://arxiv.org/abs/2303.12712).
- [BIG-bench — бенчмарк для исследования возможностей LLM](https://github.com/google/BIG-bench).
