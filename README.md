# YSDA Natural Language Processing course
* This is the 2023 version. For previous year' course materials, go to [this branch](https://github.com/yandexdataschool/nlp_course/tree/2022)
* Lecture and seminar materials for each week are in ./week* folders, see README.md for materials and instructions
* Any technical issues, ideas, bugs in course materials, contribution ideas - add an [issue](https://github.com/yandexdataschool/nlp_course/issues)
* Installing libraries and troubleshooting: [this thread](https://github.com/yandexdataschool/nlp_course/issues/1).


# Syllabus
- [__week01__](./week01_embeddings) __Word Embeddings__
  - Lecture: Word embeddings. Distributional semantics. Count-based (pre-neural) methods. Word2Vec: learn vectors. GloVe: count, then learn. Evaluation: intrinsic vs extrinsic. Analysis and Interpretability. [Interactive lecture materials and more.](https://lena-voita.github.io/nlp_course.html#preview_word_emb)
  - Seminar: Playing with word and sentence embeddings
  - Homework: Embedding-based machine translation system

- [__week02__](./week02_classification) __Text Classification__
  - Lecture: Text classification: introduction and datasets. General framework: feature extractor + classifier. Classical approaches: Naive Bayes, MaxEnt (Logistic Regression), SVM. Neural Networks: General View, Convolutional Models, Recurrent Models. Practical Tips: Data Augmentation. Analysis and Interpretability. [Interactive lecture materials and more.](https://lena-voita.github.io/nlp_course.html#preview_text_clf)
  - Seminar: Text classification with convolutional NNs.
  - Homework: Statistical & neural text classification.
  
- [__week03__](./week03_lm) __Language Modeling__
  - Lecture: Language Modeling: what does it mean? Left-to-right framework. N-gram language models. Neural Language Models: General View, Recurrent Models, Convolutional Models. Evaluation. Practical Tips: Weight Tying. Analysis and Interpretability. [Interactive lecture materials and more.](https://lena-voita.github.io/nlp_course.html#preview_lang_models)
  - Seminar: Build a N-gram language model from scratch
  - Homework: Neural LMs & smoothing in count-based models.
  
- [__week04__](./week04_seq2seq) __Seq2seq and Attention__
  - Lecture: Seq2seq Basics: Encoder-Decoder framework, Training, Simple Models, Inference (e.g., beam search). Attention: general, score functions, models. Transformer: self-attention, masked self-attention, multi-head attention; model architecture. Subword Segmentation (BPE). Analysis and Interpretability: functions of attention heads; probing for linguistic structure. [Interactive lecture materials and more.](https://lena-voita.github.io/nlp_course.html#preview_seq2seq_attn)
  - Seminar: Basic sequence to sequence model
  - Homework: Machine translation with attention
  
- [__week05__](./week05_transfer) __Transfer Learning__
  - Lecture: What is Transfer Learning? Great idea 1: From Words to Words-in-Context (CoVe, ELMo). Great idea 2: From Replacing Embeddings to Replacing Models (GPT, BERT). (A Bit of) Adaptors. Analysis and Interpretability. [Interactive lecture materials and more.](https://lena-voita.github.io/nlp_course.html#preview_transfer)
  - Homework: fine-tuning a pre-trained BERT model


- [__week06__](./week6_llm) __LLMs and Prompting__
  - Lecture: Scaling laws. Emergent abilities. Prompting (aka "in-context learning"): techiques that work; questioning whether model "understands" prompts. Hypotheses for why and how in-context learning works. Analysis and Interpretability.
  - Homework: manual prompt engneering and chain-of-thought reasoning

- [__week07__] __Transformer architecture and training__
  - Lecture: training tips for transformers; the evolution of transformer architecture from Vaswani et al (2017) to modern LLMs; parameter-efficient fine-tuning (PEFT)
  - Homework: fine-tuning a large language model with PEFT algorithms


More TBA

# Contributors & course staff
Course materials and teaching performed by
- [Elena Voita](https://lena-voita.github.io) - course admin, lectures, seminars, homeworks
- [Valentina Broner](https://www.hse.ru/org/persons/831207784/?_gl=1%2a1hz2yht%2a_ga%2aMTg3MTM2ODIwMS4xNjk4NTEyODg5%2a_ga_D145P1R4PL%2aMTY5ODUxMjg4OC4xLjAuMTY5ODUxMjg4OC42MC4wLjA.) - course admin for on-campus students
- [Boris Kovarsky](https://github.com/kovarsky), [David Talbot](https://github.com/drt7), [Sergey Gubanov](https://github.com/esgv), [Just Heuristic](https://github.com/justheuristic) - help build course materials and/or held some classes
- [30+ volunteers](https://github.com/yandexdataschool/nlp_course/graphs/contributors) who contributed and refined the notebooks and course materials. Without their help, the course would not be what it is today
- [A mighty host of TAs](https://lk.yandexdataschool.ru/courses/2023-autumn/7.1171-avtomaticheskaia-obrabotka-tekstov/) who stoically grade hundreds of homework submissions from on-campus students each year

