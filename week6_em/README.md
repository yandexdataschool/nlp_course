#### Generative Models, Expectation-Maximization and Word Alignment Models

Today's lecture will cover simple generative models, maximum likelihood estimation from both complete and incomplete data and latent variable word alignment models.

The Expectation-Maximization algorithm is a general algorithm for estimating models when some variables are not observed. It can be seen as a form of _variational inference_.

* (Slides 1) [Generative models, MLE and EM](generative_models_em.pdf) 
* (Slides 2) [Word alignment models](word_alignment.pdf).
* (Notes) [Detailed notes](word_alignment_notes.pdf)

__Videos:__
* our [lecture](https://yadi.sk/i/4ZA6BCJBQAxy3Q) and [seminar](https://yadi.sk/i/uPuNKeGkULujVg) (in english!)
* alternative lecture on EM (outside NLP)
Seminar will use this [notebook](mle_em_seminar.ipynb).

#### Homework (due in class next week)


In preparation for next week's class on Machine Translation, you should form groups of five or six students, pick one of the following questions and be prepared to give a short presentation during the lecture. 

Each person should read at least one paper and your group should probably meet in advance of the class to finalize your presentation. 

As well as explaining the main ideas in the papers, please also pay attention to any problems with the experimental set up in the paper and comment on whether their conclusions are well supported by their results.


1. What are the main computational and statistical bottlenecks in NMT? How can we reduce them?
  * [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909.pdf)
  * [Simple, Fast Noise-Contrastive Estimation for Large RNN Vocabularies](http://www.aclweb.org/anthology/N16-1145.pdf)
  * [Vocabulary Manipulation for Neural Machine Translation](http://www.aclweb.org/anthology/P16-2021)
  * [Fully Character-Level Neural Machine Translation without Explicit Segmentation](http://aclweb.org/anthology/Q17-1026)
  * [Using the Output Embedding to Improve Language Models](http://www.aclweb.org/anthology/E17-2025)

2. What are the pros/cons of different Encoder-Decoder architectures? (RNNs, ConvS2S, Transformer, etc.)
  * [Google's Neural Machine Translation System](https://arxiv.org/pdf/1609.08144.pdf)
  * [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  * [Convolutional Sequence to Sequence Learning](https://arxiv.org/pdf/1705.03122.pdf)
  * [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/pdf/1701.06538.pdf)
  * [The Importance of Being Recurrent for Modeling Hierarchical Structure](https://arxiv.org/pdf/1803.03585.pdf)
  * [Colorless green recurrent networks dream hierarchically](https://arxiv.org/pdf/1803.11138.pdf)

3. How can monolingual data be used to improve NMT?
  * [On Using Monolingual Corpora in Neural Machine Translation](https://arxiv.org/pdf/1503.03535.pdf)
  * [Improving Neural Machine Translation Models with Monolingual Data](https://arxiv.org/abs/1511.06709)
  * [Iterative Back-Translation for Neural Machine Translation](http://www.aclweb.org/anthology/W18-2703)
  * [Understanding Back-Translation at Scale](https://arxiv.org/pdf/1808.09381.pdf)
  * [Back-Translation Sampling by Targeting Difficult Words in Neural Machine Translation](https://arxiv.org/pdf/1808.09006.pdf)

4. How can we build NMT systems for language pairs with very little parallel data?
  * [Multi-Way, Multilingual Neural Machine Translation with a Shared Attention Mechanism](http://www.aclweb.org/anthology/N16-1101)
  * [Contextual Parameter Generation for Universal Neural Machine Translation](https://arxiv.org/abs/1808.08493)
  * [Dual Learning for Machine Translation](https://arxiv.org/pdf/1611.00179.pdf)
  * [Zero-Shot Dual Machine Translation](https://arxiv.org/abs/1805.10338)
  * [Phrase-Based & Neural Unsupervised Machine Translation](https://arxiv.org/pdf/1804.07755.pdf)

5. Has NMT really bridged the gap between MT and human translation? What problems remain?
  * [Google's Neural Machine Translation System](https://arxiv.org/pdf/1609.08144.pdf)
  * [Achieving Human Parity on Automatic Chinese to English News Translation](https://www.microsoft.com/en-us/research/uploads/prod/2018/03/final-achieving-human.pdf)
  * [Six Challenges for NMT](http://www.aclweb.org/anthology/W17-3204)
  * [Analyzing Uncertainty in Neural Machine Translation](https://arxiv.org/pdf/1803.00047.pdf)
  * [Towards Robust Neural Machine Translation](https://arxiv.org/pdf/1805.06130.pdf)
  * [Context-Aware Neural Machine Translation Learns Anaphora Resolution](http://www.aclweb.org/anthology/P18-1117)
