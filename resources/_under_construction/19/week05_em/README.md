#### Generative Models, Expectation-Maximization and Hidden Markov Models

Today's lecture will cover generative models, maximum likelihood estimation from incomplete data using Expectation-Maximization and Hidden Markov models.

The Expectation-Maximization algorithm is a general algorithm for estimating models when some variables are not observed. We'll eventually use it to learn alignments between words in parallel text, but today we'll focus on some toy examples and some deciphering problems.

* (Slides) [Generative models and EM](https://github.com/yandexdataschool/nlp_course/blob/2019/week05_em/generative_models_and_em.pdf) 
* (Videos) [part1](https://yadi.sk/i/4LcSl4Lg4B6Rsg), [part2](https://yadi.sk/i/v5LEWUQKRPpO3g)
* First exercise on MLE and EM [notebook](coins-seminar.ipynb)
* Homework notebook using HMM to decipher text [notebook](hmm-seminar.ipynb) This homework is worth 15 points and is due by the 24th Oct. 15.30 MSK time).

Please submit your results for TASK 5 from the hmm-seminar.ipynb.

1. The decrypted text for each file from the 'encrypted' directory using HMMs trained on any files you want to from the 'plaintext' directory:

Please submit this as a tarball archive (i.e. tar -cvzf results.tar.gz decrypted/*.txt)

Please note you should not be able to perfectly decrypt the text using HMMs. Doing so, will be an indication that you have used other methods and will not be counted!

2. For each of the encrypted texts, please indicate the most likely author or book/volume title using one of the file names from the plaintext directory. 

Please submit this as a single text file in the following format:

0_encrypted.txt shakespeare.txt  
1_encrypted.txt mayakovsky.txt  

etc.


