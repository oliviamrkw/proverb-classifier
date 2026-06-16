## Are Multilingual LLMs Culturally-Diverse Reasoners? An Investigation into Multicultural Proverbs and Sayings

This folder contains the dataset MAPS.

`test_proverbs` is the evaluation split.
`fs_proverbs` is used for few-shot demonstrations (you could also use it as a dev split for picking prompts).
`machine_translation_2en` is machine translated data to English using Google Translate (in August/September 2023, Google Translate maybe very different now).

In this distribution, we include the `explanation` column, which we have not vetted for quality nor coverage. This column is collected from various online sources. Please use it with discretion.

If you find the dataset useful, please cite:

@misc{liu2023multilingual,
      title={Are Multilingual LLMs Culturally-Diverse Reasoners? An Investigation into Multicultural Proverbs and Sayings}, 
      author={Chen Cecilia Liu and Fajri Koto and Timothy Baldwin and Iryna Gurevych},
      year={2023},
      eprint={2309.08591},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
