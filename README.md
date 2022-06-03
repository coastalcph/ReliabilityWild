## Evaluating Deep Taylor Decomposition for Reliability Assessment in the Wild

This repository contains the code for the paper `Evaluating Deep Taylor Decomposition for Reliability Assessment in the Wild` which was presented at ICWSM 2022.  

```
@article{Brandl_Hershcovich_Søgaard_2022, 
    title={Evaluating Deep Taylor Decomposition for Reliability Assessment in the Wild}, 
    volume={16}, url={https://ojs.aaai.org/index.php/ICWSM/article/view/19389}, 
    number={1}, 
    journal={Proceedings of the International AAAI Conference on Web and Social Media}, 
    author={Brandl, Stephanie and Hershcovich, Daniel and Søgaard, Anders}, 
    year={2022}, 
    month={May}, 
    pages={1368-1372} 
}
```

### Dataset  
You can find the journalists' annotations from all 3 conditions in `./data/annotations_exp{condition}`.  
`exp0` refers to the condition where model confidence and feature attribution was shown  
`exp1` includes only model confidence  
`exp2` shows only the respective article without model support  

Each row in the dataframes corresponds to one journalist's answer given the `start_time_n`, `end_time_n` and the corresponding the assessment in `n`.

There are 2 additional files including the stimuli: `stimuli123.pkl` and `stimuli456.pkl` as we have shown one of the two sets of stimuli to each journalist. You can find the corresponding information in the annotations in the column `user_code`.

### Analysis  
Running `main.py` should output the corresponding plots from the paper as well as the accuracy and average time per article for each condition. Please note that the error bars might differ from the ones in the paper as they are bootstrapped.