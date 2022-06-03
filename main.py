import utils
import loader
import numpy as np

data = loader.load_data('data')
stimuli = loader.load_stimuli('data')

import ipdb;ipdb.set_trace()

acc_female = np.zeros([3, 1])
acc_male = np.zeros([3, 1])
std_female = np.zeros([3, 1])
std_male = np.zeros([3, 1])
acc_experience = np.zeros([3, 3])
std_experience = np.zeros([3, 3])
acc_usefulness = np.zeros([2, 4])
std_usefulness = np.zeros([2, 4])

dict_usefulness = {'not useful': 0, 'rarely useful': 1, 'useful': 2, 'very useful': 3}

data[0]['usefulness_int'] = data[0]['usefulness'].apply(lambda col: col[0]).map(dict_usefulness)
data[1]['usefulness_int'] = data[1]['usefulness'].apply(lambda col: col[0]).map(dict_usefulness)

df_words = [[] for _ in range(2)]


def main():
    #loop over all conditions
    for i in range(3):
        #general accuracy
        acc = utils.accuracy(data[i], stimuli)
        data[i]['acc'] = np.around(np.ravel(acc), decimals=2)
        time_mean, time_std, time = utils.time_diff(data[i])
        print("acc:", np.around(np.mean(acc), decimals=2), "time:", time_mean)
        if i < 2:
            df_words[i] = utils.extract_pos(stimuli[i])
            utils.extract_attributes(df_words[i], str(i))
            # accuracy by usefulness
            acc_usefulness[i], std_usefulness[i] = utils.accuracy_by_usefulness(data[i], stimuli)
        # accuracy by gender
        acc_female[i], acc_male[i], std_female[i], std_male[i] = utils.accuracy_by_gender(data[i], stimuli)
        # accuracy by experience
        acc_experience[i], std_experience[i] = utils.accuracy_by_experience(data[i], stimuli)

    utils.barplot_experience(acc_experience, std_experience)
    utils.barplot_gender(acc_female, acc_male, std_female, std_male)
    utils.barplot_usefulness(acc_usefulness, std_usefulness)


if __name__ == '__main__':
    main()