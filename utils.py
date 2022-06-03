import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import seaborn as sns
nlp = spacy.load("en_core_web_sm")


def bootstrap(acc):
    n = len(acc)  # size of the sample you want
    return np.std(np.random.choice(acc, n))


def time_diff(data):
    time = np.zeros([len(data), 40])
    for index, row in data.iterrows():
        for i in range(40):
            try:
                start_dt = dt.datetime.strptime(row["start_time_" + str(i+1)][0], '%H:%M:%S')
                end_dt = dt.datetime.strptime(row["end_time_" + str(i+1)][0], '%H:%M:%S')
                diff = (end_dt - start_dt)
                time[index, i] = diff.seconds

            except Exception as err:
                time[index, i] = np.nan
                print(err)

    return np.nanmean(time), np.nanstd(time), time


def accuracy(data, stimuli):
    acc = np.zeros([len(data), 1])
    for index, row in data.iterrows():
        # iterate through the stimuli that matches the user code
        annotations = [row[str(i)][0] == 'reliable' for i in range(1, 41) if row[str(i)] == row[str(i)]]

        if row["user_code"] == 123:
            gold = (stimuli[0]["true"] == 'True').tolist()
            agreement = [1 if gold[ii] == annotations[ii] else 0 for ii in range(len(annotations))]
        elif row["user_code"] == 456:
            gold = (stimuli[1]["true"] == 'True').tolist()
            agreement = [1 if gold[ii] == annotations[ii] else 0 for ii in range(len(annotations))]
        else:
            raise NotImplementedError

        acc[index] = np.sum(agreement) / len(agreement)

    return acc


def accuracy_by_gender(data, stimuli):
    acc_female = []
    acc_male = []
    for index, row in data.iterrows():
        # iterate through the stimuli that matches the user code
        annotations = [row[str(i)][0] == 'reliable' for i in range(1, 41) if row[str(i)] == row[str(i)]]
        if row["user_code"] == 123:
            gold = (stimuli[0]["true"] == 'True').tolist()
            agreement = [1 if gold[ii] == annotations[ii] else 0 for ii in range(len(annotations))]
        elif row["user_code"] == 456:
            gold = (stimuli[1]["true"] == 'True').tolist()
            agreement = [1 if gold[ii] == annotations[ii] else 0 for ii in range(len(annotations))]
        else:
            raise NotImplementedError

        if row['gender'][0] == 'female':
            acc_female.append(np.sum(agreement)/len(agreement))
        elif row['gender'][0] == 'male':
            acc_male.append(np.sum(agreement) / len(agreement))

    return np.around(np.mean(acc_female), decimals=2), np.around(np.mean(acc_male), decimals=2), \
           np.around(bootstrap(acc_female), decimals=2), np.around(bootstrap(acc_male), decimals=2)


def accuracy_by_experience(data, stimuli):
    acc0 = []
    acc3 = []
    acc10 = []
    output = np.zeros([3,1])
    for index, row in data.iterrows():
        # iterate through the stimuli that matches the user code
        annotations = [row[str(i)][0] == 'reliable' for i in range(1, 41) if row[str(i)] == row[str(i)]]
        if row["user_code"] == 123:
            gold = (stimuli[0]["true"] == 'True').tolist()
            agreement = [1 if gold[ii] == annotations[ii] else 0 for ii in range(len(annotations))]
        elif row["user_code"] == 456:
            gold = (stimuli[1]["true"] == 'True').tolist()
            agreement = [1 if gold[ii] == annotations[ii] else 0 for ii in range(len(annotations))]
        else:
            raise NotImplementedError

        if row['experience'][0] in ['<1', '1-3']:
            acc0.append(np.sum(agreement) / len(agreement))
        elif row['experience'][0] in ['3-5', '5-10']:
            acc3.append(np.sum(agreement) / len(agreement))
        elif row['experience'][0] == 'Morethan10':
            acc10.append(np.sum(agreement) / len(agreement))
        else:
            print('skip', index, '(no information about experience)')

    output[0] = np.around(np.mean(acc0), decimals=2) if len(acc0)>0 else 1
    output[1] = np.around(np.mean(acc3), decimals=2) if len(acc3) > 0 else 1
    output[2] = np.around(np.mean(acc10), decimals=2) if len(acc10) > 0 else 1

    std = [np.around(bootstrap(acc0), decimals=2) if len(acc0) > 0 else 1,
           np.around(bootstrap(acc3), decimals=2) if len(acc3) > 0 else 1,
           np.around(bootstrap(acc10), decimals=2) if len(acc3) > 0 else 1]

    return np.ravel(output), np.ravel(std)



def accuracy_by_usefulness(data, stimuli):
    acc = [[] for _ in range(4)]
    output = np.zeros([4, 1])
    std = np.zeros([4, 1])
    for index, row in data.iterrows():
        if row['usefulness_int']!=row['usefulness_int']:
            print('skip', index, '(no information about usefulness)')
            continue
        # iterate through the stimuli that matches the user code
        annotations = [row[str(i)][0] == 'reliable' for i in range(1, 41) if row[str(i)] == row[str(i)]]
        if row["user_code"] == 123:
            gold = (stimuli[0]["true"] == 'True').tolist()
            agreement = [1 if gold[ii] == annotations[ii] else 0 for ii in range(len(annotations))]
        elif row["user_code"] == 456:
            gold = (stimuli[1]["true"] == 'True').tolist()
            agreement = [1 if gold[ii] == annotations[ii] else 0 for ii in range(len(annotations))]
        else:
            raise NotImplementedError

        usefulness = int(row['usefulness_int'])
        acc[usefulness].append(np.sum(agreement) / len(agreement))

    for ii in range(4):
        output[ii] = np.around(np.mean(acc[ii]), decimals=2) if len(acc[ii]) >0 else 1
        std[ii] = np.around(bootstrap(acc[ii]), decimals=2) if len(acc[ii]) > 0 else 0

    return np.ravel(output), np.ravel(std)


def barplot_gender(acc_female, acc_male, std_female, std_male):
    width = 0.3
    fig, ax = plt.subplots()
    X = np.arange(3)
    ax.bar(X - .15, 1 - np.ravel(acc_female)[::-1], width, yerr=np.ravel(std_female)[::-1], label='female')
    ax.bar(X + .15, 1 - np.ravel(acc_male)[::-1], width, yerr=np.ravel(std_male)[::-1], label='male')
    ax.set_xticks(X)
    ax.set_xticklabels(['Text', '+Confidence', '+Confidence\n+Feat.Attr.'])
    ax.legend()
    ax.set_ylabel('Error')
    plt.savefig('error_gender', dpi=300)


def barplot_experience(acc, std):
    experience_by_year = ['0-3', '3-10', 'Morethan10']
    width = 0.25
    fig, ax = plt.subplots()
    X = np.arange(3)
    ax.bar(X - .25, 1 - np.ravel(acc[:, 0])[::-1], width, yerr=np.ravel(std[:, 0])[::-1], label=experience_by_year[0])
    ax.bar(X, 1 - np.ravel(acc[:, 1])[::-1], width, yerr=np.ravel(std[:, 1])[::-1], label=experience_by_year[1])
    ax.bar(X + .25, 1 - np.ravel(acc[:, 2])[::-1], width, yerr=np.ravel(std[:, 2])[::-1], label=experience_by_year[2])
    ax.set_xticks(X)
    ax.set_xticklabels(['Text', '+Confidence', '+Confidence\n+Feat.Attr.'])
    ax.legend()
    ax.set_ylabel('Error')
    plt.savefig('error_experience', dpi=300)


def barplot_usefulness(acc, std):
    usefulness = ['not useful', 'rarely useful', 'useful', 'very useful']
    width = 0.2
    fig, ax = plt.subplots()
    X = np.arange(2)
    ax.bar(X - .3, 1 - np.ravel(acc[:, 0])[::-1], width, yerr=np.ravel(std[:, 0])[::-1], label=usefulness[0])
    ax.bar(X - .1, 1 - np.ravel(acc[:, 1])[::-1], width, yerr=np.ravel(std[:, 1])[::-1], label=usefulness[1])
    ax.bar(X + .1, 1 - np.ravel(acc[:, 2])[::-1], width, yerr=np.ravel(std[:, 2])[::-1], label=usefulness[2])
    ax.bar(X + .3, 1 - np.ravel(acc[:, 3])[::-1], width, yerr=np.ravel(std[:, 3])[::-1], label=usefulness[3])
    ax.set_xticks(X)
    ax.set_xticklabels(['Confidence', 'Confidence\n+Feat.Attr.'])
    ax.legend()
    ax.set_ylabel('Error')
    plt.savefig('error_usefulness', dpi=300)


def extract_pos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Part-of-Speech Tag and add column to input dataframe.
    Args:
        df: pd.DataFrame on word-level from which POS is to be extracted
    Returns
        pd.DataFrame: `df` with additional column `pos`
    """
    df_words = df.apply(pd.Series.explode).reset_index()
    # df_words = df.explode('tokens').reset_index()
    for index, row in df_words.iterrows():
        doc = nlp(row['tokens'])
        pos = [token.pos_ for token in doc]
        df_words.loc[index, 'pos'] = pos[0]

    return df_words


def extract_attributes(df, ii):

    vals = []
    min = []
    max = []

    for group_name, subdf in df.groupby('id'):
        if np.min(subdf.attributions.values) != 0:
            vals.extend(subdf.pos.values[np.argsort(subdf.attributions.values)[:10]])
            min.append(group_name)
        if np.max(subdf.attributions.values) != 0:
            vals.extend(subdf.pos.values[np.argsort(subdf.attributions.values)[-10:]])
            max.append(group_name)

    df_pos = pd.DataFrame(vals, columns=['pos'])
    ax = sns.countplot(x="pos",
                       data=df_pos,
                       order=df_pos['pos'].value_counts().index)

    plt.savefig('pos_attributes_' + ii, dpi=300)