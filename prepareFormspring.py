# Experiment 1. Script of sentiments analytics.
__author__      = 'Sergio Jiménez Barrio'
__copyright__   = "Copyright 2018, University of Seville"
__credits__     = ["Teodoro Alamo"]
__email__       = "sergio.jimbar@gmail.com"
__status__      = "BETA"
__version__     = "0.1"

import pandas as pd
import re

# Load csv
df = pd.read_csv('./data/formspring_data.csv', sep='\t', encoding='utf-8')
# Get only the question (usually, in this part localize the ciberbullying)
list_ = []
for post in df['post'].tolist():
    m = re.search("Q: (.*?)<br>", post)
    list_.append(m.group(1))
# Merge with the ciberbullying tag (ans1)
df_post = pd.DataFrame(columns=['text'], data=list_)
df_ans = pd.DataFrame(columns=['ciberbullying'], data=df.ans1.tolist())
df_data = pd.DataFrame(columns=['text', 'ciberbullying'])
df_data['text'] = df_post['text'].copy()
df_data['ciberbullying'] = df_ans['ciberbullying'].copy()

# Compensate the dataset (A lot of no ciberbullying questions)
df_withoutCB = df_data[df_data.ciberbullying == 'No']
df_withCB = df_data[df_data.ciberbullying == 'Yes']

print ('Posts with ciberbullying facts:' + str(len(df_withCB)))
print ('Posts without ciberbullying facts:' + str(len(df_withoutCB)))

# Get posts without ciberbullying with 7 words or more
i = 0
for index, row in df_withoutCB.iterrows():
    if len(row.text.split()) < 7:
        df_withoutCB = df_withoutCB.drop(index)
        i += 1

print ('Posts without ciberbullying deleted: ' + str(i))
print ('Posts without ciberbullying facts: with 7 words or more: ' + str(len(df_withoutCB)))

# Get the same number of posts withput ciberbullying that with ciberbullying to compesate dataset
df_withoutCB = df_withoutCB.sample(len(df_withCB))
print ('New DF size without ciberbullying in post: ' + str(len(df_withoutCB)))

# Merge both datasets with and without bullying
df = pd.concat([df_withoutCB, df_withCB])
# Mixing dataset
df = df.sample(frac=1).reset_index(drop=True)

# Cleaning text
df = df.replace({'&#39;': "'"}, regex=True)
# Changed by 1 and 0
df['ciberbullying'] = df['ciberbullying'].replace('Yes', 1)
df['ciberbullying'] = df['ciberbullying'].replace('No', 0)



# Save final dataset
df.to_csv('./data/formspring_compensated_data.csv', encoding='utf8', sep=';')
print ('Data exported correctly! Good Bye :-)')

