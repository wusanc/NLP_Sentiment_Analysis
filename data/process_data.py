import pandas as pd
from sklearn.model_selection import train_test_split

# part.csv: 0=中性(1), 1=正面(2)
part = pd.read_csv('data/part.csv', sep='\t', header=None, names=['text','label'])
part['label'] = part['label'].map({0: 1, 1: 2})

# waimai_10k.csv: 0=负面(0), 1=正面(2)
waimai = pd.read_csv('data/waimai_10k.csv')
waimai = waimai.rename(columns={'review': 'text'})[['text','label']]
waimai['label'] = waimai['label'].map({0: 0, 1: 2})

df = pd.concat([part, waimai], ignore_index=True)
df = df.dropna(subset=['text','label'])
df['label'] = df['label'].astype(int)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

labels = {0: '负面', 1: '中性', 2: '正面'}
print('合并后:', df.shape)
for k, v in labels.items():
    count = (df['label'] == k).sum()
    print('  %s(%d): %d' % (v, k, count))

train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
train.to_csv('data/train.csv', index=False, encoding='utf-8-sig')
test.to_csv('data/test.csv', index=False, encoding='utf-8-sig')

print('\ntrain: %d, test: %d' % (len(train), len(test)))
for k, v in labels.items():
    tc = (train['label'] == k).sum()
    vc = (test['label'] == k).sum()
    print('  train %s: %d, test %s: %d' % (v, tc, v, vc))
print('\nDone')
