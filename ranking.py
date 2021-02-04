import numpy as np
import pandas as pd
from collections import defaultdict
from difflib import SequenceMatcher

class PropertyRanker(object):

    def __init__(self):
        self.table_counts = self._build_table()
        self.table_names = self._build_names()

    def _build_table(self):

        claims_counts = pd.read_csv('data/claims.label.entity.counts.tsv.gz', sep='\t', usecols=['node1','node2']).set_index('node1')
        claims_counts.columns = [['main value']]

        qualifier_counts = pd.read_csv('data/qualifiers.label.property.counts.tsv.gz', sep='\t', usecols=['node1','node2']).set_index('node1')
        qualifier_counts.columns = [['qualifier']]

        total_counts = pd.read_csv('data/all.label.property.counts.tsv.gz', sep='\t', usecols=['node1','node2']).set_index('node1')
        total_counts.columns = [['total']]

        glossory = pd.concat([claims_counts, qualifier_counts, total_counts], axis=1).fillna(0).astype(int)
        glossory.columns = [x[0] for x in glossory]
        glossory['both'] = glossory['main value'] + glossory['qualifier']

        glossory['p:main_value'] = glossory['main value'] / glossory['both']
        glossory['p:qualifier'] = 1.0 - glossory['p:main_value']

        return glossory

    def _build_names(self):

        def gen_df_properties(fname):
            for chunk in pd.read_csv(fname, sep='\t', usecols=['node1', 'node2'], chunksize=10000):
                mask = chunk['node1'].apply(lambda x: x.startswith('P'))
                if mask.all():
                    yield chunk
                else:
                    yield chunk.loc[mask]
                    break

        labels = pd.concat(gen_df_properties('data/labels.en.tsv.gz'))
        labels['node2'] = labels['node2'].apply(lambda x: x[1:-4])
        labels.columns = ['pnode', 'label']
        labels = labels.groupby('pnode')['label'].apply(list).reset_index().set_index('pnode')

        aliases = pd.concat(gen_df_properties('data/aliases.en.tsv.gz'))
        aliases['node2'] = aliases['node2'].apply(lambda x: x[1:-4])
        aliases.columns = ['pnode', 'alias']
        aliases = aliases.groupby('pnode')['alias'].apply(list).reset_index().set_index('pnode')

        merged = pd.concat([aliases, labels], axis=1)
        for row in merged.loc[merged.alias.isnull(), 'alias'].index:
            merged.at[row, 'alias'] = []
        merged['names'] = merged.apply(lambda r: r['label'] + r['alias'], axis=1)

        return merged

    def gen_counts(self, pnodes, scope='both'):
        counts = defaultdict(int)
        for node in pnodes:
            try:
                counts[node] = int(self.table_counts.loc[node][scope])
            except KeyError:
                counts[node] = 0
        return counts

    def gen_similarity(self, pnodes, query):
        sim = defaultdict(float)
        for node in pnodes:
            try:
                sim[node] = np.max([SequenceMatcher(None, query, n).ratio() for n in self.table_names.loc[node]['names']])
            except KeyError:
                sim[node] = 0.0
        return sim

    def rank(self, pnodes, query, scope='both'):

        ranking = defaultdict(float)
        counts = self.gen_counts(pnodes, scope)
        sim = self.gen_similarity(pnodes, query)

        for pnode in pnodes:
            ranking[pnode] = sim[pnode] * np.log(counts[pnode]+1)

        return dict(sorted(ranking.items(), key=lambda x:x[1], reverse=True))

    def rank_wlabel(self, pnodes, query, scope='both'):

        ranking = self.rank(pnodes, query, scope)
        ranking_wlabel = []
        for r, v in ranking.items():
            try:
                ranking_wlabel.append((r, self.table_names.loc[r]['label'][0], v))
            except:
                pass

        return ranking_wlabel
