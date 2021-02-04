from requests import get
import pandas as pd
from collections import defaultdict
import json
from math import isnan, nan
from ranking import PropertyRanker

from flask import request
from time import time

import warnings
warnings.filterwarnings("ignore")

allowed_types = ['commonsMedia', 'wikibase-item', 'external-id', 'url', 'string',
                    'quantity', 'time', 'globe-coordinate', 'monolingualtext',
                    'wikibase-property', 'math', 'geo-shape', 'tabular-data',
                    'wikibase-lexeme', 'wikibase-form', 'wikibase-sense',
                    'musical-notation']

type_aliases = {'media': 'commonsMedia', 'item': 'wikibase-item', 'id': 'external-id', 'coordinate': 'globe-coordinate',
                'property': 'wikibase-property', 'lexeme': 'wikibase-lexeme', 'form': 'wikibase-form', 'sense': 'wikibase-sense',
                'country': 'wikibase-item', 'location': 'wikibase-item'
                }

class PropertyFinder(object):

    def __init__(self, host='https://kgtk.isi.edu/api',
                        metadata_type='metadata.property.datatypes.tsv.gz',
                        labels='labels_.tsv.gz',
                        metadata_constraints='constraints.json',
                        query_size=500):
        self.host = host
        self.types = defaultdict(lambda:'', pd.read_csv(metadata_type, sep='\t', usecols=['node1','node2']).set_index('node1').to_dict()['node2'])
        self.labels = defaultdict(lambda:'', pd.read_csv('labels_.tsv.gz', sep='\t', usecols=['pnode','label']).set_index('pnode').to_dict()['label'])

        self.map_P1696 = self.gen_relation('P1696')
        self.map_P1647 = self.gen_relation('P1647', False)
        self.map_P6609 = self.gen_relation('P6609', False)
        self.map_P1659 = self.gen_relation('P1659')

        with open(metadata_constraints) as fp:
            self.constraints = json.load(fp)

        self.query_size = query_size
        self.ranker = PropertyRanker()

    def _query(self, label, type_=None):

        # response1 = get(f'{self.host}/{label}?extra_info=true&language=en&item=property&size={self.query_size}', verify=False)
        response2 = get(f'{self.host}/{label[:10]}?extra_info=true&language=en&item=property&type=ngram&size={self.query_size}&instance_of=', verify=False)

        if type_:
            return [x['qnode'] for x in response2.json() if self.types[x['qnode']] == type_]

        return [x['qnode'] for x in response2.json()]

    def filter_by_set(self, s, l):
        r = [i for i in l if not i in s]
        r = list(set(r))
        for i in r:
            s.add(i)
        return s, r

    def gen_relation(self, label, twoway=True):

        pr = pd.read_csv('data/claims.properties.tsv.gz', sep='\t', usecols=['node1','label','node2'])
        pr = pr[pr['label'].apply(lambda x: x == label)].reset_index(drop=True)
        pr = pr[['node1', 'node2']]

        pr1 = pr.groupby('node1')['node2'].apply(list).reset_index()
        pr_dict = pr1.set_index('node1').to_dict()['node2']

        pr_dict_r = defaultdict(list)
        for k, v in pr_dict.items():
            for vi in v:
                pr_dict_r[k].append(vi)

        return pr_dict_r

    def get_candidates(self, name_, type_):

        result = self._query(name_, type_)

        ranked = {}
        loaded = set()
        loaded, ranked[1] = self.filter_by_set(loaded, result)

        ranked[2] = []
        for z in result:
            ranked[2] += self.map_P1696[z] + self.map_P1647[z] + self.map_P6609[z]
        loaded, ranked[2] = self.filter_by_set(loaded, ranked[2])

        ranked[3] = []
        for z in result:
            ranked[3] += self.map_P1659[z]
        loaded, ranked[3] = self.filter_by_set(loaded, ranked[3])

        if type_ is None:
            return ranked

        r = {}
        r[0] = []
        for i, L in enumerate(ranked):
            r[i+1] = []
            for e in ranked[L]:
                if self.types[e] == type_:
                    r[i+1].append(e)

        r[4] = []

        return r

    def tup(self, pnode):
        try:
            return pnode, self.labels[pnode]
        except KeyError:
            return None

    def generate_label(self, ranked):

        r = defaultdict(list)
        for k, pnodes in ranked.items():
            for pnode in pnodes:
                t = self.tup(pnode)
                if not t is None:
                    r[k].append(t)
        return r

    def filter_ranked(self, ranked, scope, constraint, otherProperties, minV, maxV):

        ranked = self.filter_by_item(ranked)

        if scope != 'both':
            ranked = self.filter_by_scope(ranked, scope)

        ranked = self.filter_by_allowed_qualifiers(ranked, constraint)
        ranked = self.filter_by_required_qualifiers(ranked, constraint)

        ranked = self.filter_by_conflicts(ranked, otherProperties)
        ranked = self.filter_by_range(ranked, minV, maxV)
        return ranked

    def filter_by_item(self, ranked):

        r = defaultdict(list)
        for k, pnodes in ranked.items():
            for pnode in pnodes:
                if pnode in self.constraints and 'noitem' in self.constraints[pnode]:
                    continue
                r[k].append(pnode)
        return r

    def filter_by_scope(self, ranked, scope='both'):

        r = defaultdict(list)
        for k, pnodes in ranked.items():
            for pnode in pnodes:

                if not pnode in self.constraints:
                    r[k].append(pnode)
                    continue

                info = self.constraints[pnode]
                if scope == 'qualifier':
                    if 'scope' in info and not 'Q' in info['scope']:
                        if 'scope_man' in info:
                            continue
                        r[4].append(pnode)
                        continue
                else:
                    if 'scope' in info and not 'V' in info['scope']:
                        if 'scope_man' in info:
                            continue
                        r[4].append(pnode)
                        continue
                r[k].append(pnode)

        return r

    def filter_by_allowed_qualifiers(self, ranked, constraint):


        if constraint is None:
            return ranked

        if not constraint in self.constraints:
            return ranked

        constr_dic = self.constraints[constraint]

        if not 'allowed_qualifiers' in constr_dic:
            return ranked

        r = defaultdict(list)
        for k, pnodes in ranked.items():
            for pnode in pnodes:
                if not pnode in constr_dic['allowed_qualifiers']:
                    continue
                r[k].append(pnode)

        return r

    def filter_by_range(self, ranked, minV, maxV):

        if minV is None and maxV is None:
            return ranked

        r = defaultdict(list)
        for k, pnodes in ranked.items():
            for pnode in pnodes:
                if pnode in self.constraints and 'min/max' in self.constraints[pnode]:
                    if not minV is None and not isnan(self.constraints[pnode]['min/max'][0]) and minV<self.constraints[pnode]['min/max'][0]:
                        r[4].append(pnode)
                        continue
                    if not maxV is None and not isnan(self.constraints[pnode]['min/max'][1]) and maxV>self.constraints[pnode]['min/max'][0]:
                        r[4].append(pnode)
                        continue
                r[k].append(pnode)

        return r

    def filter_by_required_qualifiers(self, ranked, constraint):

        if constraint is None:
            return ranked

        if not constraint in self.constraints:
            return ranked

        constr_dic = self.constraints[constraint]

        if not 'required_qualifiers' in constr_dic:
            return ranked

        r = defaultdict(list)
        for k, pnodes in ranked.items():
            for pnode in pnodes:
                if pnode in constr_dic['required_qualifiers']:
                    r[0].append(pnode)
                    continue
                r[k].append(pnode)

        return r

    def filter_by_conflicts(self, ranked, otherProperties):

        if otherProperties == '':
            return ranked

        properties = otherProperties.split(',')
        disallowed = set()
        for pnode in properties:
            if pnode in self.constraints and 'conflicts' in self.constraints[pnode]:
                for p in self.constraints[pnode]['conflicts']:
                    disallowed.add(p)

        r = defaultdict(list)
        for k, pnodes in ranked.items():
            for pnode in pnodes:
                if pnode in disallowed:
                    continue
                r[k].append(pnode)

        return r


    def find_property(self, label, type_=None, scope='both', filter=True, constraint=None, otherProperties='',
                        integerOnly=False, minV=None, maxV=None):

        if type_ in type_aliases:
            type_ = type_aliases[type_]

        candidates = self.get_candidates(label, type_)

        if filter:
            candidates = self.filter_ranked(candidates, scope, constraint, otherProperties, minV, maxV)

        for r in candidates:
            candidates[r] = self.ranker.rank_wlabel(candidates[r], label, scope=scope)

        return dict(sorted(candidates.items(), key=lambda x: x[0]))

    def generate_top_candidates(self, label, type_=None, scope='both', filter=True, constraint=None, otherProperties='',
                                integerOnly=False, minV=None, maxV=None, size=10):

        candidates = self.find_property(label, type_, scope, filter, constraint, otherProperties, integerOnly, minV, maxV)

        results = []
        for lv in candidates:
            for tup in candidates[lv]:
                results.append(tup)

        return results[:size]

    def find(self, label):

        type_ = request.args.get('type', None)
        scope = request.args.get('scope', 'both')

        constraint = request.args.get('constraint', None)
        otherProperties = request.args.get('otherProperties', '')
        minV = float(request.args.get('min', nan))
        maxV = float(request.args.get('max', nan))

        return self.find_property(label, type_, scope, filter=True, constraint=constraint,
                                    otherProperties=otherProperties, minV=minV, maxV=maxV)
