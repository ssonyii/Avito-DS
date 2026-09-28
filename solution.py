import os
import sys

# фиксируем хэширование строк до любых вычислений: перезапускаем интерпретатор с PYTHONHASHSEED=0
if os.environ.get('PYTHONHASHSEED') != '0':
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)
import numpy as np
import pandas as pd
import re
import pymorphy3
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import defaultdict
from sklearn.preprocessing import normalize
from gensim.models import FastText
import time
import multiprocessing as mp
import lightgbm as lgb

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('AVITO_DATA', os.path.join(
    BASE, 'NLP_avito_interns-dataset'))  # путь к parquet-файлам
CACHE = os.path.join(BASE, 'cache')  # промежуточные результаты
os.makedirs(CACHE, exist_ok=True)

# Общие утилиты: загрузка кэша, валидационный сплит, метрика.

SEED = 42
GKEY = ['search_query', 'search_location_id',
        'search_infm_params_text', 'search_category']


def load():
    items = pd.read_parquet(os.path.join(CACHE, 'items.parquet'))
    queries = pd.read_parquet(os.path.join(CACHE, 'queries.parquet'))
    train = pd.read_parquet(os.path.join(CACHE, 'train.parquet'))
    return items, queries, train


def make_val(train, items, n=3000):
    """Валидация, имитирующая бенчмарк: поисковые сессии из train, чьи выбранные объявления есть в корпусе.
    Возвращает (val_queries с колонкой targets, train без этих сессий)."""
    in_c = train.item_id.isin(set(items.item_id))
    g = train.groupby(GKEY, sort=False).ngroup()
    cand = np.unique(g[in_c])
    rng = np.random.RandomState(SEED)
    val_g = set(rng.choice(cand, size=n, replace=False))
    is_val = g.isin(val_g)
    v = train[is_val & in_c]
    vq = v.groupby(GKEY, sort=False).agg(q=('q', 'first'), qf=('qf', 'first'),
                                         targets=('item_id', lambda s: list(set(s)))).reset_index()
    vq['query_id'] = ['v%05d' % i for i in range(len(vq))]
    vq['search_is_delivery_search'] = 0
    return vq, train[~is_val].reset_index(drop=True)


def recall(preds, targets):
    return float(np.mean([len(set(p) & set(t)) / len(t) for p, t in zip(preds, targets)]))

# Шаг 1. Нормализация текстов: токенизация + лемматизация (pymorphy3)


TOKEN_RE = re.compile(r'[a-zа-я0-9]+')
morph = pymorphy3.MorphAnalyzer()
_lemma_cache = {}


def lemma(w):
    # кэш по слову: уникальных слов на порядки меньше, чем вхождений
    r = _lemma_cache.get(w)
    if r is None:
        r = morph.parse(w)[0].normal_form if not w.isdigit() else w
        _lemma_cache[w] = r
    return r


def norm(text, max_len=None):
    # Нижний регистр, ё->е, токены, леммы. Возвращает строку лемм через пробел.
    if not isinstance(text, str) or not text:
        return ''
    if max_len:
        text = text[:max_len]
    text = text.lower().replace('ё', 'е')
    return ' '.join(lemma(t) for t in TOKEN_RE.findall(text))


def clean_params(s):
    # Из параметров оставляем смысловую часть: служебные поля (время, дни недели, цены) — шум.
    if not isinstance(s, str):
        return ''
    s = re.sub(r'(Время|График)[^А-ЯЁ]*?(от|до|с) [\d:]+', ' ', s)
    s = re.sub(
        r'(Рабочие дни|Время для связи, дни недели|График работы, дни недели) \S+', ' ', s)
    s = re.sub(r'Признак [а-яё ]+ \d', ' ', s)
    s = re.sub(
        r'Место оказания услуг .*?(?= Тип| Вид| Направление| Опыт|$)', ' ', s)
    return s


def prep_main():
    items = pd.read_parquet(os.path.join(DATA, 'benchmark_items.parquet'))
    queries = pd.read_parquet(os.path.join(DATA, 'benchmark_queries.parquet'))
    train = pd.read_parquet(os.path.join(DATA, 'train.parquet'),
                            columns=['search_query', 'search_location_id', 'search_is_delivery_search',
                                     'search_infm_params_text', 'search_category', 'item_id',
                                     'item_location_id', 'item_microcat_id', 'item_title_raw',
                                     'item_latitude', 'item_longitude'])

    # Корпус: заголовок, параметры, описание (обрезаем до 3000 символов — хвосты описаний в основном SEO-спам)
    it = pd.DataFrame({'item_id': items.item_id})
    it['title'] = [norm(t) for t in items.item_title_raw]
    it['params'] = [norm(clean_params(t)) for t in items.item_infm_params_text]
    it['desc'] = [norm(t, 3000) for t in items.item_description_raw]
    for c in ['item_location_id', 'item_microcat_id', 'item_category_id', 'item_rating',
              'item_rating_reviews_count']:
        it[c] = items[c].values
    for c in ['item_latitude', 'item_longitude', 'item_price']:
        it[c] = pd.to_numeric(items[c], errors='coerce').values
    it['title_raw'] = items.item_title_raw.fillna(
        '').str.lower().str.replace('ё', 'е').values
    it['params_raw'] = items.item_infm_params_text.fillna('').values
    it.to_parquet(os.path.join(CACHE, 'items.parquet'))
    print('items done', len(_lemma_cache))

    q = queries.copy()
    q['q'] = [norm(t) for t in q.search_query]
    q['qf'] = [norm(t) for t in q.search_infm_params_text]
    q.to_parquet(os.path.join(CACHE, 'queries.parquet'))

    tr = train.copy()
    uq = pd.Series(tr.search_query.unique())
    mq = dict(zip(uq, [norm(t) for t in uq]))
    tr['q'] = tr.search_query.map(mq)
    uf = pd.Series(tr.search_infm_params_text.unique())
    mf = dict(zip(uf, [norm(t) for t in uf]))
    tr['qf'] = tr.search_infm_params_text.map(mf)
    ut = pd.Series(tr.item_title_raw.unique())
    mt = dict(zip(ut, [norm(t) for t in ut]))
    tr['title'] = tr.item_title_raw.map(mt)
    for c in ['item_latitude', 'item_longitude']:
        tr[c] = pd.to_numeric(tr[c], errors='coerce')
    tr.drop(columns=['item_title_raw']).to_parquet(
        os.path.join(CACHE, 'train.parquet'))
    print('train done')

# Индексы и признаки для пар (запрос, объявление).


class BM25:
    # BM25 на разреженной матрице: вес терма в документе считается один раз, скор запроса — сумма столбцов.

    def __init__(self, docs, k1=1.2, b=0.75, min_df=1):
        self.cv = CountVectorizer(
            token_pattern=r'\S+', min_df=min_df, dtype=np.float32)
        tf = self.cv.fit_transform(docs).tocsr()
        dl = np.asarray(tf.sum(1)).ravel()
        avg = dl.mean() if dl.mean() > 0 else 1.0
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        n = tf.shape[0]
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        denom = tf.data + k1 * (1 - b + b * dl[rows] / avg)
        tf.data = tf.data * (k1 + 1) / denom * self.idf[tf.indices]
        self.W = tf.tocsc()
        self.vocab = self.cv.vocabulary_

    def qvec(self, q):
        # Индексы термов запроса (уникальные, только известные словарю).
        return sorted({self.vocab[t] for t in q.split() if t in self.vocab})

    def score(self, q):
        cols = self.qvec(q)
        if not cols:
            return np.zeros(self.W.shape[0], np.float32), 0
        return np.asarray(self.W[:, cols].sum(1)).ravel(), len(cols)

    def coverage(self, q):
        """Доля термов запроса (по idf), найденных в документе."""
        cols = self.qvec(q)
        if not cols:
            return np.zeros(self.W.shape[0], np.float32)
        B = self.W[:, cols].copy()
        B.data[:] = 1
        w = self.idf[cols]
        return np.asarray(B @ w).ravel() / w.sum()


class CharTfidf:
    # Косинус по символьным n-граммам — устойчив к опечаткам и словоформам («ремон генераторов»).

    def __init__(self, docs):
        self.v = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2, sublinear_tf=True,
                                 dtype=np.float32, max_features=400000)
        self.X = self.v.fit_transform(docs).T.tocsr()  # term x doc

    def score_many(self, qs):
        Q = self.v.transform(qs)  # q x term
        return Q @ self.X  # q x doc, sparse


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class LocModel:
    ''' Модель локаций.
    - центр каждой location_id — медиана координат объявлений в ней (корпус + train);
    - для локации поиска без своих объявлений (регион, область) — медиана координат выбранных в train объявлений;
    - P(item_loc | search_loc) — эмпирическое распределение из train (ловит связи «область -> города области»).'''

    def __init__(self, items, train):
        pts = pd.concat([
            items[['item_location_id', 'item_latitude', 'item_longitude']],
            train[['item_location_id', 'item_latitude', 'item_longitude']]]).dropna()
        c = pts.groupby('item_location_id')[
            ['item_latitude', 'item_longitude']].median()
        s = train.dropna(subset=['item_latitude']).groupby('search_location_id')[
            ['item_latitude', 'item_longitude']].median()
        s.columns = c.columns
        c = pd.concat([c, s[~s.index.isin(c.index)]])
        self.center = c
        t = train.groupby(['search_location_id', 'item_location_id']).size()
        tot = train.groupby('search_location_id').size()
        p = t / tot.reindex(t.index.get_level_values(0)).values
        self.trans = {sl: g.droplevel(0).to_dict()
                      for sl, g in p.groupby(level=0)}
        self.search_n = tot.to_dict()
        # доля «своих» локаций по поисковой локации: для регионов она низкая
        same = train[train.search_location_id == train.item_location_id].groupby(
            'search_location_id').size()
        self.self_rate = (same / tot).fillna(0).to_dict()

    def trans_of(self, sl):
        """{item_loc: P(item_loc | search_loc)}"""
        return self.trans.get(sl, {})

    def query_center(self, loc):
        if loc in self.center.index:
            r = self.center.loc[loc]
            return r.iloc[0], r.iloc[1]
        return np.nan, np.nan

# Статистики, выученные по train (кликовые логи): априоры подкатегорий, популярность, похожие запросы.
# Строятся по части train, из которой исключены оцениваемые сессии (без утечки таргета).


def _dist(df, key, val):
    """{key: {val: доля}} — эмпирическое распределение val при фиксированном key."""
    c = df.groupby([key, val]).size()
    tot = c.groupby(level=0).transform('sum')
    p = (c / tot)
    out = defaultdict(dict)
    for (k, v), x in p.items():
        out[k][v] = x
    return out


class Stats:
    def __init__(self, tr, items, n_nbr=30):
        self.items = items
        # подкатегория (microcat) объявления по тексту запроса / фильтру / словам запроса
        self.mc_q = _dist(tr, 'q', 'item_microcat_id')
        self.mc_f = _dist(tr, 'qf', 'item_microcat_id')
        w = tr[['q', 'item_microcat_id']].copy()
        w['w'] = w.q.str.split()
        w = w.explode('w').dropna()
        self.mc_w = _dist(w, 'w', 'item_microcat_id')
        self.w_cnt = w.groupby('w').size().to_dict()
        # популярность и «память»: какие объявления выбирали по этому же запросу
        self.pop = tr.groupby('item_id').size().to_dict()
        self.item_q = tr.groupby(['q', 'item_id']).size().to_dict()
        self.item_qloc = tr.groupby(
            ['q', 'search_location_id', 'item_id']).size().to_dict()
        self.q_items = tr.groupby('q').item_id.apply(
            lambda s: list(set(s))).to_dict()
        # соседи-запросы: TF-IDF по леммам + символьным n-граммам уникальных запросов train
        uq = tr.groupby('q').size()
        self.uq = uq.index.values
        self.uq_cnt = uq.values.astype(np.float32)
        self.vw = TfidfVectorizer(
            token_pattern=r'\S+', sublinear_tf=True, dtype=np.float32)
        self.vc = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), sublinear_tf=True, min_df=2,
                                  dtype=np.float32)
        Xw = self.vw.fit_transform(self.uq)
        Xc = self.vc.fit_transform(self.uq)
        self.QX = normalize(sp.hstack([Xw, Xc]).tocsr()).T.tocsr()
        self.n_nbr = n_nbr
        # для соседей: распределения microcat и «мешок лемм заголовков» выбранных объявлений
        self.uq_idx = {q: i for i, q in enumerate(self.uq)}
        mc = tr.groupby(['q', 'item_microcat_id']).size().reset_index(name='n')
        self.mc_ids, mc_codes = np.unique(np.concatenate([mc.item_microcat_id.values,
                                                          items.item_microcat_id.values]), return_inverse=True)
        mc_codes = mc_codes[:len(mc)]
        self.mc_index = {m: i for i, m in enumerate(self.mc_ids)}
        rows = mc.q.map(self.uq_idx).values
        M = sp.csr_matrix((mc.n.values.astype(np.float32), (rows, mc_codes)),
                          shape=(len(self.uq), len(self.mc_ids)))
        self.QM = normalize(M, norm='l1')
        # мешок лемм заголовков выбранных объявлений (для расширения запроса)
        self.tv = TfidfVectorizer(
            token_pattern=r'\S+', sublinear_tf=True, min_df=2, dtype=np.float32)
        tt = tr.groupby('q').title.apply(' '.join).reindex(self.uq)
        self.QT = normalize(self.tv.fit_transform(tt.values)).tocsr()
        self.item_mc_code = np.array(
            [self.mc_index[m] for m in items.item_microcat_id.values])
        # заголовки корпуса в том же словаре — для скоринга расширенного запроса
        self.IT = normalize(self.tv.transform(items.title)).tocsr()
        # запрос train -> выбранные объявления корпуса (для «коллаборативной памяти» по соседям)
        pos = {x: j for j, x in enumerate(items.item_id.values)}
        qi = tr[['q', 'item_id']].assign(
            p=tr.item_id.map(pos)).dropna(subset=['p'])
        self.QI = sp.csr_matrix((np.ones(len(qi), np.float32), (qi.q.map(self.uq_idx).values, qi.p.astype(int).values)),
                                shape=(len(self.uq), len(items)))

    def neighbors(self, qs):
        # Для пачки запросов: соседи в train (индексы уникальных запросов и сходства).
        Qv = normalize(
            sp.hstack([self.vw.transform(qs), self.vc.transform(qs)]).tocsr())
        S = (Qv @ self.QX).tocsr()
        res = []
        for i in range(S.shape[0]):
            row = S.getrow(i)
            idx, val = row.indices, row.data
            if len(idx) > self.n_nbr:
                top = np.argpartition(-val, self.n_nbr)[:self.n_nbr]
                idx, val = idx[top], val[top]
            res.append((idx, val))
        return res

    def mc_prior(self, q, qf, nbr):
        # Вероятности подкатегорий для запроса из 4 источников; возвращает массивы по объявлениям корпуса.
        n = len(self.mc_ids)
        # точное совпадение текста запроса
        p_q = np.zeros(n, np.float32)
        for m, x in self.mc_q.get(q, {}).items():
            p_q[self.mc_index[m]] = x
        # по фильтру
        p_f = np.zeros(n, np.float32)
        if qf:
            for m, x in self.mc_f.get(qf, {}).items():
                p_f[self.mc_index[m]] = x
        # по словам: среднее распределений, взвешенное по log(частоты слова)
        p_w = np.zeros(n, np.float32)
        tot = 0.0
        # sorted: фиксированный порядок суммирования float -> воспроизводимость
        for w in sorted(set(q.split())):
            d = self.mc_w.get(w)
            if d:
                # редкие слова информативнее
                wt = 1.0 / np.log(2 + self.w_cnt[w])
                for m, x in d.items():
                    p_w[self.mc_index[m]] += wt * x
                tot += wt
        if tot:
            p_w /= tot
        # по соседним запросам (взвешено сходством)
        idx, val = nbr
        p_n = np.zeros(n, np.float32)
        if len(idx):
            wv = val ** 2
            p_n = (sp.csr_matrix(wv / wv.sum()) @
                   self.QM[idx]).toarray().ravel()
        c = self.item_mc_code
        return p_q[c], p_f[c], p_w[c], p_n[c]

    def neighbor_items(self, nbr):
        # Сколько раз объявление выбирали по похожим запросам train (с весом сходства).
        idx, val = nbr
        if not len(idx):
            return np.zeros(self.QI.shape[1], np.float32)
        return (sp.csr_matrix(val ** 2) @ self.QI[idx]).toarray().ravel()

    def expansion_score(self, nbr):
        ''' Расширение запроса: косинус заголовка объявления с взвешенным центроидом заголовков,
        выбранных по соседним запросам train (работает как выученный «перевод» запрос -> заголовок).'''
        idx, val = nbr
        if not len(idx):
            return np.zeros(self.IT.shape[0], np.float32)
        v = normalize(sp.csr_matrix(val ** 2) @ self.QT[idx])
        return (self.IT @ v.T).toarray().ravel()

# Семантические эмбеддинги без внешних моделей: FastText (gensim), обученный локально на текстах
# корпуса и train. Субсловные n-граммы дают устойчивость к опечаткам, контексты описаний — синонимию
# («парикмахер» ~ «окрашивание волос»). Эмбеддинг текста — среднее векторов слов с весом idf.


DIM = 128


def train_ft(items, train):
    path = os.path.join(CACHE, 'ft.model')
    if os.path.exists(path):
        return FastText.load(path)
    # тексты: заголовок+описание объявлений корпуса, а также пары «запрос + заголовок» из train —
    # последние сближают формулировки запросов с формулировками объявлений
    sents = [(t + ' ' + ' '.join(d.split()[:100])).split()
             for t, d in zip(items.title, items.desc)]
    # пары с объявлениями из корпуса исключаем: на них строится валидация (иначе утечка таргета)
    pairs = train.loc[~train.item_id.isin(set(items.item_id)), [
        'q', 'title']].drop_duplicates()
    sents += [(q + ' ' + t).split() for q, t in zip(pairs.q, pairs.title)]
    m = FastText(vector_size=DIM, window=5, min_count=3, sg=1, epochs=5, min_n=3, max_n=5,
                 # workers=1 — ради детерминизма (воспроизводимость ответа)
                 workers=1, seed=SEED)
    m.build_vocab(corpus_iterable=sents)
    m.train(corpus_iterable=sents, total_examples=len(sents), epochs=m.epochs)
    m.save(path)
    return m


class Embedder:
    def __init__(self, model, docs_for_idf):
        self.m = model
        df = {}
        for d in docs_for_idf:
            for w in set(d.split()):
                df[w] = df.get(w, 0) + 1
        n = len(docs_for_idf)
        self.idf = {w: np.log(n / c) for w, c in df.items()}
        self.default_idf = np.log(n)

    def encode(self, texts):
        out = np.zeros((len(texts), DIM), np.float32)
        for i, t in enumerate(texts):
            ws = t.split()
            if not ws:
                continue
            w = np.array([self.idf.get(x, self.default_idf)
                         for x in ws], np.float32)
            v = (self.m.wv[ws] * w[:, None]).sum(0)
            nrm = np.linalg.norm(v)
            out[i] = v / nrm if nrm > 0 else v
        return out


def emb_main():
    items, queries, train = load()
    m = train_ft(items, train)
    e = Embedder(m, items.title.tolist())
    np.save(os.path.join(CACHE, 'emb_title.npy'),
            e.encode(items.title.tolist()))
    # описание: первые 60 слов — там обычно суть услуги
    np.save(os.path.join(CACHE, 'emb_desc.npy'),
            e.encode([' '.join(d.split()[:60]) for d in items.desc]))
    print('done')

# Шаг 2. Кандидатогенерация + признаки для пар (запрос, объявление).
# Схема:
# 1) по каждому запросу считаем по всему корпусу несколько дешёвых скоров
# (BM25 по полям, априор подкатегории, расширение запроса, расстояние);
# 2) объединяем топы этих скоров в пул кандидатов (~740 на запрос);
# 3) для пула считаем полный набор признаков -> дальше LightGBM (train_rank.py).
#
# Разметка для обучения/валидации: сессии train, чьи выбранные объявления есть в корпусе
# (по сути такие же пары, как в бенчмарке). Они делятся на 5 фолдов; признаки для фолда k
# считаются по статистикам train БЕЗ сессий фолда k, чтобы не было утечки таргета.


N_FOLDS = 5
SESS_PER_Q = 2  # сессий на уникальный запрос (см. fold_split)
P_UNSEEN = 0.45  # доля запросов, превращаемых в «новые» (см. fold_split)
TOP_EACH = 250  # топ по каждому источнику кандидатов

G = {}  # глобальные объекты для воркеров (fork)


def build_static(items):
    # Индексы, не зависящие от train: BM25 по полям, символьный TF-IDF по заголовкам.
    t = time.time()
    G['items'] = items
    G['bt'] = BM25(items.title)
    G['bd'] = BM25(items.desc)
    G['bp'] = BM25(items.params)
    G['ch'] = CharTfidf(items.title)
    G['ilat'] = items.item_latitude.values
    G['ilon'] = items.item_longitude.values
    G['iloc'] = items.item_location_id.values
    G['loc_cnt'] = items.item_location_id.value_counts().to_dict()
    G['params_raw'] = items.params_raw.values
    G['item_ids'] = items.item_id.values
    G['rating'] = items.item_rating.values
    G['nrev'] = items.item_rating_reviews_count.values
    G['price'] = items.item_price.values
    G['desc_len'] = items.desc.str.len().values
    G['filter_cache'] = {}
    G['emb_t'] = np.load(os.path.join(CACHE, 'emb_title.npy'))
    G['emb_d'] = np.load(os.path.join(CACHE, 'emb_desc.npy'))
    G['emb'] = Embedder(FastText.load(os.path.join(
        CACHE, 'ft.model')), items.title.tolist())
    print('static built %.0fs' % (time.time() - t), flush=True)


FILTER_KEYS = r'(?= Тип услуги| Вид услуги| Кто оказывает услуги| Онлайн-запись| Рейтинг пользователя|$)'


def filter_match(qf_raw, key):
    """Совпадает ли фильтр поиска (напр. «Вид услуги Ремонт и отделка») с параметрами объявления.
    NaN — фильтра нет."""
    m = re.search(key + r' (.+?)' + FILTER_KEYS, qf_raw or '')
    if not m:
        return None
    s = key + ' ' + m.group(1)
    fc = G['filter_cache']
    if s not in fc:
        fc[s] = np.array([s in p for p in G['params_raw']], dtype=np.float32)
    return fc[s]


def topk(x, k):
    k = min(k, len(x) - 1)
    return np.argpartition(-x, k)[:k]


def process(i):
    # Кандидаты и признаки для одного запроса.
    r = G['Q'].iloc[i]
    st, loc, nbrs = G['stats'], G['loc'], G['nbrs']
    q = r.q
    # текстовые скоры по всему корпусу
    bt, _ = G['bt'].score(q)
    bd, _ = G['bd'].score(q)
    bp, _ = G['bp'].score(q)
    text = bt + 0.3 * bd + 0.3 * bp
    ch = G['ch_rows'][i].toarray().ravel()
    exp = st.expansion_score(nbrs[i])
    qv = G['qemb'][i]
    e_t = G['emb_t'] @ qv  # косинус FastText-эмбеддингов запроса и заголовка
    e_d = G['emb_d'] @ qv  # и начала описания
    p_q, p_f, p_w, p_n = st.mc_prior(q, r.qf, nbrs[i])
    nb_it = st.neighbor_items(nbrs[i])
    fv = filter_match(r.search_infm_params_text, 'Вид услуги')
    ft = filter_match(r.search_infm_params_text, 'Тип услуги')
    filt = (fv if fv is not None else 0) + (ft if ft is not None else 0)
    # гео
    sl = r.search_location_id
    same = (G['iloc'] == sl).astype(np.float32)
    la, lo = loc.query_center(sl)
    dist = haversine(la, lo, G['ilat'], G['ilon'])
    ld = np.log1p(np.nan_to_num(dist, nan=5000.0))
    geo = -2 * ld + 3 * same
    # «переходы» локаций из train: поиск по области -> объявления в городах области
    tr_p = pd.Series(G['iloc']).map(loc.trans_of(sl)).fillna(0).values
    geo_t = np.log(tr_p + 1e-3) + 3 * same
    # кандидаты: объединение топов нескольких источников
    mc = p_q + p_n + 0.5 * p_w
    srcs = [text + geo, text - ld + 3 * same, 8 * mc + 0.2 * text + geo,
            8 * exp + 0.2 * text + geo, 8 * mc + 8 * exp + text + geo,
            10 * (e_t + 0.5 * e_d) + 0.2 * text + geo,
            # семантика (эмбеддинги + априор подкатегории + фильтры) без требования совпадения слов
            8 * mc + 8 * e_t + 3 * filt + geo,
            10 * mc + 5 * e_t + 0.3 * text + 3 * filt + geo_t,
            # объявления, которые выбирали по похожим запросам
            3 * np.log1p(nb_it) + 0.2 * text + 5 * mc + geo]
    cand = set()
    for s in srcs:
        cand.update(topk(s, TOP_EACH).tolist())
    # объявления, выбранные по этому же запросу в train («память»)
    idmap = G['id2pos']
    for it in st.q_items.get(q, []):
        j = idmap.get(it)
        if j is not None:
            cand.add(j)
    c = np.fromiter(cand, dtype=np.int64)
    # признаки кандидатов
    f = pd.DataFrame({
        'qi': i, 'pos': c,
        'bm_t': bt[c], 'bm_d': bd[c], 'bm_p': bp[c], 'text': text[c],
        'cov_t': G['bt'].coverage(q)[c], 'cov_d': G['bd'].coverage(q)[c],
        'ch_t': ch[c], 'exp': exp[c], 'e_t': e_t[c], 'e_d': e_d[c],
        'p_q': p_q[c], 'p_f': p_f[c], 'p_w': p_w[c], 'p_n': p_n[c],
        'same': same[c], 'dist': dist[c],
        'trans': tr_p[c], 'nb_it': nb_it[c],
        'rating': G['rating'][c], 'nrev': G['nrev'][c], 'price': G['price'][c],
        'desc_len': G['desc_len'][c],
    })
    f['f_vid'] = fv[c] if fv is not None else np.nan
    f['f_tip'] = ft[c] if ft is not None else np.nan
    ids = G['item_ids'][c]
    f['pop'] = [st.pop.get(x, 0) for x in ids]
    f['iq'] = [st.item_q.get((q, x), 0) for x in ids]
    f['iqloc'] = [st.item_qloc.get((q, sl, x), 0) for x in ids]
    # признаки запроса
    f['q_len'] = len(q.split())
    f['q_has_f'] = int(bool(r.qf))
    f['q_cat0'] = int(r.search_category == 0)
    f['q_self_rate'] = loc.self_rate.get(sl, np.nan)
    f['q_loc_items'] = G['loc_cnt'].get(sl, 0)
    f['q_seen'] = int(q in st.q_items)
    f['q_nbr_sim'] = nbrs[i][1].max() if len(nbrs[i][1]) else 0.0
    # относительные признаки внутри запроса
    for col in ['text', 'exp', 'ch_t', 'p_n', 'e_t']:
        m = f[col].max()
        f[col + '_rel'] = f[col] / m if m > 0 else 0.0
    f['base_rank'] = (-(f['text'] - 2 *
                      np.log1p(f['dist'].fillna(5000)) + 3 * f['same'])).rank()
    if 'targets' in r and isinstance(r.targets, (list, np.ndarray)):
        t = set(r.targets)
        f['label'] = [int(x in t) for x in ids]
    return f


def run_block(Q, stats, loc, workers=8):
    # Признаки для набора запросов Q при заданных статистиках.
    G['Q'] = Q.reset_index(drop=True)
    G['stats'], G['loc'] = stats, loc
    G['nbrs'] = stats.neighbors(G['Q'].q.tolist())
    G['ch_rows'] = G['ch'].score_many(G['Q'].q.tolist()).tocsr()
    G['qemb'] = G['emb'].encode(G['Q'].q.tolist())
    ctx = mp.get_context('fork')
    with ctx.Pool(workers) as pool:
        parts = pool.map(process, range(len(G['Q'])), chunksize=16)
    return pd.concat(parts, ignore_index=True)


def session_table(train, items):
    """Сессии train с объявлениями из корпуса. Фолд назначается по тексту запроса,
    чтобы один и тот же запрос не попадал и в оценку, и в обучение ранкера."""
    in_c = train.item_id.isin(set(items.item_id))
    g = train.groupby(GKEY, sort=False).ngroup()
    train = train.assign(g=g.values)
    pos = train[in_c]
    S = pos.groupby('g').agg(**{k: (k, 'first') for k in GKEY}, q=('q', 'first'), qf=('qf', 'first'),
                             targets=('item_id', lambda s: sorted(set(s)))).reset_index()
    rng = np.random.RandomState(SEED)
    uq = S.q.unique()
    fold_of = dict(zip(uq, rng.permutation(len(uq)) % N_FOLDS))
    S['fold'] = S.q.map(fold_of)
    return train, S


def fold_split(train, S, k):
    # Запросы фолда k и train для статистик.
    rng = np.random.RandomState(SEED + k)
    Sk = S[S.fold == k].sample(frac=1, random_state=SEED + k)
    Sk = Sk.assign(n=Sk.groupby('q').cumcount())
    Qk = Sk[Sk.n < SESS_PER_Q].sort_values(
        'n', kind='stable').reset_index(drop=True)
    Qk['main'] = (Qk.n == 0).astype(int)
    uq = Qk.q.unique()
    unseen = set(uq[rng.rand(len(uq)) < P_UNSEEN])
    drop = train.g.isin(set(Qk.g)) | train.q.isin(unseen)
    return Qk.drop(columns='n'), train[~drop]


def feats_for(Q, stats, loc):
    out = []
    for s in range(0, len(Q), 3000):
        out.append(run_block(Q.iloc[s:s + 3000], stats, loc))
        out[-1]['qi'] += s
    return pd.concat(out, ignore_index=True)


def pipeline_main():
    items, queries, train = load()
    build_static(items)
    G['id2pos'] = {x: i for i, x in enumerate(items.item_id.values)}
    train, S = session_table(train, items)
    S.to_parquet(os.path.join(CACHE, 'sessions.parquet'))
    for k in range(N_FOLDS):
        path = os.path.join(CACHE, 'feats_fold%d.parquet' % k)
        if os.path.exists(path):
            continue
        t = time.time()
        # фолд 0 — валидация, остальные — обучение ранкера
        Qk, tr_k = fold_split(train, S, k)
        stats, loc = Stats(tr_k, items), LocModel(items, tr_k)
        F = feats_for(Qk, stats, loc)
        F.to_parquet(path)
        Qk.to_parquet(os.path.join(CACHE, 'q_fold%d.parquet' % k))
        rec = F.groupby('qi').label.sum() / Qk.targets.map(len)
        print('fold %d: %d q, seen=%.2f, %d rows, cand/q=%.0f, cand recall=%.4f, %.0fs' % (
            k, len(Qk), Qk.q[Qk.main == 1].isin(set(tr_k.q)).mean(), len(F), len(F) / len(Qk), rec.reindex(range(len(Qk))).fillna(0)[Qk.main == 1].mean(), time.time() - t),
            flush=True)
    path = os.path.join(CACHE, 'feats_bench.parquet')
    if not os.path.exists(path):
        stats, loc = Stats(train, items), LocModel(items, train)
        F = feats_for(queries, stats, loc)
        F.to_parquet(path)
        print('bench: %d rows' % len(F), flush=True)

# Шаг 3. LightGBM-ранкер по пулу кандидатов, валидация на фолде 0, ответ для бенчмарка.
# python train_rank.py val — обучение на фолдах 1..4, Recall@50 на фолде 0 (+ разбивка ошибок)
# python train_rank.py final — обучение на всех фолдах, запись answer.csv


DROP = {'qi', 'pos', 'label', 'fold'}
PARAMS = dict(objective='lambdarank', metric='ndcg', eval_at=[50], lambdarank_truncation_level=60,
              learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=SEED, deterministic=True,
              num_threads=8, verbose=-1)
N_ROUNDS = 600
POP_DROP = 0.2  # см. debias_pop


def debias_pop(F, seed):
    # Коррекция смещения отбора. Позитивы валидации — объявления, попавшие в корпус, а туда чаще попадают популярные.
    rng = np.random.RandomState(seed)
    m = (F.label.values == 1) & (rng.rand(len(F)) < POP_DROP)
    F.loc[m, ['pop', 'iq', 'iqloc']] = 0
    return F


def load_fold(k):
    F = pd.read_parquet(os.path.join(CACHE, 'feats_fold%d.parquet' % k))
    Q = pd.read_parquet(os.path.join(CACHE, 'q_fold%d.parquet' % k))
    return F, Q


RANK_COLS = ['text', 'e_t', 'e_d', 'exp', 'p_n',
             'nb_it', 'cov_d', 'pop', 'nrev', 'trans', 'ch_t']


def add_rank_feats(F):
    """Ранг кандидата внутри запроса по ключевым скорам: масштаб скоров сильно плавает между запросами,
    ранги сравнимы (+0.3 п.п. Recall@50 на валидации)."""
    g = F.groupby('qi')
    for c in RANK_COLS:
        F['rk_' + c] = g[c].rank(ascending=False,
                                 method='min').astype(np.float32)
    F['rk_dist'] = g['dist'].rank(method='min').astype(np.float32)
    return F


def prep_train(Fs):
    # Склейка фолдов; запросы без позитивов в пуле для ранжирования бесполезны — выкидываем.
    parts = []
    for k, F in Fs:
        F = debias_pop(F.copy(), SEED + k)
        F['qi'] = F['qi'] + k * 100000  # уникальный id запроса между фолдами
        parts.append(F)
    F = pd.concat(parts, ignore_index=True)
    has_pos = F.groupby('qi').label.transform('max') > 0
    F = F[has_pos].sort_values('qi', kind='stable')
    return add_rank_feats(F)


def fit(F):
    cols = [c for c in F.columns if c not in DROP]
    grp = F.groupby('qi', sort=False).size().values
    ds = lgb.Dataset(F[cols], F.label, group=grp)
    return lgb.train(PARAMS, ds, N_ROUNDS), cols


def top50(F, score, item_ids, item_cat):
    # Для каждого запроса — 50 лучших item_id по скору ранкера.
    score = score - 100.0 * (item_cat[F.pos.values] != 114)
    F = F.assign(s=score).sort_values(
        ['qi', 's'], ascending=[True, False], kind='stable')
    F = F.groupby('qi', sort=False).head(50)
    return F.groupby('qi').pos.apply(lambda p: list(item_ids[p.values]))


def recall_at50(pred, Q):
    r = []
    for i, t in enumerate(Q.targets):
        p = set(pred.get(i, []))
        r.append(len(p & set(t)) / len(t))
    return np.array(r)


def rank_main(mode):
    items, queries, _ = load()
    item_ids = items.item_id.values
    item_cat = items.item_category_id.values
    if mode == 'val':
        tr = prep_train([(k, load_fold(k)[0]) for k in range(1, N_FOLDS)])
        model, cols = fit(tr)
        F0, Q0 = load_fold(0)
        F0 = debias_pop(F0, SEED)
        # оценка — только по основной сессии каждого запроса (одна сессия на уникальный запрос, как в бенчмарке)
        main = np.flatnonzero(Q0.main.values == 1)
        F0 = F0[F0.qi.isin(main)]
        Q0 = Q0.iloc[main].reset_index(drop=True)
        F0['qi'] = F0.qi.map({old: new for new, old in enumerate(main)})
        F0 = add_rank_feats(F0)
        pred = top50(F0, model.predict(
            F0[cols], num_threads=8), item_ids, item_cat)
        r = recall_at50(pred, Q0)
        base = top50(F0, -F0.base_rank.values, item_ids, item_cat)
        print('Recall@50 ranker=%.4f  baseline(BM25+geo)=%.4f  pool=%.4f' % (
            r.mean(), recall_at50(base, Q0).mean(),
            (F0.groupby('qi').label.sum() / Q0.targets.map(len)).reindex(range(len(Q0))).fillna(0).mean()))
        # разбивка по типам запросов
        seen = F0.groupby('qi').q_seen.first().reindex(range(len(Q0))).values
        loc0 = F0.groupby('qi').q_loc_items.first().reindex(
            range(len(Q0))).values
        print('seen: %.4f  unseen: %.4f' %
              (r[seen == 1].mean(), r[seen == 0].mean()))
        print('loc without items: %.4f  big city(>5000): %.4f  other: %.4f' % (
            r[loc0 == 0].mean(), r[loc0 > 5000].mean(), r[(loc0 > 0) & (loc0 <= 5000)].mean()))
        imp = pd.Series(model.feature_importance('gain'),
                        cols).sort_values(ascending=False)
        print((imp / imp.sum()).round(4).to_string())
        Q0.assign(recall=r).to_parquet(
            os.path.join(CACHE, 'val_result.parquet'))
    else:
        tr = prep_train([(k, load_fold(k)[0]) for k in range(N_FOLDS)])
        model, cols = fit(tr)
        FB = add_rank_feats(pd.read_parquet(
            os.path.join(CACHE, 'feats_bench.parquet')))
        pred = top50(FB, model.predict(
            FB[cols], num_threads=8), item_ids, item_cat)
        answer = pd.DataFrame({
            'query_id': queries.query_id.values,
            'answer': [' '.join(pred.get(i, [])) for i in range(len(queries))],
        })
        out = os.path.join(BASE, 'answer.csv')
        answer.to_csv(out, index=False)
        print('saved', out)

# Проверка answer.csv на соответствие требованиям формата.


def check_answer():
    path = os.path.join(BASE, 'answer.csv')

    a = pd.read_csv(path, dtype=str, keep_default_na=False)
    q = pd.read_parquet(os.path.join(
        DATA, 'benchmark_queries.parquet'), columns=['query_id'])
    items = set(pd.read_parquet(os.path.join(
        DATA, 'benchmark_items.parquet'), columns=['item_id']).item_id)

    assert list(a.columns) == ['query_id', 'answer'], a.columns
    assert a.query_id.is_unique and set(
        a.query_id) == set(q.query_id) and len(a) == len(q)
    assert a.query_id.str.len().eq(16).all()
    lens = []
    for s in a.answer:
        ids = s.split(' ')
        assert s == s.strip() and '  ' not in s, 'лишние пробелы'
        assert len(ids) <= 50 and len(ids) == len(
            set(ids)), 'больше 50 или повторы'
        assert all(re.fullmatch(r'[0-9a-f]{16}', x)
                   for x in ids), 'формат item_id'
        assert all(x in items for x in ids), 'item_id нет в корпусе'
        lens.append(len(ids))
    print('OK: %d строк, item_id в строке: min=%d, mean=%.1f' %
          (len(a), min(lens), sum(lens) / len(lens)))

# Запуск всех этапов


if __name__ == '__main__':
    prep_main()  # лемматизация текстов
    emb_main()  # FastText-эмбеддинги
    pipeline_main()  # кандидаты + признаки
    rank_main('val')  # Recall@50 на отложенном фолде
    rank_main('final')  # обучение на всех фолдах -> answer.csv
    check_answer()  # проверка формата
