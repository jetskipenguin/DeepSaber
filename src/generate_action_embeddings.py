'''
This script is an adaption of the original create_action_embeddings notebook in src/notebooks
'''
from typing import Tuple
import pandas as pd
from pathlib import Path
from typing import Tuple, Any
from itertools import product
from importlib import reload
import logging
import gensim
from bayes_opt import BayesianOptimization

def load_datasets(storage_folder) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return [pd.read_pickle(storage_folder / f'{phase}_beatmaps.pkl') for phase in
            ['train', 'val', 'test']]

def generate_action_embeddings():
    storage_folder = Path('../data/new_datasets')
    train, val, test = load_datasets(storage_folder)

    def create_sentence(x):
        x.name = 'word'
        x = x.reset_index('time')
        x = x.drop_duplicates('time')
        x = x.word.str.cat(sep=' ')
        return x

    def word_df2text(df: pd.DataFrame):
        return df.droplevel(2).word.groupby(['name', 'difficulty']).apply(create_sentence).str.cat(sep='\n')

    train_text = word_df2text(train)
    val_text = word_df2text(val)

    with open(storage_folder / 'train_text.cor', 'w') as wf:
        wf.write(train_text)
    with open(storage_folder / 'val_text.cor', 'w') as wf:
        wf.write(val_text)

    def subword_tuple2string(tuple: Tuple[Any]):
        return ''.join([str(x) for x in tuple])

    def word_tuple2string(tuple: Tuple[Any]):
        for subword_id in range(0, len(tuple), 4):
            if not (tuple[subword_id + 0] in 'LR'
                    and 0 <= tuple[subword_id + 1] < 3
                    and 0 <= tuple[subword_id + 2] < 4
                    and 0 <= tuple[subword_id + 3] < 9):
                return None
        return '_'.join(
            [subword_tuple2string(tuple[subword_id:subword_id + 4]) for subword_id in range(0, len(tuple), 4)])
    
    

    def add_valid_translations_one_hand(lines, translation):
        def create_word_tuples(hand, position, rotation, translation):
            tuple_from = hand, *position, rotation
            tuple_to = [a + b for a, b in zip(tuple_from, translation)]
            return tuple_from, tuple_to

        add_valid_translation(lines, translation, create_word_tuples)
        return lines


    def add_valid_translations_one_hand_doublebeat(lines, translation, rotation):
        def create_word_tuples_multiplication(hand, position, rotation, translation):
            tuple_from = hand, *position, rotation
            tuple_to = *tuple_from, *[a + b for a, b in zip(tuple_from, translation)]
            return tuple_from, tuple_to

        def create_word_tuples_demultiplication(hand, position, rotation, translation):
            return create_word_tuples_multiplication(hand, position, rotation, translation)[::-1]

        for hand in 'LR':
            add_valid_translation_only_positions(lines, hand, rotation, translation, create_word_tuples_multiplication)
            add_valid_translation_only_positions(lines, hand, rotation, translation, create_word_tuples_demultiplication)
        return lines


    def add_valid_translations_two_hand(lines, translation, hand_translation):
        def create_word_tuples(hand, position, rotation, translation):
            left_from = hand, *position, rotation
            right_from = 'R', *position, rotation
            tuple_from = *left_from, *[a + b for a, b in zip(right_from, hand_translation)]
            tuple_to = [a + b for a, b in zip(tuple_from, translation * 2)]
            return tuple_from, tuple_to

        for hand, rotation in product('L', range(-8, 9)):
            add_valid_translation_only_positions(lines, hand, rotation, translation, create_word_tuples)
        return lines


    def add_valid_translations_one_hand_demultiply(lines, translation, rotation):
        def create_word_tuples(hand, position, rotation, translation):
            tuple_to = hand, *position, rotation
            tuple_from = *tuple_to, *[a + b for a, b in zip(tuple_to, translation)]
            return tuple_from, tuple_to

        for hand in 'LR':
            add_valid_translation_only_positions(lines, hand, rotation, translation, create_word_tuples)
        return lines


    def add_valid_translation(lines, translation, create_word_tuples):
        for hand, rotation in product('LR', range(9)):
            add_valid_translation_only_positions(lines, hand, rotation, translation, create_word_tuples)


    def add_valid_translation_only_positions(lines, hand, rotation, translation, create_word_tuples):
        for example_position in product(range(3), range(4)):
            example_from, example_to = create_word_tuples(hand, example_position, rotation, translation)

            for question_position in product(range(3), range(4)):
                question_from, question_to = create_word_tuples(hand, question_position, rotation, translation)
                add_valid_analogy(lines, example_from, example_to, question_from, question_to)


    def add_valid_analogy(lines, example_from, example_to, question_from, question_to):
        """Add analogy to `lines` if all words are valid"""
        if example_from != question_from:  # skip identical analogies
            analogy = [word_tuple2string(x) for x in [example_from, example_to, question_from, question_to]]
            if None not in analogy:
                lines.append(' '.join(analogy))

    
    reload(logging)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


    def create_analogies(path: Path):
        """
        Shape of a beat element is [hand][y][x][rotation]. A word is of shape [beat-element][_beat-element]*.
        """
        lines = []

        def new_section(name: str):
            logging.info(f'Generating analogies for section {name}')
            lines.append(f': {name}')

        new_section('single-hand-translation-x')
        for dx in range(-3, 4):
            if dx == 0:  # remove identity translation
                continue
            translation = '', 0, dx, 0
            add_valid_translations_one_hand(lines, translation)

        new_section('single-hand-translation-y')
        for dy in range(-2, 3):
            if dy == 0:  # remove identity translation
                continue
            translation = '', dy, 0, 0
            add_valid_translations_one_hand(lines, translation)

        new_section('single-hand-translation-other')
        for dydx in product(range(-2, 3), range(-3, 4)):
            if 0 in dydx:  # previously used translations
                continue
            translation = '', *dydx, 0

            add_valid_translations_one_hand(lines, translation)

        new_section('single-hand-rotation')
        for rotation_change in range(-8, 9):
            if rotation_change == 0:  # remove identity rotation change
                continue
            translation = '', 0, 0, rotation_change
            add_valid_translations_one_hand(lines, translation)

        new_section('single-hand-hand-swap')
        for hand_id in range(2):
            hand_from = 'LR'[hand_id]
            hand_to = 'RL'[hand_id]

            for example_position in product(range(3), range(4), range(9)):
                example_from = hand_from, *example_position
                example_to = hand_to, *example_position
                for qustion_position in product(range(3), range(4), range(9)):
                    question_from = hand_from, *qustion_position
                    question_to = hand_to, *qustion_position
                    add_valid_analogy(lines, example_from, example_to, question_from, question_to)

        new_section('single-hand-doublebeat-y')
        for dy, rotation in product(range(-2, 3), [0, 1]):
            if dy == 0:  # remove identity multiplication
                continue
            translation = '', dy, 0, 0
            add_valid_translations_one_hand_doublebeat(lines, translation, rotation)

        new_section('single-hand-doublebeat-x')
        for dx, rotation in product(range(-3, 4), [2, 3]):
            if dx == 0:  # remove identity multiplication
                continue
            translation = '', 0, dx, 0
            add_valid_translations_one_hand_doublebeat(lines, translation, rotation)

        new_section('single-hand-doublebeat-other')
        for dx, dy, rotation in product(range(-2, 3), range(-3, 4), range(9)):
            if 0 in dydx:  # previously used multiplication
                continue
            translation = '', dy, dx, 0
            add_valid_translations_one_hand_doublebeat(lines, translation, rotation)

        new_section('double-hand-translation-x')
        for dx, right_dx, right_drotation in product(range(-3, 4), range(-3, 4), range(-8, 9)):
            if dx == 0 or right_dx == 0:  # remove identity translation
                continue
            translation = '', 0, dx, 0
            right_translation = '', 0, right_dx, right_drotation
            add_valid_translations_two_hand(lines, translation, right_translation)

        new_section('double-hand-translation-y')
        for dy, right_dy, right_drotation in product(range(-2, 3), range(-2, 3), range(-8, 9)):
            if dy == 0 or right_dy == 0:  # remove identity translation
                continue
            translation = '', dy, 0, 0
            right_translation = '', right_dy, 0, right_drotation
            add_valid_translations_two_hand(lines, translation, right_translation)

        # Generates mostly unused beat combinations, yet doubles the testing time
        # new_section('double-hand-translation-other')
        # for dy, dx, right_dy, right_dx, right_drotation in product(range(-2, 3), range(-3, 4),
        #                                                            range(-2, 3), range(-3, 4),
        #                                                            range(-8, 9)):
        #     if 0 in (dy, dx, right_dy, right_dx):  # remove previously used
        #         continue
        #     translation = '', dy, dx, 0
        #     right_translation = '', right_dy, right_dx, right_drotation
        #     add_valid_translations_two_hand(lines, translation, right_translation)

        new_section('double-hand-rotation-x')
        for rotation, right_dx, right_drotation in product(range(-8, 9), range(-3, 4), range(-8, 9)):
            if right_dx == 0:  # remove identity translation
                continue
            translation = '', 0, 0, rotation
            right_translation = '', 0, right_dx, right_drotation
            add_valid_translations_two_hand(lines, translation, right_translation)

        new_section('double-hand-rotation-y')
        for rotation, right_dy, right_drotation in product(range(-8, 9), range(-2, 3), range(-8, 9)):
            if right_dy == 0:  # remove identity translation
                continue
            translation = '', 0, 0, rotation
            right_translation = '', right_dy, 0, right_drotation
            add_valid_translations_two_hand(lines, translation, right_translation)

        # Generates mostly unused beat combinations, yet doubles the testing time
        # new_section('double-hand-rotation-other')
        # for rotation, right_dy, right_dx, right_drotation in product(range(-8, 9),
        #                                                              range(-2, 3), range(-3, 4),
        #                                                              range(-8, 9)):
        #     if 0 in (right_dy, right_dx):  # remove previously used
        #         continue
        #     translation = '', 0, 0, rotation
        #     right_translation = '', right_dy, right_dx, right_drotation
        #     add_valid_translations_two_hand(lines, translation, right_translation)

    
        with open(path, 'w') as wf:
            wf.write('\n'.join(lines) + '\n')

    print("Generating analogies...")
    create_analogies(storage_folder / 'beat_analogies.txt')

    def create_train_model(corpus_file, model_type: str='fasttext', **kwargs):
        kwargs = {key: int(val) for key, val in kwargs.items()}
        kwargs['vector_size'] = 2 ** kwargs['vector_size']
        workers = 12
        # change `workers` to suit our machine thread count
        if model_type.lower() == 'fasttext':
            model = gensim.models.FastText(corpus_file=str(corpus_file), **kwargs, workers=workers)
        else:
            model = gensim.models.Word2Vec(corpus_file=str(corpus_file), **kwargs, workers=workers)
        
        return model

    def create_eval_function(corpus_file: Path, model_type: str):
        def eval_model(**kwargs):
            model = create_train_model(corpus_file, model_type, **kwargs)

            res = model.wv.evaluate_word_analogies(storage_folder / 'beat_analogies.txt')

            return res[0]
        
        return eval_model
        
    from scipy import stats

    accuracies = []
    for size, _ in product(range(4, 9), range(2)):
        acc = create_eval_function(storage_folder / 'train_text.cor', 'word2vec')(epochs=5, vector_size=size)
        accuracies.append(acc)

    accuracy = {}
    acc_desc = stats.describe(accuracies)
    accuracy['random'] = acc_desc.minmax[1]  # get maximum
    print(f'{acc_desc.mean} +-{2 * acc_desc.variance}')

    # Train Word2Vec
    bool_ = (0.1, 1.9)
    pbounds = {
        'vector_size': (4, 8),         # log int
        'window': (1, 7),       # int
        'epochs': (1.1, 20),      # int  : Number of iterations (epochs) over the corpus.
        'sg': bool_,            # bool : skip-gram if `sg=1`, otherwise CBOW.
        'hs': bool_,            # bool : If 1, hierarchical softmax will be used for model training.
                                #        If set to 0, and `negative` is non-zero, negative sampling will be used.
        'cbow_mean': bool_,     # bool : If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
    }

    word2vec_optimizer = BayesianOptimization(
        f=create_eval_function(storage_folder / 'train_text.cor', 'word2vec'),
        pbounds=pbounds,
        random_state=1,
    )

    word2vec_optimizer.maximize(
        init_points=2,
        n_iter=6,
    )

    # Train FastText
    bool_ = (0.1, 1.9)
    pbounds = {
        'vector_size': (4, 8),         # log int
        'window': (1, 7),       # int
        'epochs': (1.1, 20),      # int  : Number of iterations (epochs) over the corpus.
        'sg': bool_,            # bool : skip-gram if `sg=1`, otherwise CBOW.
        'hs': bool_,            # bool : If 1, hierarchical softmax will be used for model training.
                                #        If set to 0, and `negative` is non-zero, negative sampling will be used.
        # 'sample': (0, 1e-5),   # float: The threshold for configuring which higher-frequency words are randomly downsampled,
        # 'negative': (0, 20),   # int  : If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
        'cbow_mean': bool_,     # bool : If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
        'min_n': (2, 5),        # int  : Minimum length of char n-grams to be used for training word representations.
        'max_n': (3, 9),        # int  : Max length of char ngrams to be used for training word representations. Set `max_n` to be lesser than `min_n` to avoid char ngrams being used.
    }

    fasttext_optimizer = BayesianOptimization(
        f=create_eval_function(storage_folder / 'train_text.cor', 'fasttext'),
        pbounds=pbounds,
        random_state=1,
    )

    fasttext_optimizer.maximize(
        init_points=2,
        n_iter=6,
    )

    rdf = pd.DataFrame(word2vec_optimizer.res)
    accuracy['search word2vec'], wordvec_params = rdf.loc[rdf['target'].idxmax()].to_list()

    rdf = pd.DataFrame(fasttext_optimizer.res)
    accuracy['search fasttext'], fasttext_params = rdf.loc[rdf['target'].idxmax()].to_list()

    word2vec_params = {
        'cbow_mean': True,
        'hs': False,
        'epochs': 3,
        'sg': False,
        'vector_size': 5,
        'window': 1,
    }

    # ngrams
    fasttext_params = {
        'cbow_mean': True,
        'hs': False, 
        'epochs': 3,
        'max_n': 3,
        'min_n': 2,
        'sg': False,
        'vector_size': 9,
        'window': 1,
        'word_ngrams': 1, 
    }   

    # without ngrams
    fasttext_params_no_ngrams = {
        'cbow_mean': True,
        'hs': False,
        'epochs': 3,
        'max_n': 3,
        'min_n': 2,
        'sg': False,
        'vector_size': 8,
        'window': 1,
        'word_ngrams': 0 
    }   


    model = create_train_model(storage_folder / 'train_text.cor', 'word2vec', **word2vec_params)
    accuracy['good word2vec'] = model.wv.evaluate_word_analogies(storage_folder / 'beat_analogies.txt')[0]

    model = create_train_model(storage_folder / 'train_text.cor', 'fasttext', **fasttext_params)
    accuracy['good fasttext'] = model.wv.evaluate_word_analogies(storage_folder / 'beat_analogies.txt')[0]

    # Performance on train set
    train_perf = pd.DataFrame(data=accuracy.values(), index=accuracy.keys(), columns=['best top1 accuracy [%]']).sort_values('best top1 accuracy [%]') * 100
    logging.info(f"Train Performance  Head:\n{train_perf.head().to_string()}")

    # Performance on val set
    model = create_train_model(storage_folder / 'train_text.cor', 'fasttext', **fasttext_params)
    train_accuracy = model.wv.evaluate_word_analogies(storage_folder / 'beat_analogies.txt')[0]

    val_accuracy = create_eval_function(storage_folder / 'val_text.cor', 'fasttext')(**fasttext_params)
    print(f'New model achieved {train_accuracy * 100:7.4} % accuracy on the train data.')
    print(f'New model achieved {val_accuracy * 100:7.4} % accuracy on the validation data.')

    # Save best model
    model.wv.save_word2vec_format(str(storage_folder / 'word2vec.model'), binary=False)

    model.wv.save(str(storage_folder / 'fasttext.model'))

    # test load
    gensim.models.KeyedVectors.load_word2vec_format(str(storage_folder / 'word2vec.model'))
    loaded_model = gensim.models.KeyedVectors.load(str(storage_folder / 'fasttext.model'))

    # check shape
    logging.info(loaded_model['R125_R217_R000_LLLL'].shape)   # fabricated word

if __name__ == "__main__":
    generate_action_embeddings()