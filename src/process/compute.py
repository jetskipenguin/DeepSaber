import json
import os
import signal
import sys
from sys import stderr
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')
sys.path.append('.')

import numba
import numpy as np
import pandas as pd
import soundfile as sf
import speechpy
from tensorflow.python.distribute.multi_process_lib import multiprocessing

from utils.functions import progress
from utils.types import Config, JSON


def one_beat_element_per_hand(df: pd.DataFrame) -> pd.Series:
    """

    :param df: beat elements
    :return:
    """
    hands = []
    for hand in range(2):
        hands.append(df.loc[df['_type'] == hand][:1])

    # if only one hand has beat element, both predict the same
    for hand in range(2):
        if hands[hand].empty:
            hands[hand] = hands[hand - 1].copy()

    for hand in range(2):
        hands[hand] = hands[hand].iloc[0].drop(['_type', '_time'])

    return pd.concat([hands[0].add_prefix('l'), hands[1].add_prefix('r')])


@numba.njit()
def compute_true_time(beat_elements: np.ndarray, bpm_changes: np.ndarray, start_bpm: float) -> np.ndarray:
    """
    Calculate beat elements times in seconds. Originally in beats since beginning of the song.
    :param beat_elements: times of beat elements, sorted
    :param bpm_changes: [time, bpm], sorted by time
    :param start_bpm: initial bpm
    :return: time of beat_elements in seconds
    """
    true_time = np.zeros_like(beat_elements, dtype=np.float64)

    current_bpm = start_bpm
    current_beat = 0.0  # Current time in beats since beginning
    current_time = 0.0  # Current time in seconds since beginning
    event_index = 0

    for i in range(beat_elements.shape[0]):
        # Apply BPM changes that happened between this and last beat element
        while event_index < bpm_changes.shape[0] and bpm_changes[event_index, 0] < beat_elements[i]:
            bpm_change = bpm_changes[event_index]
            current_time += (bpm_change[0] - current_beat) * (60.0 / current_bpm)
            current_beat = bpm_change[0]
            current_bpm = bpm_change[1]
            event_index += 1

        current_time += (beat_elements[i] - current_beat) * (60.0 / current_bpm)
        current_beat = beat_elements[i]
        true_time[i] = current_time
    return true_time


def compute_time_cols(df):
    """
    Compute `prev`, `next`, `part`.
    :param df: beat elements
    :return: beat elements
    """
    df['time'] = df.index
    # previous beat in seconds
    df['prev'] = df['time'].diff().astype('float32')
    df['prev'] = df['prev'].fillna(df['prev'].max())
    # next beat in seconds
    df['next'] = df['prev'].shift(periods=-1).astype('float32')
    df['next'] = df['next'].fillna(df['next'].max())
    # which part of the song each beat belongs to
    df['part'] = (df['time'] / df['time'].max()).astype('float32')
    df = df.drop(columns='time')

    return df


def create_bpm_df(beatmap: JSON) -> pd.DataFrame:
    # 1. Safely detect version
    version_str = str(beatmap.get('version', beatmap.get('_version', '2')))
    is_v3 = version_str.startswith('3')
    
    # Initialize an empty DataFrame with guaranteed columns
    bpm_df = pd.DataFrame(columns=['_time', '_value'])
    
    if is_v3:
        raw_events = beatmap.get('bpmEvents', [])
        if raw_events:
            temp_df = pd.DataFrame(raw_events)
            if 'b' in temp_df.columns and 'm' in temp_df.columns:
                bpm_df = temp_df[['b', 'm']].rename(columns={'b': '_time', 'm': '_value'})
    else:
        events = beatmap.get('_events', [])
        if events:
            temp_df = pd.DataFrame(events)
            if '_type' in temp_df.columns and '_time' in temp_df.columns:
                bpm_df = temp_df.loc[temp_df['_type'] == 14]
                if not bpm_df.empty and '_value' in bpm_df.columns:
                    bpm_df = bpm_df[['_time', '_value']].copy()
                    bpm_df['_value'] /= 1000
                else:
                    bpm_df = pd.DataFrame(columns=['_time', '_value'])

    # Handle the extra BPM changes field
    if '_BPMChanges' in beatmap:
        raw_changes = beatmap.get('_BPMChanges', [])
        if raw_changes:
            bpm_changes_df = pd.DataFrame(raw_changes)
            if '_time' in bpm_changes_df.columns and '_BPM' in bpm_changes_df.columns:
                bpm_changes_df = bpm_changes_df[['_time', '_BPM']].rename(columns={'_BPM': '_value'})
                
                to_concat = [df for df in [bpm_df, bpm_changes_df] if not df.empty]
                if to_concat:
                    bpm_df = pd.concat(to_concat, ignore_index=True)

    if bpm_df.empty:
        return pd.DataFrame(columns=['_time', '_value'])

    bpm_df['_time'] = pd.to_numeric(bpm_df['_time'], errors='coerce')
    bpm_df['_value'] = pd.to_numeric(bpm_df['_value'], errors='coerce')
    
    return bpm_df.loc[bpm_df['_value'] >= 30].sort_values('_time').dropna()


def beatmap2beat_df(beatmap: JSON, info: JSON, config: Config) -> pd.DataFrame:
    # 1. Safely detect version
    version_str = str(beatmap.get('version', beatmap.get('_version', '2')))
    is_v3 = version_str.startswith('3')
    
    if is_v3:
        notes_data = beatmap.get('colorNotes', [])
        # old_k (JSON key) : new_k (DataFrame key)
        key_map = {'b': '_time', 'c': '_type', 'layer': '_lineLayer', 'x': '_lineIndex', 'd': '_cutDirection'}
        time_key = 'b'
    else:
        notes_data = beatmap.get('_notes', [])
        key_map = {'_time': '_time', '_type': '_type', '_lineLayer': '_lineLayer', '_lineIndex': '_lineIndex', '_cutDirection': '_cutDirection'}
        time_key = '_time'

    if not notes_data:
        raise ValueError("No notes found in beatmap file.")

    # 2. Extract notes safely (FIXED LOOP)
    raw_notes = []
    for x in notes_data:
        if x.get(time_key) is not None:
            note_dict = {}
            for old_k, new_k in key_map.items():
                val = x.get(old_k)
                # Safely cast to float, defaulting to 0 if the key is missing or explicitly null in JSON
                note_dict[new_k] = float(val) if val is not None else 0.0
            raw_notes.append(note_dict)

    if not raw_notes:
        raise ValueError("No valid notes with time data found.")

    # 3. Explicitly define columns so Pandas correctly maps the dictionaries
    df = pd.DataFrame(raw_notes, columns=['_time', '_type', '_lineLayer', '_lineIndex', '_cutDirection'])
    
    # Clip modded values out of bounds BEFORE astype
    df['_lineLayer'] = df['_lineLayer'].clip(0, 2)
    df['_lineIndex'] = df['_lineIndex'].clip(0, 3)
    df['_cutDirection'] = df['_cutDirection'].clip(0, 8)
    df['_type'] = df['_type'].clip(0, 1) 

    # Ensure types are correct and sort
    df = df.astype({
        '_type': 'int8', 
        '_lineLayer': 'int8', 
        '_lineIndex': 'int8', 
        '_cutDirection': 'int8'
    }).sort_values('_time')

    # Throw away bombs
    df = df.loc[df['_type'] != 3]
    df = df.sort_values(by=['_time', '_lineLayer'])

    # Round to 2 decimal places for block alignment
    df['_time'] = round(df['_time'], 2)

    # 4. Compute actual time in seconds
    bpm_df = create_bpm_df(beatmap)
    
    if bpm_df.empty:
        bpm_df = pd.DataFrame({'_time': [0.0], '_value': [info.get("_beatsPerMinute", 120.0)]})
        
    df['_time'] = np.around(compute_true_time(
        df['_time'].to_numpy(dtype=np.float64),
        bpm_df.to_numpy(dtype=np.float64),
        info.get("_beatsPerMinute", 120.0)
    ), 3)

    out_df = merge_beat_elements(df)
    out_df['word'] = compute_action_words(df)
    check_column_ranges(out_df, config)
    out_df = compute_time_cols(out_df)
    out_df.index = out_df.index.rename('time')

    return out_df


def compute_action_words(df):
    """
    Transform all beat elements with the same time stamp into one action, represented by a word.
    """
    df = df.set_index('_time')
    df['hand'] = 'L'
    df.loc[df['_type'] == 1, 'hand'] = 'R'
    df['word'] = df['hand'].str.cat([df[x].astype(str) for x in ['_lineLayer', '_lineIndex', '_cutDirection']])
    df = df.sort_values('word')
    temp = df['word'].groupby(level=0).apply(lambda x: x.str.cat(sep='_'))
    return temp


def check_column_ranges(out_df, config):
    for col in config.beat_preprocessing.beat_elements:
        minimum, maximum = out_df[col].min(), out_df[col].max()
        num_classes = [num for ending, num in config.dataset.num_classes.items() if col.endswith(ending)][0] - 1
        if minimum < 0 or num_classes < maximum:
            raise ValueError(
                f'[process|compute] column {col} with range <{minimum}, {maximum}> outside range <0, {num_classes}>')


def merge_beat_elements(df: pd.DataFrame):
    """
    Per each beat each hand should have exactly one beat element.
    :param df: beat elements
    :return:
    """
    # NOTE: You already fixed this section in your provided code, keeping it intact!
    hands = [df.loc[df['_type'] == x]
                 .drop_duplicates(subset='_time', keep='last')
                 .set_index('_time')
                 .drop(columns='_type')
             for x, hand in [[0, 'l'], [1, 'r']]]
    for hand in [0, 1]:
        not_in = hands[hand - 1].index.difference(hands[hand].index)
        hands[hand] = pd.concat([hands[hand], hands[hand - 1].loc[not_in]])
    hands = [x.add_prefix(hand) for x, hand in zip(hands, ['l', 'r'])]
    out_df = pd.concat(hands, axis=1)
    return out_df


def path2beat_df(beatmap_path, info_path, config: Config) -> pd.DataFrame:
    with open(info_path, encoding='utf-8') as info_data:
        info = json.load(info_data)
        if 'beatsPerMinute' in info:
            info['_beatsPerMinute'] = info['beatsPerMinute']
    with open(beatmap_path, encoding='utf-8') as beatmap_data:
        beatmap = json.load(beatmap_data)
        return beatmap2beat_df(beatmap, info, config)


def process_song_folder(folder, config: Config, order=(0, 1)):
    """
    Return processed and concatenated dataframe of all songs in `folder`.
    Returns `None` if an error occurs.

    Each beat is determined by multiindex of song name, difficulty and time (in seconds).
    Each beat contains information about:
    - MFCC of audio
    - beat elements
    - previous beat elements
    - time (in seconds) to previous / next beat
    - proportion of the song it belongs to
    """
    progress(*order, config=config, name='Processing song folders')

    files = [] 
    for dirpath, dirnames, filenames in os.walk(folder):
        files.extend(filenames)
        break
        
    info_path = os.path.join(folder, [x for x in files if 'info' in x.lower()][0])
    file_ogg = os.path.join(folder, [x for x in files if x.endswith('gg')][0])
    folder_name = folder.split('/')[-1]
    df_difficulties = []

    try:
        mfcc_df = path2mfcc_df(file_ogg, config=config)
    except (ValueError, FileNotFoundError, AttributeError) as e:
        print(f'\n\t[process | process_song_folder] Skipped file {folder_name}  |  {folder}:\n\t\t{e}', file=stderr)
        return None

    # Load Info.dat to map difficulties to actual filenames
    difficulty_mapping = {}
    with open(info_path, 'r', encoding='utf-8') as f:
        info_data = json.load(f)
        
    # Handle V2 and V3 Info structures
    beatmap_sets = info_data.get('_difficultyBeatmapSets') or info_data.get('difficultyBeatmapSets') or []
    for bset in beatmap_sets:
        mode = bset.get('_beatmapCharacteristicName') or bset.get('beatmapCharacteristicName')
        # Filter strictly for 'Standard' to ignore Lawless/360 modes
        if mode == 'Standard':
            maps = bset.get('_difficultyBeatmaps') or bset.get('difficultyBeatmaps') or []
            for m in maps:
                diff = m.get('_difficulty') or m.get('difficulty')
                fname = m.get('_beatmapFilename') or m.get('beatmapFilename')
                if diff and fname:
                    difficulty_mapping[diff] = fname

    for difficulty in ['Easy', 'Normal', 'Hard', 'Expert', 'ExpertPlus']:
        # Retrieve the specific filename from the Info.dat mapping
        beatmap_file = difficulty_mapping.get(difficulty)
        
        # Fallback to older string-matching if Info.dat didn't contain it
        if not beatmap_file:
            matches = [x for x in files if difficulty in x and x.endswith('.dat')]
            if matches:
                beatmap_file = matches[0]

        if beatmap_file and beatmap_file in files:
            try:
                beatmap_path = os.path.join(folder, beatmap_file)
                df = path2beat_df(beatmap_path, info_path, config)
                df = join_closest_index(df, mfcc_df, 'mfcc')
                df = add_multiindex(df, difficulty, folder_name)

                df_difficulties.append(df)
            except (ValueError, IndexError, KeyError, UnicodeDecodeError) as e:
                print(
                    f'\n\t[process | process_song_folder] Skipped file {folder_name}/{difficulty} | {folder}:\n\t\t{e}',
                    file=stderr)

    if df_difficulties:
        return pd.concat(df_difficulties)
    return None


def add_multiindex(df, difficulty, folder_name):
    df['difficulty'] = difficulty
    df['name'] = folder_name
    df = df.set_index(['name', 'difficulty'], append=True).reorder_levels(['name', 'difficulty', 'time'])
    return df


def add_previous_prediction(df: pd.DataFrame, config: Config):
    beat_elements_pp = config.dataset.beat_elements_previous_prediction
    beat_actions_pp = config.dataset.beat_actions_previous_prediction
    df_shifted = df[config.dataset.beat_elements + config.dataset.beat_actions].shift(1)
    df[beat_elements_pp + beat_actions_pp] = df_shifted
    df = df.dropna().copy()
    df.loc[:, beat_elements_pp] = df[beat_elements_pp].astype('int8')

    indexes_to_drop = ['name', 'difficulty']
    df = df.reset_index(level=indexes_to_drop).drop(columns=indexes_to_drop)
    return df


def join_closest_index(df: pd.DataFrame, other: pd.DataFrame, other_name: str = 'other') -> pd.DataFrame:
    """
    Join `df` with the closest row (by index) of `other`
    :param df: index in time,
    :param other: index in time, constant intervals
    :param other_name: name of the joined columns
    :return: df
    """
    original_index = df.index
    round_index = other.index.values[1] - other.index.values[0]
    df.index = np.floor(df.index / round_index).astype(int)
    other_offset = (other.index / round_index).astype(int).min() - 1
    other = other.reset_index(drop=True)
    other.index = other.index + other_offset

    other.name = other_name
    if len(other.columns) == 1:
        df = df.join(other)
    else:
        df = df.join(other, rsuffix=f'_{other_name}')
    df.index = original_index
    return df


def path2mfcc_df(ogg_path, config: Config) -> pd.DataFrame:
    cache_path = f'{".".join(ogg_path.split(".")[:-1])}.pkl'

    if os.path.exists(cache_path):
        df = pd.read_pickle(cache_path)
    else:
        if config.audio_processing.use_cache:
            raise FileNotFoundError('Cache file not found')
        signal, samplerate = sf.read(ogg_path)
        df = audio2mfcc_df(signal, samplerate, config)
        df.to_pickle(cache_path)

    if config.audio_processing.use_temp_derrivatives:
        df = df.join(df.diff().fillna(0), rsuffix='_d')

    if config.audio_processing.time_shift is not None:
        df_shifted = df.copy()
        df_shifted.index = df_shifted.index + config.audio_processing.time_shift
        df = join_closest_index(df, df_shifted, 'shifted')
        df.dropna(inplace=True)

    flatten = np.split(df.to_numpy().astype('float16').flatten(), len(df.index))
    return pd.DataFrame(data={'mfcc': flatten}, index=df.index)


def audio2mfcc_df(signal: np.ndarray, samplerate: int, config: Config) -> pd.DataFrame:
    if len(signal) > config.audio_processing.signal_max_length:
        raise ValueError('[process|audio] Signal longer than set maximum')

    if signal.ndim == 2:
        if signal.shape[1] == 2:
            signal = (signal[:, 0] + signal[:, 1]) / 2
        else:
            signal = signal[:, 0]

    signal_preemphasized = speechpy.processing.preemphasis(signal, cof=0.98)

    mfcc = speechpy.feature.mfcc(signal_preemphasized,
                                 sampling_frequency=samplerate,
                                 frame_length=config.audio_processing.frame_length,
                                 frame_stride=config.audio_processing.frame_stride,
                                 num_filters=40,
                                 fft_length=512,
                                 num_cepstral=config.audio_processing.num_cepstral)

    index = np.arange(0,
                      (len(mfcc) - 0.5) * config.audio_processing.frame_stride,
                      config.audio_processing.frame_stride) + config.audio_processing.frame_length
    return pd.DataFrame(data=mfcc, index=index, dtype='float16')


def init_worker():
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def create_ogg_cache(ogg_path, config: Config, order=(0, 1)):
    progress(*order, config=config, name='Recalculating MFCCs')
    try:
        path2mfcc_df(ogg_path, config=config)
    except ValueError as e:
        print(f'\tSkipped file {ogg_path} \n\t\t{e}', file=stderr)


def create_ogg_caches(ogg_paths, config: Config):
    total = len(ogg_paths)
    inputs = ((s, config, (i, total)) for i, s in enumerate(ogg_paths))
    with multiprocessing.get_context("spawn").Pool(initializer=init_worker()) as pool:
        pool.starmap(create_ogg_cache, inputs)
        pool.close()
        pool.join()


def remove_ogg_cache(ogg_paths):
    for i, ogg_path in enumerate(ogg_paths):
        cache_path = f'{".".join(ogg_path.split(".")[:-1])}.pkl'
        if os.path.exists(cache_path):
            os.remove(cache_path)


def create_ogg_paths(song_folders):
    ogg_paths = []
    for folder in song_folders:
        files = []
        for dirpath, dirnames, filenames in os.walk(folder):
            files.extend(filenames)
            break
        ogg_paths.append(os.path.join(folder, [x for x in files if x.endswith('gg')][0]))
    return ogg_paths


def generate_snippets(song_df: pd.DataFrame, config: Config):
    stack = []
    ln = len(song_df)
    window = config.beat_preprocessing.snippet_window_length
    skip = config.beat_preprocessing.snippet_window_skip

    if ln < window:
        return None

    indexes_to_drop = ['name', 'difficulty']
    song_df = song_df.reset_index(level=indexes_to_drop).drop(columns=indexes_to_drop)

    for s in range(0, ln, skip):
        if s + window > ln:
            stack.append(song_df.iloc[-window:])
        else:
            stack.append(song_df.iloc[s:s + window])

    df = pd.concat(stack, keys=list(range(0, len(song_df), skip)), names=['snippet', 'time'])
    return df


if __name__ == '__main__':
    config = Config()
    df1 = process_song_folder('/home/jetskipenguin/Python/DeepSaber/data/human_beatmaps/new_dataformat/292', config=config)
    print(df1.columns)