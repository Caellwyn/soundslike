import streamlit as st
import pandas as pd
import numpy as np
import soundfile as sf
from src.src import get_closest_sounds, build_sound_df
import librosa, pathlib, json, os
from torch import tensor

# api_key = 

@st.cache_data
def get_api_key():
    if not 'api_key' in globals():
        try:
            with open('../../secrets.json') as f:
                secrets = json.load(f)
                api_key = secrets["PYANNOTE_API_KEY"]
            print('API key retrieved from secrets file')
        except Exception as e:
            try:
                api_key = os.environ.get("PYANNOTE_API_KEY")
                print('API key retrieved from environment variables')
            except Exception as e:
                print('API KEY RETRIEVAL FAILED BECAUSE',e)
    return api_key

@st.cache_data
def get_df(api_key):
    if 'df' not in globals():
        try:
            df = pd.read_json('embedding_df.json')
            df['pyannote_embeddings'] = df['pyannote_embeddings'].apply(np.array)
            print('retrieved index df from file')
        except Exception as e:
            df = build_sound_df('sounds', api_key, save_path='embedding_df.json')
            print('bult new index df', e)

    longest_sound_duration = df['duration in seconds'].max() // 1 + 1
    longest_sound_duration = int(longest_sound_duration)

    return df, longest_sound_duration

def get_sounds(query_sound, df, api_key, min_duration, max_duration, num_sounds):
    if query_sound != None:
        pass
    else:
        print('no sound passed to get_sounds')
        return None
    
    sound, sr = librosa.load(query_sound, sr=16000)
    st.write('Submitted Sound:')
    st.audio(sound, sample_rate=sr)
    query = {'waveform':tensor(sound.reshape(1,-1)), "sample_rate":sr}

    duration_filtered_df = df[(df['duration in seconds'] > min_duration) & 
                            (df['duration in seconds'] < max_duration)]
    
    return_df = get_closest_sounds(query, duration_filtered_df, api_key=api_key, n_closest=num_sounds)
    
    for i, index in enumerate(return_df.index):
        st.write(f'Option {i+1}')

        view_df = pd.DataFrame(return_df.loc[index, ['name', 'distance', 'duration in seconds']]).T
        view_df = view_df.rename({"distance":"Similarity (lower is more similar)"}, axis=1)
        
        st.dataframe(view_df)
        
        name = return_df.loc[index, 'name']
        path = pathlib.Path(return_df.loc[index, 'path'])
        suffix = path.suffix
        return_sound, sr = librosa.load(path.as_posix())
        
        st.audio(return_sound, sample_rate=sr)
        
        with open(path, 'rb') as file:
            st.download_button(label=f'Download {name} sound',
                            data=file,
                            file_name=f'{name}{suffix}',
                            type='primary',
                            key=index)


st.title('SoundsLike')

api_key = get_api_key()
df, longest_sound_duration = get_df(api_key)

with st.form('get_query_sound'):
    st.text('Please upload a sound file to find similar sounds')

    query_sound = st.file_uploader('Upload your sound here', type=['wav','mp3'])

    num_sounds = st.selectbox("Choose number of sounds to return",
                              options=list(range(1,11)), index=2)

    min_duration, max_duration = st.slider('Return sounds between these lengths (in seconds):',
                                        0, longest_sound_duration, (1, 2))

    submitted = st.form_submit_button('Get Sounds')

if submitted:
    get_sounds(query_sound, df, api_key, min_duration, max_duration, num_sounds)
