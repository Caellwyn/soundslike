from scipy.spatial.distance import cdist
import glob, re, librosa, pathlib, json
import pandas as pd
import numpy as np
import soundfile as sf

def pad_sounds(sound_paths, min_length=7000):
    for path in sound_paths:
        sample, sr = sf.read(path)
        
        if len(sample) < min_length:
            n = len(sample)
            padding_amount = (7000 - n) // 2
            padded_array = np.pad(sample, ((padding_amount, padding_amount)), 'constant', constant_values=0)
        
        sf.write(path, padded_array, sr) 

def clean_filename(filename):
  """
  Removes strings of digits with 3 or more digits from both beginning and end of filename.
  Also splits on underscores and hyphens.

  Args:
    filename: The filename to be cleaned.

  Returns:
    The cleaned filename without numbers.
  """
  path = pathlib.Path(filename)
  name = path.name
  pattern = r"^(?:\d*)(?:_*)(?:-*)(.+?)((?:_|-\d{3,})|\d{3,})?(?:\.\w+)?$"
  match = re.match(pattern, name)

  if match:
    name = match.group(1)
    return name
  else:
    print(f"Filename '{filename}' does not match the expected format.")
    return None

      
def build_sound_df(sound_folder, api_key, save_path=None):
    from pyannote.audio import Model, Inference
    model = Model.from_pretrained("pyannote/embedding", 
                              use_auth_token=api_key)
    inference = Inference(model, window="whole")
    sound_folder = pathlib.Path(sound_folder)
    paths = glob.glob(str(sound_folder / '*.*'))

    good_paths = []
    sounds = []
    durations_in_seconds = []
    for path in paths:
        try:
            path = pathlib.Path(path)
            audio, sr = librosa.load(path.as_posix(), sr=None)
            length = len(audio) 
            
            if length < 1000:
                print(f'audio at {path} is empty')
                continue
                
            if length < 7000:
                padding_amount = (7000 - length) // 2
                audio = np.pad(audio, ((padding_amount, padding_amount)), 'constant', constant_values=0)

            duration = length/sr
            sounds.append(audio)
            good_paths.append(path.as_posix())
            durations_in_seconds.append(duration)

        except Exception as e:
            print(f'File at path {path} coult not be read or is not an audio file')
            print(e)
            continue
            
    pyannote_embeddings = [inference(path) for path in good_paths]
    df = pd.DataFrame()

    df['path'] = good_paths
    df['name'] = df['path'].map(clean_filename)
    df['duration in seconds'] = durations_in_seconds
 
    df['pyannote_embeddings'] = pyannote_embeddings
    df['pyannote_len'] = df['pyannote_embeddings'].map(len)

    if save_path:
        save_path = pathlib.Path(save_path)
        df.to_json(save_path.as_posix(), orient='columns')

    return df

def get_closest_sounds(sample, sound_df, n_closest=3, api_key=None, **kwargs):
    from pyannote.audio import Model, Inference
    if api_key == None:
        print('Please provide pyannote.audio api key')
        return None
    else:    
        embedder = Inference(Model.from_pretrained("pyannote/embedding", 
                             use_auth_token=api_key), window='whole')
            
    embed = embedder(sample, **kwargs).reshape(1, -1)
    
    distances = []
    indices = []
    for i in sound_df.index:
        candidate_sound = sound_df.loc[i, 'pyannote_embeddings'].reshape(1,-1)
        distance = cdist(embed, candidate_sound, metric="cosine")[0,0]
        indices.append(i)
        distances.append(distance)

    distance_df = sound_df.loc[indices].copy()
    print(distance_df.shape)
    distance_df['distance'] = distances
    distance_df = distance_df.sort_values('distance').iloc[1:n_closest+1]
    return distance_df