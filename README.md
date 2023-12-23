# SoundsLike
## Music Similarity Search

This repository contains the code to create a music similarity search Streamlit app.  The purpose of this app is to help sound effects artist to find sounds similar to a given sound to add variety or to find the perfect sound effect for their project.  

You can see the interface for this app at [https://soundslike.streamlit.app/](https://soundslike.streamlit.app/)

## Audio Embeddings
The base algorithm works by convertity audio files into 512 dimensional audio embeddings.  These embeddings have the property that vectors that are close together in the embedding space have similar tonal and rhythmic qualities.  This is achievied using the [pyannote.audio](https://github.com/pyannote/pyannote-audio) toolkit, including a pre-trained audio embedding model.  

The pyannote.audio model is trained for spearker diarization, or differentiating different speakers in audio.  This makes it ideal for finding sound similarity as it must differentiate the cadence and tone of different speakers.

The audio embeddings are then loaded into an directory which contains:
1. The audio embedding
2. A name for the sound, scraped from the original file name
3. The filepath to each sound.
4. The length of the audio file in seconds.

## Search 
An sound file uploaded by the user is converted to an audio embedding by the model, and a search is performed to find similar sounds to return.  The user also provides the number of returned sounds they would like and the range of permissible audio file lengths.

A k-nearest neighbors search is then performed to find the embeddings in the directory that are closest to the embedded query sound using cosine-similarity.  The results are filted by the range of user-defined permissable lengths and the user-specificed number of closest sounds are returned order of most to least similary sounds.

The returned sounds are made available for listening or downloading.

## Current Limitations
1. The database of possible sounds is very limited at this point, but are available with permissive and royalty-free licenses.
2. The search algorithm would not be efficient for larger directories of sounds.  While the custom search algorithm works well for the small number of sounds currently available, it would scale linearly with the number of sounds.  Databases with tens or hundreds of thousands of available sounds might experience unacceptable latency.  The solution for this would be to implement a more efficient vector search algorithm such as the Facebook developed faiss algorithm to partition the director to reduce the necessary number of similarity calculations.
