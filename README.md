# Spotify Personalized Mood-Based Playlist Generation

This project showcases my skills in machine learning, specifically unsupervised learning through clustering techniques applied to music recommendation systems.

## Project Overview

Developed a machine learning system that generates personalized music playlists by combining mood detection with audio feature analysis. The system addresses a key limitation in current music streaming platforms: recommendations based primarily on artist similarity and popularity, rather than musical characteristics and emotional context.

## Problem Statement

Current music recommendation systems have two major limitations:

1. **Artist-Centric Recommendations**: Platforms recommend songs from similar or popular artists, ignoring the underlying musical characteristics that make songs sound alike.
2. **Missing Emotional Context**: Existing algorithms don't account for the listener's current mood or emotional state when making recommendations.

This results in generic suggestions that fail to match both the user's taste in musical features (tempo, energy, acousticness) and their current emotional needs.

## Solution

A two-level clustering system that recommends music based on:
- **Musical similarity**: Songs with comparable audio features (BPM, energy levels, acousticness, danceability, valence)
- **Mood alignment**: User's current emotional state as input for playlist generation
- **Personalization**: Analysis of the user's existing library to understand their taste in musical characteristics

The system discovers new music that sounds similar to what the user already enjoys, while matching their current mood.

## Dataset

**Source**: Kaggle Spotify Tracks Dataset (2022)

**Size**: 114,000 tracks

**Features**:
- Track metadata (artist, album, track name, duration)
- Audio features (danceability, energy, acousticness, instrumentalness, liveness, valence)
- Musical properties (tempo, loudness, key, mode, time signature, speechiness)
- Popularity and explicit content flags
- Genre classification (114 unique genres)

## Methodology

### Data Cleaning

**Global Dataset Processing**:
- Removed 32,000 duplicate tracks based on track name and primary artist
- Handled 3 missing values in artist, album, and track name fields
- Standardized text formatting (lowercase, whitespace trimming)
- Validated feature ranges and removed invalid entries (negative duration/tempo, outliers in time signature)
- Final clean dataset: ~82,000 unique tracks

**User Library Processing**:
- Imported user's Spotify library from JSON export
- Standardized text fields to match global dataset format
- Removed 700 duplicates from personal library
- Merged user library with global dataset on track name and artist

### Feature Engineering

Minimal feature engineering was required as Spotify's audio features are well-structured:

1. **Tempo Normalization**: Applied min-max scaling to tempo values
2. **Loudness Scaling**: Inverted min-max scaling for loudness (higher values = louder)
3. **Artist Parsing**: Split composite artist fields (separated by ';') into individual artist columns
4. **Binary Conversion**: Converted 'explicit' flag to binary format

### Model Selection: K-Means Clustering

Selected K-Means over Hierarchical Clustering and DBSCAN based on four key criteria:

**Speed and Scalability**:
- 600x faster than hierarchical clustering
- Essential for processing large datasets and real-time recommendations

**Prediction Capability**:
- Only algorithm that can instantly predict cluster membership for new songs
- Enables real-time playlist generation

**Geometric Fit**:
- Spherical clusters naturally match the geometry of audio feature space
- Audio features exhibit clear centroid-based groupings

**Production-Ready Architecture**:
- Supports two-level clustering (main clusters + subclusters)
- Centroids enable efficient cosine similarity calculations for recommendations
- Scalable for continuous data ingestion

### Hyperparameter Tuning

Determined optimal number of clusters using two methods:

**Elbow Method**: Analyzed within-cluster sum of squares (WCSS) across different k values

**Silhouette Score**: Measured cluster cohesion and separation

**Result**: Optimal k = 6 main mood clusters

### Two-Level Clustering Architecture

**Level 1 - Main Mood Clusters (k=6)**:

1. **Happy/Energetic** (25,335 tracks)
   - Highest energy across all clusters
   - High danceability
   - Highest valence (cheerful, bright)
   - Low acousticness (lively, upbeat)

2. **Warm/Feel-Good** (22,103 tracks)
   - Medium-high energy
   - Good danceability
   - Moderately positive valence
   - Moderate acousticness (warm, comfortable)

3. **Sad/Reflective** (14,896 tracks)
   - Low energy
   - High acousticness (soft, intimate)
   - Lower valence (emotional, wistful)
   - Slightly higher speechiness (expressive)

4. **Moody Energetic** (5,223 tracks)
   - Medium-high energy
   - Moderate danceability
   - Low valence (emotionally darker)
   - Low acousticness (tense, not warm)

5. **Intense/Driven** (5,734 tracks)
   - High energy
   - Low valence (serious, not cheerful)
   - Low acousticness (sharper, aggressive tone)
   - Very high instrumentalness (focused)

6. **Melancholy/Peaceful** (5,206 tracks)
   - Lowest energy
   - Highest acousticness
   - Lowest valence (emotional heaviness)
   - High instrumentalness (quiet, contemplative)

**Level 2 - Subclustering for Personalization**:

Applied size-based subclustering logic to each main cluster:
- Clusters with <10,000 tracks: No subclustering
- Clusters with 10,000-20,000 tracks: 2 subclusters
- Clusters with >20,000 tracks: 3 subclusters

**Result**: 12 total micro-mood categories with 2,000-12,000 songs each

This enables more granular personalization while maintaining clear mood boundaries.

## Operational Architecture

The system follows a four-stage pipeline:

1. **Mood Selection**: User selects their current mood from the 6 main categories
2. **Library Analysis**: System analyzes user's Spotify library to identify their preferred audio feature patterns
3. **Cluster Matching**: User's songs are matched to appropriate subclusters within the selected mood
4. **Recommendation Generation**: System recommends songs from the same subclusters, prioritizing those with high cosine similarity to the user's library based on audio features

This architecture balances familiarity (songs matching user's taste) with discovery (new tracks within the mood category).

## Results and Analysis

**Successes**:
- Successfully captured distinct mood categories through audio feature clustering
- Two-level architecture enables both broad mood matching and fine-grained personalization
- K-Means provided fast, scalable clustering suitable for production deployment
- T-SNE visualization confirmed clear cluster separation

**Key Insight**:
Musical features alone capture mood effectively but have limitations in capturing subjective "taste." Two songs with nearly identical audio features (tempo, energy, acousticness) can still produce very different listener responses. This suggests that additional factors beyond Spotify's audio features influence music preference.

**Challenges Identified**:
- Personalizing recommendations based purely on audio features is more difficult than mood categorization
- Feature similarity doesn't always translate to listener satisfaction
- Need to balance feature-based recommendations with other preference signals

## Future Exploration

**Enhancing Taste Personalization**:
- Incorporate collaborative filtering to capture preference patterns beyond audio features
- Add genre and artist preference weighting
- Integrate listening history patterns (skip rate, replay frequency)
- Consider social signals and contextual listening data

**Scalability Improvements**:
- Implement incremental clustering for real-time dataset updates
- Optimize similarity calculations using approximate nearest neighbors
- Develop distributed processing for larger datasets
- Create API endpoints for production deployment

## Technologies Used

- Python 3.8+
- scikit-learn (K-Means, StandardScaler, PCA, T-SNE)
- pandas and numpy (data manipulation)
- matplotlib and seaborn (visualization)
- scipy (hierarchical clustering analysis)

## Skills Demonstrated

- Unsupervised learning (K-Means clustering)
- Two-level hierarchical clustering architecture
- Feature engineering and normalization
- Hyperparameter optimization (Elbow method, Silhouette score)
- Dimensionality reduction (T-SNE for visualization)
- Large-scale data processing (114K records)
- Production system design
- Music information retrieval concepts

## Project Structure

```
spotify-mood-recommendation/
│
├── spotify_data_cleaning.ipynb          # Data cleaning and preprocessing
├── double_clustering_model.ipynb        # Clustering model and recommendations
├── README.md                            # Project documentation
└── .gitignore                          # Git ignore file
```

## How to Run

1. Install required packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

2. Download the Spotify dataset from Kaggle:
   - [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)

3. Export your Spotify library:
   - Go to Spotify Account Settings
   - Request your data download
   - Extract the JSON file containing your library

4. Run the notebooks in order:
   - First: `spotify_data_cleaning.ipynb`
   - Second: `double_clustering_model.ipynb`

## Author

Matthieu Lafont  
Master of Management in Analytics - McGill University

## Presetation Deck

See the presentation deck Spotify Personalized Mood Based Playlist Generation.pptx

## Acknowledgments

Dataset provided by Kaggle user Maharshi Pandya. This project was developed as part of coursework exploring machine learning applications in music recommendation systems.
