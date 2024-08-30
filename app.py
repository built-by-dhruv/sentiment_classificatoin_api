from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# Define a data model for the input
class ArticleInput(BaseModel):
    content: str

# Initialize FastAPI
app = FastAPI()

# Initialize the emotion analysis pipeline
emotion_analyzer = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

def split_text(text, max_length=512):
    """Split text into smaller chunks that fit the model's input size."""
    sentences = text.split('.')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= max_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_length = sentence_length

    # Add the last chunk if exists
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')

    return chunks

def analyze_emotions(text):
    """Analyze the emotions of the given text."""
    chunks = split_text(text)
    emotions = {}

    for chunk in chunks:
        # Perform emotion detection on each chunk
        emotion_result = emotion_analyzer(chunk)
        for emotion in emotion_result[0]:
            emotions[emotion['label']] = emotions.get(emotion['label'], 0) + emotion['score']

    # Normalize the emotion scores
    total_emotion_score = sum(emotions.values())
    normalized_emotions = {k: v / total_emotion_score for k, v in emotions.items()}

    # Format the output to match your desired format
    formatted_emotions = {emotion: round(normalized_emotions.get(emotion, 0), 2) for emotion in 
                          ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']}
    return formatted_emotions

@app.post("/analyze_emotions")
async def analyze_article(article: ArticleInput):
    try:
        # Analyze the emotions of the input text
        emotions = analyze_emotions(article.content)
        return {"Emotion Analysis Result": emotions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run: uvicorn app:app --reload
