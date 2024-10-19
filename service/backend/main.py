from fastapi import APIRouter, FastAPI
import matplotlib.pyplot as plt


from utils.clustering import get_leaving_reasons, get_multi_leaving_reasons
from utils.santiment import get_single_sentiment_analysis, get_multi_sentiment_analysis


app = FastAPI()
router = APIRouter()

@router.post("/single_clustering/")
async def single_clustering(data: dict):
    clusters = get_leaving_reasons(data.data)
    return {"clusters": clusters}

@router.post("/multi_clustering/")
async def multi_clustering(data: dict):
    clusters = get_multi_leaving_reasons(data)
    labels = list(set(clusters))
    sizes = [clusters.count(label) for label in labels]

    fig, ax = plt.subplots()
    ax.bar(labels, sizes)
    ax.set_xlabel('Кластеры')
    ax.set_ylabel('Количество')
    ax.set_title('Распределение кластеров')
    
    plt.close(fig)
    return {"clusters": clusters, "graph": fig}


@router.post("/single_sentiment_analysis/")
async def single_sentiment_analysis(data: dict):
    sentiment_count = get_single_sentiment_analysis(data['answer'])
    return {"analysis": sentiment_count}

@router.post("/multi_sentiment_analysis/")
async def multi_sentiment_analysis(data: dict):
    sentiment_counts = get_multi_sentiment_analysis(data['answer'])
    labels = ['Позитивный', 'Негативный', 'Нейтральный']
    sizes = [sentiment_counts['positive'],
             sentiment_counts['negative'],
             sentiment_counts['neutral']]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#66b3ff', '#ff6666', '#ffcc99'])
    ax.axis('equal')
    plt.close(fig)

    analysis_text = (f"Позитивных: {sizes[0]}, Негативных: {sizes[1]}, Нейтральных: {sizes[2]}")    
    return {"analysis": analysis_text, "graph": fig}


app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)