from sentence_transformers import SentenceTransformer, util
import numpy as np

class SimilarityComparer:
    """
    Поиск по небольшим предложениям.
    """

    def __init__(self, embeddings = None):

        # 'sentence-transformers/all-MiniLM-L6-v2' - быстрая модель для небольших текстов
        # 'sentence-transformers/all-mpnet-base-v2' - медленная модель для больших текстов.
        # Обе работают хорошо с английским, но не русским
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        if embeddings:
            self.texts = embeddings["texts"]
            self.ids = embeddings["ids"]
            self.embeddings = embeddings["embeddings"]
        else:
            self.texts = list()
            self.ids = list()
            self.embeddings = None

    def add_texts(self, texts_with_ids):
        """
        Добавляет тексты с и номера обращений
        :param texts_with_ids: список кортежей [(id, text), (id, text), ...]
        """
        if self.embeddings:
            return
        self.ids = [item[0] for item in texts_with_ids]
        self.texts = [item[1] for item in texts_with_ids]
        self.embeddings = self.model.encode(self.texts, convert_to_tensor=True)
        np.savez('embeddings_with_ids.npz',
                 texts = np.array(self.texts, dtype=object),
                 ids = np.array(self.ids),
                 embeddings = self.embeddings)

    def search(self, query, top_k=1000):
        """
        Поиск по запросу.
        :param query: Текст запроса.
        :param top_k: Количество возвращаемых результатов.
        :return: Результат в формате номер обращения: текст.
        """
        if self.embeddings is None:
            raise ValueError("Сначала добавьте тексты с помощью add_texts()")

        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, self.embeddings)[0]

        # Собираем результаты с ID
        results = []
        for idx, score in enumerate(cosine_scores):
            results.append({
                'id': self.ids[idx],
                'text': self.texts[idx],
                'score': score.item()
            })

        # Сортировка по убыванию схожести
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]