import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
#from onnxruntime.transformers.models.stable_diffusion.demo_utils import max_batch
from sentence_transformers import SentenceTransformer
from razdel import sentenize
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LongTextSearchEngine:
    def __init__(self, model_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                 chroma_db_path: str = "./chroma_db"):
        """
        Инициализация поискового движка
        :param model_name: название модели Sentence Transformers
        :param chroma_db_path: путь до базы данных
        """
        self.model = SentenceTransformer(model_name)
        self.chroma_db_path = chroma_db_path
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_db_path,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents_search",
            metadata={"hnsw:space": "cosine"}
        )
        self.documents_metadata = {}  # Для хранения соответствия между chunk_id и document_id
        self.doc_id_to_text = {}  # Для хранения оригинальных текстов документов
        self._load_data_from_collection()

    def _load_data_from_collection(self):
        """
        Загрузка данных из существующей коллекции ChromaDB
        :return:
        """
        try:
            count = self.collection.count()
            if count > 0:
                logger.info(f"Загрузка данных из существующей коллекции ({count} записей)...")

                # Получаем все данные из коллекции
                all_data = self.collection.get(
                    include=["metadatas", "documents"]
                )

                # Заполняем documents_metadata
                for i, (chunk_id, metadata, document_text) in enumerate(zip(
                        all_data["ids"],
                        all_data["metadatas"],
                        all_data["documents"]
                )):
                    doc_id = metadata["document_id"]
                    if i%1000 == 0:
                        logger.info(f"\rГенерация {i}")

                    # Сохраняем метаданные
                    self.documents_metadata[chunk_id] = {
                        "document_id": doc_id,
                        "chunk_index": metadata["chunk_index"],
                        "chunk_text": document_text,
                        "total_chunks": metadata["total_chunks"]
                    }

                    # Для doc_id_to_text нам нужно получить оригинальный текст документа
                    # Поскольку мы храним только chunks, соберем полный текст из всех chunks
                    if doc_id not in self.doc_id_to_text:
                        # Найдем все chunks для этого документа
                        doc_chunks = self.collection.get(
                            where={"document_id": doc_id},
                            include=["documents", "metadatas"]
                        )

                        # Соберем полный текст, отсортировав chunks по индексу
                        chunks_with_index = []
                        for chunk_meta, chunk_text in zip(doc_chunks["metadatas"], doc_chunks["documents"]):
                            chunks_with_index.append((chunk_meta["chunk_index"], chunk_text))

                        # Сортируем по индексу и объединяем
                        chunks_with_index.sort(key=lambda x: x[0])
                        full_text = " ".join([chunk[1] for chunk in chunks_with_index])
                        self.doc_id_to_text[doc_id] = full_text

                logger.info(f"Загружено {len(self.documents_metadata)} chunks и {len(self.doc_id_to_text)} документов")
            else:
                logger.info("Коллекция пустая, инициализированы пустые словари")

        except Exception as e:
            logger.error(f"Ошибка при загрузке данных из коллекции: {e}")
            # Инициализируем пустыми словарями в случае ошибки
            self.documents_metadata = {}
            self.doc_id_to_text = {}


    def preprocess_text(self, text: str) -> str:
        """
        Очистка текста от лишних пробелов и переносов строк
        :param text: исходный текст
        :return: обработанный текст
        """
        text = re.sub(r'\s+', ' ', text.strip())
        return text

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Разбитие текста на предложения
        :param text: текст
        :return: Список предложений
        """
        text = self.preprocess_text(text)
        sentences = [sentence.text for sentence in sentenize(text)]
        return sentences

    def create_chunks(self, sentences: List[str], chunk_size: int = 3, overlap: int = 1) -> List[str]:
        """
        Объединение предложений в chunks (отрезки).
        :param sentences: список предложений
        :param chunk_size: количество предложений в одном chunk
        :param overlap:
        :return: список chunks
        """
        chunks = []
        for i in range(0, len(sentences), chunk_size - overlap):
            chunk = ' '.join(sentences[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(sentences):
                break
        return chunks

    def add_documents(self, documents: List[Dict[str, Any]],
                      chunk_size: int = 3,
                      overlap: int = 1,
                      batch_size: int = 1000):
        """
        Добавление документов в векторную базу.
        :param documents: список словарей с ключами 'id' и 'text'
        :param chunk_size: количество предложений в одном chunk
        :param overlap:
        :param batch_size:
        """
        try:
            if not self.collection:
                self.collection = self.chroma_client.get_or_create_collection(
                    name="documents_search",
                    metadata={"hnsw:space": "cosine"}
                )

            all_chunks = []
            all_metadatas = []
            all_ids = []

            logger.info(f"Обработка {len(documents)} документов...")

            for doc in tqdm(documents, desc="Обработка документов"):
                doc_id = doc['id']
                text = doc['text']
                self.doc_id_to_text[doc_id] = text

                # Разбиваем текст на предложения
                sentences = self.split_into_sentences(text)

                # Создаем chunks из предложений
                chunks = self.create_chunks(sentences, chunk_size)

                for chunk_idx, chunk_text in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{chunk_idx}"

                    all_chunks.append(chunk_text)
                    all_metadatas.append({"document_id": doc_id, "chunk_index": chunk_idx, "total_chunks": len(chunks)})
                    all_ids.append(chunk_id)

                    # Сохраняем метаданные для быстрого доступа
                    self.documents_metadata[chunk_id] = {
                        "document_id": doc_id,
                        "chunk_index": chunk_idx,
                        "chunk_text": chunk_text,
                        "total_chunks": len(chunks)
                    }
            logger.info(f"Создано {len(all_chunks)} chunks. Генерация эмбеддингов...")

            # Генерация эмбеддингов батчами
            embeddings = []
            for i in tqdm(range(0, len(all_chunks), batch_size), desc="Генерация эмбеддингов"):
                batch_chunks = all_chunks[i:i + batch_size]
                batch_embeddings = self.model.encode(batch_chunks,
                                                 show_progress_bar=False,
                                                 batch_size=32)
                embeddings.extend(batch_embeddings.tolist())

            logger.info("Добавление данных в векторную базу...")

            for i in tqdm(range(0, len(all_chunks), batch_size), desc="Добавление в ChromaDB"):
                end_idx = min(i + batch_size, len(all_chunks))

                self.collection.add(
                    embeddings=embeddings[i:end_idx],
                    documents=all_chunks[i:end_idx],
                    metadatas=all_metadatas[i:end_idx],
                    ids=all_ids[i:end_idx]
                )

            logger.info(f"Успешно добавлено {len(documents)} документов, разбитых на {len(all_chunks)} chunks")

        except Exception as e:
            logger.error(f"Ошибка при добавлении документов: {e}")
            raise

        print(f"Добавлено {len(documents)} документов, разбитых на {len(all_chunks)} chunks")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Поиск похожих документов по запросу
        :param query: поисковый запрос (короткое предложение)
        :param top_k: количество возвращаемых результатов
        :return: список документов с оценкой релевантности
        """
        # Получаем эмбеддинг для запроса
        query_embedding = self.model.encode([query])[0].tolist()

        self.collection = self.chroma_client.get_or_create_collection(
            name="documents_search",
            metadata={"hnsw:space": "cosine"}
        )

        # Ищем похожие chunks
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 5,  # Ищем больше chunks для агрегации
            include=["metadatas", "documents", "distances"]
        )

        # Агрегируем результаты по document_id
        doc_scores = {}
        doc_chunks = {}

        for i, (metadata, chunk_text, distance) in enumerate(zip(
                results['metadatas'][0],
                results['documents'][0],
                results['distances'][0]
        )):
            doc_id = metadata['document_id']
            similarity_score = 1 - distance  # Конвертируем расстояние в схожесть

            if doc_id not in doc_scores:
                doc_scores[doc_id] = []
                doc_chunks[doc_id] = []

            doc_scores[doc_id].append(similarity_score)
            doc_chunks[doc_id].append({
                "text": chunk_text,
                "score": similarity_score,
                "chunk_index": metadata['chunk_index']
            })

        # Вычисляем агрегированные score для каждого документа
        aggregated_results = []
        for doc_id, scores in doc_scores.items():
            # Можно использовать разные стратегии агрегации:
            # max_score = np.max(scores)          # самый релевантный chunk
            avg_score = np.mean(scores)  # средняя релевантность
            # sum_score = np.sum(scores)          # суммарная релевантность

            aggregated_results.append({
                "document_id": doc_id,
                "score": avg_score,
                "original_text": self.doc_id_to_text[doc_id],
                "matching_chunks": doc_chunks[doc_id],
                "chunks_count": len(scores)
            })

        # Сортируем по убыванию релевантности
        aggregated_results.sort(key=lambda x: x['score'], reverse=True)

        return aggregated_results[:top_k]

    def print_results(self, results: List[Dict[str, Any]]):
        """Красивый вывод результатов поиска"""
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Документ ID: {result['document_id']}")
            print(f"   Общий score: {result['score']:.4f}")
            print(f"   Найдено релевантных chunks: {result['chunks_count']}")
            print(f"   Самый релевантный chunk:")
            best_chunk = max(result['matching_chunks'], key=lambda x: x['score'])
            print(f"   '{best_chunk['text']}' (score: {best_chunk['score']:.4f})")
            print("-" * 80)


# Пример использования
if __name__ == "__main__":
    # Инициализация поискового движка
    search_engine = LongTextSearchEngine()

    # Подготовка тестовых документов (длинных текстов)
    documents = [
        {
            "id": "doc_1",
            "text": """
            Искусственный интеллект революционизирует современную медицину. 
            Новые алгоритмы машинного обучения позволяют диагностировать заболевания 
            на ранних стадиях с высокой точностью. В частности, глубокое обучение 
            показывает outstanding результаты в анализе медицинских изображений. 
            Врачи всего мира начинают внедрять AI-системы в свою практику.
            """
        },
        {
            "id": "doc_2",
            "text": """
            Криптовалюты и блокчейн технологии продолжают развиваться быстрыми темпами.
            Bitcoin и Ethereum остаются лидерами рынка, но появляются и новые перспективные проекты.
            Децентрализованные финансы (DeFi) предлагают альтернативу традиционной банковской системе.
            Многие инвесторы видят в цифровых активах защиту от инфляции.
            """
        },
        {
            "id": "doc_3",
            "text": """
            Изменение климата становится одной из самых urgent проблем современности. 
            Глобальное потепление приводит к экстремальным погодным условиям по всему миру. 
            Ученые предупреждают о необходимости сокращения выбросов парниковых газов. 
            Возобновляемые источники энергии - ключ к sustainable будущему.
            """
        }
    ]

    # Добавляем документы в систему
    search_engine.add_documents(documents, chunk_size=2)

    # Примеры поисковых запросов (короткие предложения)
    queries = [
        "медицинская диагностика",
        "биткоин инвестиции",
        "глобальное потепление"
    ]

    # Выполняем поиск для каждого запроса
    for query in queries:
        print(f"\n🔍 Результаты поиска для запроса: '{query}'")
        print("=" * 80)

        results = search_engine.search(query, top_k=2)
        search_engine.print_results(results)