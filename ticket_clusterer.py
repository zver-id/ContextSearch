from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Dict, List, Union, Any
import matplotlib.pyplot as plt

class TicketClusterer:
    """
    Кластеризация текстов.
    """
    def __init__(self, model_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'):
        """
        Инициализация
        :param model_name: название модели SentenceTransformer
        """
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.document_ids = None
        self.documents = None
        self.cluster_labels = None
        self.cluster_model = None

    def prepare_data(self, id_text_dict: Dict[Any, str]) -> None:
        """
        Подготовка данных: извлечение ID и текстов
        :param id_text_dict: id_text_dict: словарь {id: текст}
        """
        self.document_ids = list(id_text_dict.keys())
        self.documents = list(id_text_dict.values())

        if len(self.documents) < 2:
            raise ValueError("Для кластеризации необходимо как минимум 2 документа")

    def create_embeddings(self, normalize: bool = True) -> np.ndarray:
        """
        Создание векторных представлений текстов
        :param normalize: нормализовать ли эмбеддинги
        :return: массив эмбеддингов
        """
        if not self.documents:
            raise ValueError("Сначала подготовьте данные с помощью prepare_data()")

        self.embeddings = self.model.encode(
            self.documents,
            show_progress_bar=True,
            normalize_embeddings=normalize
        )
        return self.embeddings

    def kmeans_cluster(self, n_clusters: int = None, max_clusters: int = 10,
                       random_state: int = 42) -> Dict[str, Any]:
        """
        Кластеризация с помощью K-means
        :param n_clusters: количество кластеров (если None, определяется автоматически)
        :param max_clusters: максимальное количество кластеров для автоматического определения
        :param random_state: random state для воспроизводимости
        :return: словарь с результатами кластеризации
        """
        if self.embeddings is None:
            self.create_embeddings()

        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(max_clusters)

        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.cluster_labels = self.cluster_model.fit_predict(self.embeddings)

        return self._format_results()

    def dbscan_cluster(self, eps: float = None, min_samples: int = 2) -> Dict[str, Any]: #eps: float = 0.5
        """
        Кластеризация с помощью DBSCAN
        :param eps: максимальное расстояние между образцами одного кластера
        :param min_samples: минимальное количество образцов в кластере
        :return: cловарь с результатами кластеризации
        """
        if self.embeddings is None:
            self.create_embeddings(normalize=True)

        # Масштабирование для DBSCAN
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(self.embeddings)

        if eps is None:
            optimal_eps = self._find_optimal_eps(self.embeddings, min_samples)
            self.cluster_model = DBSCAN(eps=optimal_eps, min_samples=min_samples)
        else:
            self.cluster_model = DBSCAN(eps=eps, min_samples=min_samples)
        self.cluster_labels = self.cluster_model.fit_predict(scaled_embeddings)

        return self._format_results()

    def _find_optimal_eps(self, embeddings: np.ndarray, min_samples: int = 2) -> float:
        """
        Автоматический подбор оптимального eps с помощью метода локтя
        """
        # Вычисление расстояний до k-го ближайшего соседа
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors_fit = neighbors.fit(embeddings)
        distances, indices = neighbors_fit.kneighbors(embeddings)

        # Сортировка расстояний
        k_distances = np.sort(distances[:, min_samples - 1], axis=0)

        # Построение графика для визуального определения "локтя"
        # plt.figure(figsize=(10, 6))
        # plt.plot(k_distances)
        # plt.xlabel('Data Points Sorted by Distance')
        # plt.ylabel(f'Distance to {min_samples}-th Nearest Neighbor')
        # plt.title('K-Distance Graph for Optimal EPS')
        # plt.grid(True)
        # plt.show()

        # Автоматическое определение "локтя" (точки наибольшей кривизны)
        gradients = np.gradient(k_distances)
        second_derivatives = np.gradient(gradients)

        # Ищем точку, где вторая производная максимальна (наибольшая кривизна)
        elbow_index = np.argmax(np.abs(second_derivatives))
        optimal_eps = k_distances[elbow_index]

        print(f"Автоматически определенный eps: {optimal_eps:.3f}")
        return optimal_eps

    def _find_optimal_clusters(self, max_clusters: int) -> int:
        """
        Поиск оптимального количества кластеров с помощью silhouette score
        """
        if len(self.embeddings) < 2:
            return 1

        max_clusters = min(max_clusters, len(self.embeddings) - 1)
        best_score = -1
        best_n = 2

        for n in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n, random_state=42)
            labels = kmeans.fit_predict(self.embeddings)

            if len(set(labels)) > 1:  # Проверка, что есть как минимум 2 кластера
                score = silhouette_score(self.embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_n = n

        print(f"Оптимальное количество кластеров: {best_n} (silhouette score: {best_score:.3f})")
        return best_n

    def _format_results(self) -> Dict[str, Any]:
        """
        Форматирование результатов кластеризации
        """
        if self.cluster_labels is None:
            raise ValueError("Сначала выполните кластеризацию")

        # Группировка по кластерам
        clusters = {}
        for doc_id, doc_text, cluster_id in zip(self.document_ids, self.documents, self.cluster_labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append({
                'id': doc_id,
                'text': doc_text
            })

        # Расчет метрик
        metrics = {}
        if len(set(self.cluster_labels)) > 1:
            try:
                metrics['silhouette_score'] = silhouette_score(self.embeddings, self.cluster_labels)
            except:
                metrics['silhouette_score'] = None

        return {
            'clusters': clusters,
            'cluster_labels': dict(zip(self.document_ids, self.cluster_labels)),
            'metrics': metrics,
            'total_clusters': len(set(self.cluster_labels)),
            'cluster_sizes': {k: len(v) for k, v in clusters.items()}
        }

    def get_cluster_members(self, cluster_id: int) -> List[Dict]:
        """
        Получить все элементы определенного кластера.
        :param cluster_id: ID кластера
        :return: список элементов кластера
        """
        results = self._format_results()
        return results['clusters'].get(cluster_id, [])

    def get_document_cluster(self, document_id: Any) -> int:
        """
        Получить кластер для конкретного документа
        :param document_id: ID документа
        :return: ID кластера
        """
        if self.cluster_labels is None:
            raise ValueError("Сначала выполните кластеризацию")

        try:
            idx = self.document_ids.index(document_id)
            return self.cluster_labels[idx]
        except ValueError:
            raise ValueError(f"Документ с ID {document_id} не найден")