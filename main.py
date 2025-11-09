import numpy as np

from long_text_search import LongTextSearchEngine
from reference import Reference
from similarity_comparer import SimilarityComparer
from ticket_clusterer import TicketClusterer

def get_tickets_dict():
    tickets = Reference("ПДД")

    closed_until = "({}.{} >= '{}')".format(tickets.reference.TableName,
                                            tickets.reference.Requisites("ДатОткр").FieldName,
                                            "15.10.2025")  # начало периода
    tickets.reference.AddWhere(closed_until)

    tickets.set_filter("ТипОбращения", ['И', 'К'])
    tickets.disable_auto_solved()
    tickets.set_filter("НаименованиеМЦ", ["5838507"])

    tickets.reference.Open()
    tickets.reference.First()
    ticket_texts = dict()
    count = 0
    while not tickets.reference.EOF:
        #topic = tickets.reference.Requisites("Содержание").AsString
        ticket_num = tickets.reference.Requisites("Код").AsString
        tickets.reference.OpenRecord()
        full_text = tickets.reference.Requisites("Текст").AsString
        tickets.reference.Cancel()
        tickets.reference.CloseRecord()
        #hyperlink = tickets.reference.hyperlink
        #ticket_texts.append({"id": f"{ticket_num}, {topic}, {hyperlink}",
        #                     "text": full_text})
        ticket_texts[ticket_num] = full_text
        tickets._next_record()
        count += 1
        if count % 10 == 0:
            print(f"\rЗагрузка обращений {count}.", end='', flush=True)
    print(f"Загружено {count} обращений.")

    return ticket_texts

def get_similarity_description():
    try:
        loaded_embeddings = np.load('embeddings_with_ids.npz', allow_pickle=True)
        comparer = SimilarityComparer(loaded_embeddings)
    except FileNotFoundError:
        tickets = Reference("ПДД")

        closed_until = "({}.{} >= '{}')".format(tickets.reference.TableName,
                                                tickets.reference.Requisites("ДатОткр").FieldName,
                                                "01.01.2025")  # начало периода
        closed_until = tickets.reference.AddWhere(closed_until)

        tickets.set_filter("ТипОбращения", ['И', 'К'])
        tickets.disable_auto_solved()
        tickets.set_filter("НаименованиеМЦ", ["5838507"])

        tickets.reference.Open()
        tickets.reference.First()
        ticket_topics = list()
        count = 0
        while not tickets.reference.EOF:
            topic = tickets.reference.Requisites("Содержание").AsString
            ticket_num = tickets.reference.Requisites("Код").AsString
            # tickets.reference.OpenRecord()
            # full_text = tickets.reference.Requisites("Текст").AsString
            # tickets.reference.Cancel()
            # tickets.reference.CloseRecord()
            hyperlink = tickets.reference.hyperlink
            ticket_topics.append((f"{ticket_num}, {topic}, {hyperlink}", topic))
            tickets._next_record()
            count += 1
            if count % 10 == 0:
                print(f"\rЗагрузка обращений {count}.", end='', flush=True)
        print(f"Загружено {count} обращений.")
        comparer = SimilarityComparer()
        comparer.add_texts(ticket_topics)
    finally:
        result = comparer.search("не работает авторизация")

    num = 1
    for topic in result:
        print(f"{num}. {topic['id']} {topic['score']}")
        num += 1

def get_similarity_text():
    search_engine = LongTextSearchEngine()

    tickets = Reference("ПДД")

    closed_until = "({}.{} >= '{}')".format(tickets.reference.TableName,
                                            tickets.reference.Requisites("ДатОткр").FieldName,
                                            "01.01.2025")  # начало периода
    tickets.reference.AddWhere(closed_until)

    tickets.set_filter("ТипОбращения", ['И', 'К'])
    tickets.disable_auto_solved()
    tickets.set_filter("НаименованиеМЦ", ["5838507"])

    tickets.reference.Open()
    tickets.reference.First()
    ticket_texts = list()
    count = 0
    while not tickets.reference.EOF:
        topic = tickets.reference.Requisites("Содержание").AsString
        ticket_num = tickets.reference.Requisites("Код").AsString
        tickets.reference.OpenRecord()
        full_text = tickets.reference.Requisites("Текст").AsString
        tickets.reference.Cancel()
        tickets.reference.CloseRecord()
        hyperlink = tickets.reference.hyperlink
        ticket_texts.append({"id": f"{ticket_num}, {topic}, {hyperlink}",
                             "text": full_text})
        tickets._next_record()
        count += 1
        if count % 10 == 0:
            print(f"\rЗагрузка обращений {count}.", end='', flush=True)
    print(f"Загружено {count} обращений.")

    # Добавляем документы в систему
    search_engine.add_documents(ticket_texts, chunk_size=2, batch_size=500)

    # Выполняем поиск для каждого запроса
    query = "Снижение производительности"
    print(f"\n Результаты поиска для запроса: '{query}'")
    print("=" * 80)

    results = search_engine.search(query, top_k=2)
    search_engine.print_results(results)

def clustering():
    # Пример данных
    documents_dict = {
        1: "Машинное обучение и искусственный интеллект",
        2: "Глубокое обучение и нейронные сети",
        3: "Программирование на Python",
        4: "Разработка веб-приложений",
        5: "Обработка естественного языка",
        6: "Компьютерное зрение и изображения",
        7: "Анализ данных и статистика",
        8: "Базы данных и SQL"
    }

    documents_dict = get_tickets_dict()

    clusterer = TicketClusterer()
    clusterer.prepare_data(documents_dict)

    # Кластеризация K-means
    results_kmeans = clusterer.kmeans_cluster()

    print("K-means результаты:")
    print(f"Количество кластеров: {results_kmeans['total_clusters']}")
    print(f"Silhouette score: {results_kmeans['metrics'].get('silhouette_score', 'N/A')}")

    for cluster_id, members in results_kmeans['clusters'].items():
        print(f"\nКластер {cluster_id} ({len(members)} документов):")
        for member in members:
            print(f"  ID {member['id']}: {member['text']}")

    # Кластеризация DBSCAN
    results_dbscan = clusterer.dbscan_cluster(eps=0.3, min_samples=2)

    print("\nDBSCAN результаты:")
    print(f"Количество кластеров: {results_dbscan['total_clusters']}")

    for cluster_id, members in results_dbscan['clusters'].items():
        print(f"\nКластер {cluster_id} ({len(members)} документов):")
        for member in members:
            print(f"  ID {member['id']}: {member['text']}")


if __name__ == "__main__":
    #get_similarity_description()
    #get_similarity_text()
    #search_engine = LongTextSearchEngine()
    #query = "Медленная работа"
    #print(f"\n Результаты поиска для запроса: '{query}'")
    #print("=" * 80)

    #results = search_engine.search(query, top_k=10)
    #search_engine.print_results(results)
    clustering()