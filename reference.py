import os
import sys

import datetime
import asyncio
try:
    import pythoncom
except ModuleNotFoundError:
    import subprocess
    subprocess.check_call(["pip", "install", "pywin32"])
    subprocess.check_call(["pip", "install", "pypiwin32"])
    import pythoncom
import win32com.client
from datetime import datetime, timedelta, time
import logging
import yaml

'''
from date_calculation import DateCalculation
from autoclick import click_yes
import config_reader as params
from tickets_list import TicketsList
'''

"""
Необходимые пакеты для импорта в venv:
pypiwin32
pywin32 - в PyCharm подтягиевается автоматически

Справочник "Обращения в службу поддержки" - "ПДД"
"""


class Reference:

    def __init__(self, reference_name):
        """
        Создание соединение с ТехКАС
        :param reference_name: название справочника в ТехКАС
        :return: справочник ТехКАС
        """
        pythoncom.CoInitialize()
        win32com.SetupEnvironment
        win32com.gen_py
        lp = win32com.client.DispatchEx("SBLogon.LoginPoint")
        ap = lp.GetApplication("systemcode=TEHKASNPO")

        # подключаем справочник "Обращения в службу поддержки"
        self.reference = ap.ReferencesFactory.ReferenceFactory(reference_name).GetComponent()
        self.filters = dict()
        #self.set_init_filters()

    def set_filter(self, props_name: str, attribute, comparison_type=True):
        """
        Установить фильтр на справочник
        :param props_name: Имя атрибута справочника;
        :param attribute: Список значений атрибута;
        :param comparison_type: тип сравнения: равно значениям из списка или не равно значению
        :return: код установленного фильтра, по которому его потом можно снять
        """
        if comparison_type:
            filter_string = self.return_query_equal(attribute, props_name)
        else:
            filter_string = self.return_query_not_equal(attribute, props_name)
        reference_filter = self.reference.AddWhere(filter_string)
        self.filters[reference_filter] = [props_name, attribute]
        return reference_filter

    def disable_auto_solved(self):
        """
        Фильтрация автоматически обработаных обращений
        :return: код установленного фильтра, по которому его потом можно снять
        """
        auto_solved_string = (f"({self.reference.TableName}.{self.reference.Requisites('ДаНет5').FieldName} = 'Н'"
                              f" or {self.reference.TableName}.{self.reference.Requisites('ДаНет5').FieldName}"
                              f" is Null)")
        auto_solved_add_where = self.reference.AddWhere(auto_solved_string)
        self.filters[auto_solved_add_where] = ['ДаНет5', "= 'H' or is Null)"]
        return auto_solved_add_where

    def delete_filter(self, filter_id):
        """
        Удаление фильтра, установленного на справочник
        :param filter_id: id установленного фильтра
        """
        self.reference.DelWhere(filter_id)
        self.filters.pop(filter_id, None)

    def return_query_equal(self, attribute_list, props_name):
        """
        Сбор запроса для метода AddWhere из списка параметров. Проверяется равенство одному из параметров
        :param attribute_list: список параметров;
        :param props_name: имя реквизита по которому собираем запрос;
        :return: запрос для применения в методе AddWhere
        """
        add_where = ""
        for attribute in attribute_list:
            if add_where:
                add_where = add_where + " or {}.{} = '{}'".format(
                    self.reference.TableName, self.reference.Requisites(props_name).FieldName, attribute)
            else:
                add_where = add_where + "({}.{} = '{}'".format(
                    self.reference.TableName, self.reference.Requisites(props_name).FieldName, attribute)
            if len(attribute_list) == attribute_list.index(attribute) + 1:
                add_where = add_where + ")"
        return add_where

    def return_query_not_equal(self, attribute, props_name):
        """
        Сбор запроса для метода AddWhere из единственного параметра. Проверяется неравенство параметру
        :param attribute: значение параметра
        :param props_name: имя реквизита
        :return: запрос для применения в методе AddWhere
        """
        return f"{self.reference.TableName}.{self.reference.Requisites(props_name).FieldName} <> {attribute}"

    def _next_record(self):
        """
        При переборе справочника отменяем изменения, закрываем обращение, выбираем следующее
        """
        self.reference.Cancel()
        self.reference.CloseRecord()
        self.reference.Next()

    def set_init_filters(self):
        # все коды поступления обращений кроме чат-бота
        not_chat_bot_list = [1761155, 1761153, 1761152, 6753420, 1761156, 3010066, 4521240, 1768268, 4081742, 1761154]
        self.set_filter("ИсточникОбращения", not_chat_bot_list)

        # отключение из выборки обращений по признаку "Решено автоматически"
        self.disable_auto_solved()

        # Убираем анонимки # анонимки (Код-338)
        self.set_filter('ОбластьПоддержки', '30262732', comparison_type=False)

        # Неизвестная организация (КАРТОЧКУ НЕ МЕНЯТЬ, СЛУЖЕБНАЯ ЗАПИСЬ!!!!) (ИД-2070891)
        self.set_filter('Организация', '2070891', comparison_type=False)

    def set_additional_filters(self, team_name):
        if getattr(sys, 'frozen', False):
            appdir = os.path.dirname(sys.executable)
        else:
            appdir = os.path.dirname(os.path.realpath(__file__))
        os.chdir(f"{os.path.abspath(appdir)}/configs")
        with open(f"{team_name}.yml", "r", encoding="utf-8") as yml_file:
            yml_config = yaml.safe_load(yml_file)
        for filter_name, value in yml_config["Additional_filters"].items():
            self.set_filter(filter_name, value[0], comparison_type=value[1])


    def get_tickets_count_by_type(self, ticket_type: list):
        """
        Считает количество обращений в списке по типу
        :param ticket_type: тип обращений, ожидается список
        :return:
        """
        tickets_type = self.set_filter("ТипОбращения", ticket_type)
        self.reference.Open()
        tickets_count = self.reference.RecordCount
        self.reference.Close()
        self.delete_filter(tickets_type)
        return tickets_count

    def registered_in_day(self, days_ago):
        """
        Обращения, зарегистрированные за предыдущий день

        !!!
        Потроганные обращения:
        1. Смотрим таблицу состояний. Если в ней есть переходы в состояние "На контроле" или "Переадресовано", то
        с обращением работали
        2. Если это запрос, то в выполненных работах смотрим работы с наименованием "ТМ" от имени сотрудника из команды
        !!!

        :param days_ago: Количество дней назад за которое считаем
        :return: словарь в формате "зарегистрировано": количество обращений, "в работе": количесвтво обращений
        """

        current_day = self.set_filter("ДатОткр", self.get_days_ago(days_ago))
        incoming = dict()
        incoming["incidents"] = self.get_tickets_count_by_type(["И"])
        incoming["requests"] = self.get_tickets_count_by_type(["З"])
        incoming["consultation"] = self.get_tickets_count_by_type(["К"])
        incoming["problems"] = self.get_tickets_count_by_type(["П"])
        self.delete_filter(current_day)
        return incoming

    def get_time_for_request(self, days_ago, team):
        """
        Считаем время, потраченное на запросы
        :return:
        """
        return asyncio.run(self.async_get_time_for_request(days_ago, team))

    async def async_get_time_for_request(self, days_ago, team):
        """
        Считает время потраченное на запросы
        :param reference: справочник
        :param days_ago: кол-во дней назад от текущей даты за которую надо считать
        :param team: список команды (Список имен)
        :return:
        """
        yesterday = (f"({self.reference.TableName}.{self.reference.Requisites('СостОбращения').FieldName} <> 'З' "
                     f"or {self.reference.TableName}.{self.reference.Requisites('ДатЗакр').FieldName} >="
                     f" '{(datetime.today() - timedelta(days=(days_ago + 2))).date().strftime('%d.%m.%Y')}')")
        yesterday = self.reference.AddWhere(yesterday)

        request = self.set_filter("ТипОбращения", ['З'])

        total = 0
        self.reference.Open()
        self.reference.First()
        while not self.reference.EOF:
            await click_yes()
            self.reference.OpenRecord()

            detail = self.reference.DetailDataSet(2)
            detail.First()

            while not detail.EOF:
                if ((detail.Requisites("ДатаТ2").AsString ==
                     (DateCalculation.get_next_workday(-days_ago)).date().strftime("%d.%m.%Y"))
                        and (detail.Requisites("РаботникТ2").DisplayText in team)):
                    total += float(detail.Requisites("Сумма5Т2").AsString)
                detail.Next()
            self._next_record()

        self.delete_filter(yesterday)
        self.delete_filter(request)
        logging.info(f"Рассчитано время на запросы {total} ч.")
        return total

    def tickets_in_work(self):
        """
        Считаем количество обращений в работе
        :return:словарь в формате "Месяц ГОД": количесвтво обращений
        """
        logging.info(f"Start tickets_in_work")
        in_work = self.set_filter("СостОбращения", ['Р'])
        self.reference.Open()
        self.reference.First()
        created_month = dict()
        monthly_requests_count = dict()
        months = {"01": "Январь", "02": "Февраль", "03": "Март", "04": "Апрель", "05": "Май", "06": "Июнь",
                  "07": "Июль",
                  "08": "Август", "09": "Сентябрь", "10": "Октябрь", "11": "Ноябрь", "12": "Декабрь"}

        while not self.reference.EOF:
            created_date = self.reference.Requisites("ДатОткр").AsString
            month = f"{months[created_date[3:5]]} {created_date[6:]}"
            if month not in created_month:
                created_month[month] = 1
                monthly_requests_count[month] = list()
                monthly_requests_count[month].append(f"{self.reference.Requisites('Код').AsString}")
            else:
                created_month[month] += 1
                monthly_requests_count[month].append(f"{self.reference.Requisites('Код').AsString}")
            self._next_record()
        self.delete_filter(in_work)
        logging.debug(f"Обращения в работе по месяцам {created_month}")
        logging.debug(f"Обращения в работе по месяцам {monthly_requests_count}")
        logging.info(f"Done tickets_in_work")
        return created_month


    @staticmethod
    def get_days_ago(day_count):
        """
        Считаем вчерашний день.
        Если сегодня понедельник посчитаем и субботу с воскресеньем
        :return: Список с датами
        """
        day_list = list()
        current_day = (datetime.today() - timedelta(days=day_count))
        day_list.append(f"{current_day.day}.{current_day.month}.{current_day.year}")
        if day_count == 0:
            return day_list

        if datetime.today().isoweekday() == 1:
            for day in range(1, 3):
                day_ago = current_day - timedelta(days=day)
                day_ago_string = f"{day_ago.day}.{day_ago.month}.{day_ago.year}"
                day_list.append(day_ago_string)
        return day_list

    def not_closed(self, days=None, active=False):
        """
        Возвращает количество и список обращений, находящихся в работе больше указанного количества времени
        :param days:количество дней после которого обращения включаются в выборку
        :param active:флаг учитывать ли обращения на контроле (по умолчанию не учитываются)
        :return:
        """
        if active:
            ticket_status = ['Р']
        else:
            ticket_status = ['Р', 'К', 'И', 'П']
        ticket_status_filter = self.set_filter("СостОбращения", ticket_status)
        tickets_type = ['И', 'К', 'З']
        tickets_type_filter = self.set_filter("ТипОбращения", tickets_type)
        # Убираем анонимки
        not_anonymous = '30262732'  # анонимки (Код-338)
        not_anonymous_filter = self.set_filter("ОбластьПоддержки", not_anonymous, comparison_type=False)
        if days:
            days_offset_query = (f"({self.reference.TableName}.{self.reference.Requisites('ДатОткр').FieldName} <="
                     f"'{(datetime.today() - timedelta(days=(days))).date().strftime('%d.%m.%Y')}')")
            days_offset = self.reference.AddWhere(days_offset_query)
        self.reference.Open()
        self.reference.First()

        list_of_tickets = TicketsList()

        while not self.reference.EOF:
            list_of_tickets.add(
                ticket_id=self.reference.Requisites("Код").AsString,
                description=self.reference.Requisites("Содержание").AsString,
                hyperlink=self.reference.Hyperlink,
                status=self.reference.Requisites("СостОбращения").DisplayText,
                employee=self.reference.Requisites("Работник").DisplayText,
                priority=self.reference.Requisites("Строка3").AsString,
                organization=self.reference.Requisites("Организация").DisplayText,
                opening_date=self.reference.Requisites("ДатОткр").AsString
            )
            self._next_record()
        self.delete_filter(ticket_status_filter)
        self.delete_filter(tickets_type_filter)
        self.delete_filter(not_anonymous_filter)
        if days:
            self.reference.DelWhere(days_offset)

        return list_of_tickets

    def time_zones(self, tickets_type, zone_identifier):
        time_zone, ticket_time = asyncio.run(self.async_time_zones(tickets_type, zone_identifier))
        return time_zone, ticket_time

    async def async_time_zones(self, tickets_type, zone_identifier):
        """
        :param tickets_type: тип обращений (Инциденты, консультации)
        :param zone_identifier: метод вычисления цветной зоны
        :return:Словарь (зона: количество), список обращений, попавших в расчет
        """
        tickets_type_filter = self.set_filter("ТипОбращения", tickets_type)
        ticket_status = ['Р', 'К', 'И', 'П']
        ticket_status_filter = self.set_filter("СостОбращения", ticket_status)

        # Убираем анонимки
        not_anonymous = '30262732'  # анонимки (Код-338)
        not_anonymous_filter = self.set_filter("ОбластьПоддержки", not_anonymous, comparison_type=False)

        tickets_list = TicketsList()
        SLA_for_priority = {
            "Критический": 4*60,
            "Высокий": 8*60,
            "Средний": 16*60,
            "Низкий": 80*60,
            "Планируемый": 168*60
        }

        self.reference.Open()
        self.reference.First()

        time_zone = {"red": 0, "yellow": 0, "sandy": 0, "green": 0}

        while not self.reference.EOF:
            await click_yes()
            self.reference.OpenRecord()
            detail = self.reference.DetailDataSet(4)
            detail.First()
            time_in_work_duration = 0
            start = datetime.now()
            end = datetime.now()
            has_start = False
            has_end = False

            while not detail.EOF:
                res = 0
                if detail.Requisites("СостОбращенияТ4").AsString == "В работе" and not has_start:
                    start = datetime.strptime(str(detail.Requisites('ДатаВремяT4').AsString), '%d.%m.%Y %H:%M:%S')
                    has_start = True

                elif (detail.Requisites("СостОбращенияТ4").AsString in ["На контроле", "Переадресовано"]) and has_start:
                    end = datetime.strptime(str(detail.Requisites('ДатаВремяT4').AsString), '%d.%m.%Y %H:%M:%S')
                    has_end = True

                detail.Next()
                if detail.EOF:
                    if has_end:
                        pass
                    else:
                        end = datetime.now()
                        has_end = True

                if has_start and has_end:
                    time_start = time.fromisoformat('09:00')
                    time_end = time.fromisoformat('17:00')
                    t = start
                    while (t := t + timedelta(minutes=1)) <= end:
                        if ((t.weekday() < 5 and time_start <= t.time() <= time_end
                             and t.date().strftime("%d-%m") not in params.holidays)
                                or (t.date().strftime("%d-%m") in params.working_holidays
                                    and time_start <= t.time() <= time_end)):
                            res += 1
                    has_start = False
                    has_end = False

                time_in_work_duration += res

            tickets_list.add(
                ticket_id=self.reference.Requisites("Код").AsString,
                description=self.reference.Requisites("Содержание").AsString,
                hyperlink=self.reference.Hyperlink,
                status=self.reference.Requisites("СостОбращения").DisplayText,
                employee=self.reference.Requisites("Работник").DisplayText,
                priority=self.reference.Requisites("Строка3").AsString,
                organization=self.reference.Requisites("Организация").DisplayText,
                opening_date=self.reference.Requisites("ДатОткр").AsString,
                time_in_work=round(time_in_work_duration / 60, 2),
                sla_percent=round(((time_in_work_duration /
                                    SLA_for_priority[self.reference.Requisites("Строка3").AsString]))*100, 0)
            )
            tickets_list.tickets = tickets_list.tickets.sort_values(by=["Время в работе"], ascending=False)

            self.reference.Cancel()
            self.reference.CloseRecord()
            time_zone[zone_identifier(time_in_work_duration, self.reference.Requisites("Строка3").AsString)] += 1
            self._next_record()
        self.delete_filter(ticket_status_filter)
        self.delete_filter(tickets_type_filter)
        self.delete_filter(not_anonymous_filter)
        return time_zone, tickets_list


    @staticmethod
    def return_zone_for_time_in_work(time_in_work, priority) -> str:
        if time_in_work < 480:
            return 'green'
        elif 480 <= time_in_work < 960:
            return 'sandy'
        elif 960 <= time_in_work < 1440:
            return 'yellow'
        else: # time_in_work >= 1440
            return 'red'

    @staticmethod
    def return_zone_taking_into_account_priority(time_in_work, priority) -> str:
        """
        Возвращает зону с учётом приоритета
        :param time_in_work:
        :param priority:
        :return:
        """
        SLA_for_priority = {
            "Критический": 4*60,
            "Высокий": 8*60,
            "Средний": 16*60,
            "Низкий": 80*60,
            "Планируемый": 168*60
        }
        zone_percent = time_in_work / SLA_for_priority[priority]
        if zone_percent < 0.25:
            return "green"
        elif 0.25 <= zone_percent < 0.5:
            return "sandy"
        elif 0.5 <= zone_percent < 0.75:
            return "yellow"
        else: # 0.75 <= zone_percent
            return "red"

    @staticmethod
    def get_negative_grades(command_names, days_ago):
        """
        Считает оценки, полученные определенное количество дней назад
        :param command_names: список команды (имена и фамилии)
        :param days_ago: количество рабочих дней назад от сегодня (для пересчета за предыдущие дни)
        :return:количество плохих оценок, список оценок
        """
        logging.info(f"Start get_negative_grades")
        solution_marks = Reference("REQUEST_SOLUTION_MARKS")
        tickets = Reference("ПДД")

        current_day = Reference.get_days_ago(
            days_ago)  # Если сегодня понедельник посчитаем и субботу с воскресеньем
        logging.debug(f"Расчет за {current_day}")

        solution_marks.set_filter("ДатЗакр", current_day)

        solution_marks.reference.Open()
        solution_marks.reference.First()

        bad_marks = 0
        list_of_grades = list()

        while not solution_marks.reference.EOF:
            solution_marks.reference.OpenRecord()
            emoloyee = tickets.set_filter('Код',
                                          [f'    {solution_marks.reference.Requisites("Обращение").AsInteger}'])
            tickets.reference.Open()
            tickets.reference.First()
            if tickets.reference.Requisites("Работник").DisplayText in command_names:
                list_of_grades.append([solution_marks.reference.Requisites("Обращение").AsInteger,
                                       solution_marks.reference.Requisites("Текст").AsString.replace("=", ""),
                                       "Хорошо" if solution_marks.reference.Requisites("ISBIntNumber").AsInteger == 4
                                       else "Плохо",
                                       tickets.reference.Requisites("Работник").DisplayText,
                                       tickets.reference.Requisites("Содержание").AsString,
                                       tickets.reference.Requisites("Организация").DisplayText])
                if solution_marks.reference.Requisites("ISBIntNumber").AsInteger == 2:
                    bad_marks += 1
            tickets.delete_filter(emoloyee)
            tickets.reference.Close()
            solution_marks._next_record()
        logging.debug(f"Оценки: {list_of_grades}")
        logging.info(f"Done get_negative_grades")
        return bad_marks, list_of_grades


def duration_in_work(reference):
    reference.OpenRecord()
    detail = reference.DetailDataSet(4)
    detail.First()
    time_in_work_duration = 0
    start = datetime.datetime.now()
    end = datetime.datetime.now()
    while not detail.EOF:
        res = 0
        if detail.Requisites("СостОбращенияТ4").AsString == "В работе":
            start = detail.Requisites('ДатаВремяT4').AsDate
        elif detail.Requisites("СостОбращенияТ4").AsString == "На контроле":
            end = detail.Requisites('ДатаВремяT4').AsDate
            time_start = time.fromisoformat('09:00')
            time_end = time.fromisoformat('17:00')
            res = 0
            t = start
            while (t := t + datetime.timedelta(minutes=1)) <= end:
                if t.weekday() < 5 and time_start <= t.time() <= time_end:
                    res += 1

        time_in_work_duration += res
        # print(detail.Requisites("РаботникТ4").AsString)
        # print(detail.Requisites("СостОбращенияТ4").AsString)
        # print(detail.Requisites('ДатаВремяT4').AsDate)
        detail.Next()
    reference.Cancel()
    reference.CloseRecord()
    return round(time_in_work_duration/60, 2)
