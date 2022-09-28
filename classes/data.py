import pandas as pd


from classes.contour import ContourOfCell, ContourOfTable
from classes.page import Page


class Data:
    def __init__(self):
        self.df = pd.DataFrame({
            'Счет': pd.Series(dtype='str'),
            'Номер страницы': pd.Series(dtype='int'),
            'Вид объекта': pd.Series(dtype='str'),
            'Номер таблицы': pd.Series(dtype='int'),
            'Строка': pd.Series(dtype='int'),
            'Колонка': pd.Series(dtype='int'),
            'Номер строки': pd.Series(dtype='int'),
            'Текст': pd.Series(dtype='str'),
            'Координаты': pd.Series(dtype='str')})

    def add_cell(self, cell: ContourOfCell, table: ContourOfTable, page: int, filr_id: str):
        self.df = self.df.append({
            'Счет': filr_id,
            'Номер страницы': page,
            'Вид объекта': "Таблица",
            'Номер таблицы': table.number,
            'Строка': cell.row,
            'Колонка': cell.column,
            'Номер строки': 0,
            'Текст': cell.text,
            'Координаты': str([table.top_left, table.down_right])}, ignore_index=True)
