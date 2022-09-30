import pandas as pd


from classes.contour import ContourOfCell, ContourOfTable, ContourOfLine
from classes.pageImage import PageImage


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
            'Координаты': pd.Series(dtype='str'),
            'КоординатаДляСортировки': pd.Series(dtype='int')})

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
            'Координаты': str([table.top_left, table.down_right]),
            'КоординатаДляСортировки': table.top_left[1]}, ignore_index=True)

    def add_line(self, line: ContourOfLine, page: int, filr_id: str):
        self.df = self.df.append({
            'Счет': filr_id,
            'Номер страницы': page,
            'Вид объекта': "Строка",
            'Номер таблицы': 0,
            'Строка': 0,
            'Колонка': 0,
            'Номер строки': line.number,
            'Текст': line.text,
            'Координаты': str([line.top_left, line.down_right]),
            'КоординатаДляСортировки': line.top_left[1]}, ignore_index=True)

    def sort(self):
        self.df = self.df.sort_values(by=['Номер страницы', 'КоординатаДляСортировки', 'Номер таблицы', 'Строка','Колонка','Номер строки'])
        # self.df = self.df.drop('КоординатаДляСортировки')
