import os
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt
import io
from PIL import Image
from nanodesclib.classes import *


colors = ['#0000FF', '#0AFFFF', '#22CE83', '#FFFF33', '#EB5406', '#F98B88', '#2B1B17', '#C4AEAD', '#D462FF', '#FAF0DD',
          '#F8B88B', '#B86500']


class DataAnalyzer:
    def __init__(self, df):
        self.df = df

    def hist(self, df, column):
        fig = make_subplots(rows=1, cols=2, subplot_titles=[f'Распределение {column}', f'Violin Plot для {column}'])
        hist_fig = px.histogram(df, x=column, nbins=60,
                                color_discrete_sequence=colors,
                                opacity=0.7)
        fig.add_trace(hist_fig['data'][0], row=1, col=1)
        violin_fig = px.violin(df, y=column, color_discrete_sequence=colors, box=True)
        fig.add_trace(violin_fig['data'][0], row=1, col=2)  # Добавление графика с указанием расположения
        fig.update_layout(showlegend=False, title_text="Гистограмма и Violin Plot")
        img_bytes = fig.to_image(format="png")
        image = Image.open(io.BytesIO(img_bytes))
        return image

    def pie(self, df, column, col_to_group):
        df_pie = pd.DataFrame(df.groupby(column)[col_to_group].count())
        df_pie = df_pie.reset_index()
        fig = px.pie(df_pie, values=col_to_group, names=column, color_discrete_sequence=colors,
                     title=f'{column} pie chart')
        img_bytes = fig.to_image(format="png")
        image = Image.open(io.BytesIO(img_bytes))
        return image

    def bar(self, df, column):
        plt.figure(figsize=(12, 6))
        self.df[column].explode().value_counts().plot(kind='bar')
        plt.title(column)
        plt.tight_layout()
        return plt

    def formula_check(self, df, column):
        incorrect = []
        for formula in tuple(set(df[column])):
            try:
                assign_class(formula).consist()
            except:
                incorrect.append(formula)
        return incorrect

    def types_check(self, df, column, output_dir):
        for t in df['type'].unique():
            fig = px.histogram(df[df['type'] == t], y=column, width=800, height=800)
            fig.update_layout(
                title={
                    'text': f'{t}',
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'})
            img_bytes = fig.to_image(format="png")
            image = Image.open(io.BytesIO(img_bytes))
            image.save(f"{output_dir}/{t}_hist.png")

    def perform_eda(self, bar_dict, pie_dict, hist_dict, column, output_dir: str = "eda_results"):
        os.makedirs(output_dir, exist_ok=True)

        self.types_check(self.df, column, output_dir)

        for key, value in bar_dict.items():
            self.bar(self.df, key).savefig(f"{output_dir}/{value}.png")

        for key, value in pie_dict.items():
            self.pie(self.df, key, column).save(f"{output_dir}/{value}_pie_chart.png")

        for key, value in hist_dict.items():
            self.hist(self.df, key).save(f"{output_dir}/{value}_hist.png")

        summary = {
            'Количество нулевых значений': sum(self.df.isna().sum()),
            'Количество строк': len(self.df),
            'Количество уникальных формул': len(set(self.df[column])),
            'Некорректные формулы': self.formula_check(self.df, column)
        }

        return summary
