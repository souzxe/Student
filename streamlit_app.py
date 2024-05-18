import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from pathlib import Path
from apps import home, prediction
from plotly.io import templates
templates.default = "plotly"

st.set_page_config(
    page_title="Анализ данных студентов",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    # Преобразование столбцов к категориальным типам, где это уместно
    categorical_features = df.select_dtypes(include=['object']).columns
    for feature in categorical_features:
        df[feature] = df[feature].astype('category')
    return df


class Menu:
    apps = [
        {
            "func": home.app,
            "title": "Главная",
            "icon": "house-fill"
        },
        {
            "func": prediction.app,
            "title": "Прогнозирование",
            "icon":  "bar-chart-line-fill"
        },
    ]

    def run(self):
        with st.sidebar:
            titles = [app["title"] for app in self.apps]
            icons = [app["icon"] for app in self.apps]

            selected = option_menu(
                "Меню",
                options=titles,
                icons=icons,
                menu_icon="cast",
                default_index=0,
            )
            st.info(
                """
                ## Анализ данных студентов
                Эта система анализа данных предназначена для образовательного сектора с целью выявления ключевых факторов, влияющих на успеваемость студентов. Она анализирует данные о студентах, предоставляя ценные инсайты для оптимизации учебных программ и поддержки студентов.
                """
            )
        return selected


if __name__ == '__main__':
    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()

    df = preprocess_data(current_dir / 'student_data.csv')

    menu = Menu()
    st.sidebar.image(str(current_dir / 'images' / 'logo.png'))
    selected = menu.run()
    for app in menu.apps:
        if app["title"] == selected:
            app["func"](df, current_dir)
            break
