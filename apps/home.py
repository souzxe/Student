import streamlit as st
import pandas as pd
from pathlib import Path
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.express as px


@st.cache_data
def get_data_info(df):
    info = pd.DataFrame()
    info.index = df.columns
    info['Тип данных'] = df.dtypes
    info['Уникальных'] = df.nunique()
    info['Количество значений'] = df.count()
    return info


@st.cache_data
def get_profile_report(df):
    from pandas_profiling import ProfileReport
    pr = ProfileReport(df)
    return pr


@st.cache_data
def create_histogram(df, column_name):
    fig = px.histogram(
        df,
        x=column_name,
        marginal="box",
        color='sex',
        title=f"Распределение {column_name}",
    )
    return fig


@st.cache_data
def create_correlation_matrix(df, features):
    corr = df[features].corr().round(2)
    fig1 = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale='plotly3',
        annotation_text=corr.values
    )
    fig1.update_layout(height=800)

    # Выбираем только корреляцию с целевым признаком 'cnt'
    corr = corr['G3'].drop('G3')
    corr = corr[abs(corr).argsort()]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=corr.values,
        y=corr.index,
        orientation='h',
        marker_color=list(range(len(corr.index)))
    ))
    fig2.update_layout(
        title='Корреляция с G3',
        height=700,
        xaxis=dict(title='Признак'),  # Название оси x
        yaxis=dict(title='Корреляция'),
    )
    return fig1, fig2


@st.cache_data
def get_simple_histograms(df, selected_category):
    fig = px.histogram(
        df,
        x=selected_category,
        color=selected_category,
        title=f'Распределение по {selected_category}',
        template='plotly'
    )
    return fig


@st.cache_data
def display_age_distribution(df):
    fig = px.box(
        df,
        x="education",
        y="age",
        color="education",
        title="Распределение возраста по уровням образования",
        labels={"age": "Возраст", "education": "Образование"}
    )
    fig.update_layout(height=700)
    return fig


@st.cache_data
def display_age_distribution_by_marital_status(df):
    fig = px.box(
        df,
        x='marital',
        y='age',
        color='y',
        title='Распределение возраста по семейному положению и откликам',
        facet_col='y',
        labels={'marital': 'Семейное положение', 'y': 'Отклик'}
    )
    fig.update_layout(height=700)
    return fig


# @st.cache_data
def create_pairplot(df, selected_features, hue=None):
    sns.set_theme(style="whitegrid")
    pairplot_fig = sns.pairplot(
        df,
        vars=selected_features,
        hue=hue,
        plot_kws={'s': 80, 'edgecolor': 'k'},
        height=3
    )
    plt.subplots_adjust(top=0.95)
    return pairplot_fig


def display_metrics(df):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Количество студентов", value=len(df))
    with col2:
        st.metric(label="Средний возраст", value=round(df['age'].mean(), 2))
    with col3:
        st.metric(label="Среднее количество пропусков занятий", value=round(df['absences'].mean(), 2))


def display_box_plot(df, numerical_features, categorical_features):
    c1, c2, c3 = st.columns(3)
    feature1 = c1.selectbox('Первый признак', numerical_features, key='box_feature1')
    feature2 = c2.selectbox('Второй признак', categorical_features, key='box_feature2')
    filter_by = c3.selectbox('Фильтровать по', [None, *categorical_features], key='box_filter_by')

    if feature2 == filter_by:
        filter_by = None

    fig = px.box(
        df,
        x=feature1, y=feature2,
        color=filter_by,
        title=f"Распределение {feature1} по разным {feature2}",
    )
    fig.update_layout(height=900)
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def plot_g123(df):
    # Настройки шрифтов
    plt.rcParams.update({'font.size': 14})

    # Создание сетки графиков
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Гистограммы для оценок G1, G2, G3
    for i, col in enumerate(['G1', 'G2', 'G3']):
        sns.histplot(data=df, x=col, ax=axes[0, i], color='blue', kde=True)
        axes[0, i].set_title(f'Распределение оценок {col}', fontsize=18)
        axes[0, i].set_xlabel('Оценки', fontsize=16)
        axes[0, i].set_ylabel('Количество студентов', fontsize=16)

    # Ящики с усами для оценок G1, G2, G3
    for i, col in enumerate(['G1', 'G2', 'G3']):
        sns.boxplot(data=df, x=col, ax=axes[1, i], color='skyblue')
        axes[1, i].set_title(f'Ящик с усами для оценок {col}', fontsize=18)
        axes[1, i].set_xlabel('Оценки', fontsize=16)
        axes[1, i].set_ylabel('')
        axes[1, i].set_xticklabels(axes[1, i].get_xticklabels(), rotation=90, fontsize=14)

    plt.tight_layout()
    st.pyplot(plt)


@st.cache_data
def plot_custom_boxplots(df):
    sns.set_palette('pastel')
    # Настройки шрифтов
    plt.rcParams.update({'font.size': 14})

    # Названия характеристик на русском
    features_ru = ['Поддержка в школе', 'Семейная поддержка', 'Платные дополнительные занятия',
                   'Внеклассные мероприятия', 'Посещение детского сада', 'Желание получить высшее образование',
                   'Наличие интернета дома', 'Романтические отношения']

    # Создание сетки графиков
    fig, axes = plt.subplots(4, 2, figsize=(15, 19))

    # Увеличение расстояний между графиками
    plt.subplots_adjust(wspace=0.4, hspace=0.6)

    # Создание boxplot для каждой характеристики
    for i, item in enumerate(
            ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']):
        ax = axes[i // 2, i % 2]  # Определение позиции графика в сетке
        order_by = df.groupby(item)['G3'].median().sort_values(ascending=False).index
        sns.boxplot(x=df[item], y=df['G3'], ax=ax, order=order_by)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=12)
        ax.set_title(f'{features_ru[i]} и оценки G3', fontsize=16)
        ax.set_xlabel('')
        ax.set_ylabel('Оценки G3', fontsize=14)

    plt.tight_layout()
    st.pyplot(fig)


@st.cache_data
def plot_custom_visualizations(df):
    # Установка темы и палитры
    sns.set_theme(style='whitegrid')

    # Настройки шрифтов
    plt.rcParams.update({'font.size': 14})

    # Пастельные цвета
    colors = sns.color_palette('pastel')

    # Гистограммы для числовых признаков: возраст, пропуски, итоговые оценки
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df['age'], bins=8, ax=axes[0], color=colors[0], kde=True)
    axes[0].set_title('Распределение возраста студентов', fontsize=16)
    axes[0].set_xlabel('Возраст', fontsize=14)
    axes[0].set_ylabel('Количество студентов', fontsize=14)

    sns.histplot(df['absences'], bins=30, ax=axes[1], color=colors[1], kde=True)
    axes[1].set_title('Распределение пропущенных занятий', fontsize=16)
    axes[1].set_xlabel('Пропущенные занятия', fontsize=14)
    axes[1].set_ylabel('Количество студентов', fontsize=14)

    sns.histplot(df['G3'], bins=15, ax=axes[2], color=colors[2], kde=True)
    axes[2].set_title('Распределение итоговых оценок', fontsize=16)
    axes[2].set_xlabel('Итоговые оценки', fontsize=14)
    axes[2].set_ylabel('Количество студентов', fontsize=14)

    plt.tight_layout()
    st.pyplot(fig)

    # Ящики с усами для возраста, пропусков, итоговых оценок
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    sns.boxplot(x=df['age'], ax=ax[0], color=colors[0])
    ax[0].set_title('Ящик с усами для возраста', fontsize=16)
    ax[0].set_xlabel('Возраст', fontsize=14)

    sns.boxplot(x=df['absences'], ax=ax[1], color=colors[1])
    ax[1].set_title('Ящик с усами для пропусков', fontsize=16)
    ax[1].set_xlabel('Пропущенные занятия', fontsize=14)

    sns.boxplot(x=df['G3'], ax=ax[2], color=colors[2])
    ax[2].set_title('Ящик с усами для итоговых оценок', fontsize=16)
    ax[2].set_xlabel('Итоговые оценки', fontsize=14)

    plt.tight_layout()
    st.pyplot(fig)

    # Столбчатые диаграммы для пола, школы, желания получить высшее образование
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.countplot(x='sex', data=df, ax=axes[0])
    axes[0].set_title('Распределение по полу', fontsize=16)
    axes[0].set_xlabel('Пол', fontsize=14)
    axes[0].set_ylabel('Количество студентов', fontsize=14)

    sns.countplot(x='school', data=df, ax=axes[1])
    axes[1].set_title('Распределение по школам', fontsize=16)
    axes[1].set_xlabel('Школа', fontsize=14)
    axes[1].set_ylabel('Количество студентов', fontsize=14)

    sns.countplot(x='higher', data=df, ax=axes[2])
    axes[2].set_title('Желание получить высшее образование', fontsize=16)
    axes[2].set_xlabel('Желание получить высшее образование', fontsize=14)
    axes[2].set_ylabel('Количество студентов', fontsize=14)

    plt.tight_layout()
    st.pyplot(fig)


@st.cache_data
def plot_gender_exam_comparison(df):
    custom_colors = ['#ff9999', '#004c99']
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for i in range(3):
        sns.barplot(x='sex', y=f'G{i + 1}', data=df, ci=None, ax=axs[i], palette=custom_colors)
        axs[i].set_xlabel('Пол', fontsize=14)
        axs[i].set_ylabel(f'Экзамен {i + 1}', fontsize=14)
        axs[i].set_xticks([0, 1])
        axs[i].set_xticklabels(["Женский", "Мужской"], fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)


@st.cache_data
def plot_final_grade_comparison(df):
    custom_colors = ['#ff9999', '#004c99']
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=df, x="sex", y="G3", hue="school",
        ci=None, palette=custom_colors, ax=ax
    )
    ax.set_xticklabels(["Женский", "Мужской"], fontsize=12)
    ax.set_xlabel("Пол", fontsize=14)
    ax.set_ylabel("Итоговая оценка", fontsize=14)
    ax.legend(title="Школа", fontsize=12)
    ax.set_title("Сравнение итоговых оценок G3 по полу и школам", fontsize=16)

    # Подгонка макета
    plt.tight_layout()
    st.pyplot(fig)


def app(df, current_dir: Path):
    st.title("Анализ успеваемости студентов")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            ## Область применения
            Для нашего случая мы выбираем область образования, конкретно анализ данных о студентах. Это позволит выявить ключевые факторы, влияющие на успеваемость студентов, и способствует оптимизации учебных программ и личностного развития студентов.
            
            Анализ данных — это процесс исследования, очистки, преобразования и моделирования данных с целью обнаружения полезной информации, подтверждения выводов и поддержки принятия решений. В данном случае, мы сосредоточимся на анализе данных студентов, загруженных из предоставленного файла CSV. Этот процесс включает несколько ключевых этапов, начиная от первичного анализа и обработки данных до глубокого статистического анализа и визуализации результатов.
            """
        )
    with col2:
        st.image(str(current_dir / "images" / "main1.png"), use_column_width='auto')

    st.markdown("""
        ## Ключевые параметры и характеристики данных
        Перед началом анализа необходимо понять, какие данные у нас есть и как они структурированы. Для этого сначала загрузим данные из файла и изучим структуру таблицы, описав каждую колонку.
    """)
    tab1, tab2 = st.tabs(["Показать описание данных", "Показать пример данных"])
    with tab1:
        st.markdown(
            r"""
            ## Описание данных
           
            В датасете присутствуют следующие параметры:
            
            | Параметр    | Описание                                                       |
            |-------------|----------------------------------------------------------------|
            | school      | Школа, которую посещает ученик (GP - Gabriel Pereira, MS - Mousinho da Silveira) |
            | sex         | Пол ученика (M - мужской, F - женский)                         |
            | age         | Возраст ученика                                                |
            | address     | Тип адреса ученика (U - городской, R - сельский)               |
            | famsize     | Размер семьи (LE3 - до трех человек, GT3 - больше трех)        |
            | Pstatus     | Статус совместного проживания родителей (T - вместе, A - раздельно) |
            | Medu        | Образование матери (от 0 - нет до 4 - высшее)                  |
            | Fedu        | Образование отца (от 0 - нет до 4 - высшее)                    |
            | Mjob        | Работа матери                                                  |
            | Fjob        | Работа отца                                                    |
            | reason      | Причина выбора школы (home - близость к дому, reputation - репутация школы и т.д.) |
            | guardian    | Опекун                                                         |
            | traveltime  | Время в пути до школы (от 1 - <15 мин. до 4 - >1 час)          |
            | studytime   | Время на учёбу помимо школы в неделю (от 1 - <2 часов до 4 - >10 часов) |
            | failures    | Количество внеучебных неудач                                  |
            | schoolsup   | Дополнительная образовательная поддержка (yes или no)         |
            | famsup      | Семейная образовательная поддержка (yes или no)               |
            | paid        | Дополнительные платные занятия по предмету курса (yes или no) |
            | activities  | Внеучебные активности (yes или no)                            |
            | nursery     | Посещал ли ученик детский сад (yes или no)                    |
            | higher      | Хочет ли ученик получить высшее образование (yes или no)      |
            | internet    | Наличие интернета дома (yes или no)                           |
            | romantic    | Наличие романтических отношений (yes или no)                  |
            | famrel      | Качество семейных отношений (от 1 - очень плохо до 5 - отлично)|
            | freetime    | Свободное время после школы (от 1 - очень мало до 5 - очень много)|
            | goout       | Проведение времени с друзьями (от 1 - очень мало до 5 - очень много)|
            | Dalc        | Потребление алкоголя в будни (от 1 - очень мало до 5 - очень много)|
            | Walc        | Потребление алкоголя в выходные (от 1 - очень мало до 5 - очень много)|
            | health      | Текущее состояние здоровья (от 1 - очень плохо до 5 - отлично)|
            | absences    | Количество пропущенных занятий                                |
            | G1          | Оценки за первый период                                       |
            | G2          | Оценки за второй период                                       |
            | G3          | Итоговые оценки                                               |                                     |
            """
        )
    with tab2:
        st.header("Пример данных")
        st.dataframe(df.head(50), height=600)

    categorical_features = df.select_dtypes(include='category').columns.tolist()
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    st.header("Предварительный анализ данных")
    st.dataframe(get_data_info(df), use_container_width=True)

    st.markdown("""
        В ходе первичного анализа данных были получены следующие выводы:
        * Все колонки полностью заполнены, пропущенные значения отсутствуют.
        * Данные содержат как числовые, так и категориальные признаки.
        * Преобразование некоторых текстовых признаков в категориальные типы было выполнено для оптимизации обработки и анализа.
        
        Как видим, данные полные, пропусков нет, поэтому нет необходимости заполнять пропуски.
    """)

    st.header("Основные статистики для признаков")
    display_metrics(df)

    tab1, tab2 = st.tabs(["Числовые признаки", "Категориальные признаки"])
    with tab1:
        st.header("Рассчитаем основные статистики для числовых признаков")
        st.dataframe(df.describe(), use_container_width=True)
        st.markdown("""
             Основные статистические показатели для числовых данных были рассчитаны:
            * Статистика показывает значительное разнообразие по возрасту, оценкам и пропускам учебных занятий среди учащихся.
            * Возраст студентов колеблется от 15 до 22 лет.
            * Оценки (G1, G2, G3) имеют стандартные отклонения, которые указывают на различие в академической успеваемости студентов.
            * Средние оценки укладываются в диапазон примерно от 10 до 11 баллов, что может указывать на среднюю сложность учебного материала или критериев оценки.
        """)
    with tab2:
        st.header("Рассчитаем основные статистики для категориальных признаков")
        st.dataframe(df.describe(include='category'), use_container_width=True)
        st.markdown("""
            Для категориальных признаков было рассчитано количество уникальных значений, выявив:
            * Различие в выборе школ, профессий родителей, причин поступления в школу и так далее.
            * Например, большинство студентов посещают школу GP, что может говорить о большей популярности или доступности этой школы.
        """)

    st.header("Визуализация данных")

    st.subheader("Визуализация числовых признаков")
    selected_feature = st.selectbox(
        "Выберите признак",
        numerical_features,
        key="create_histogram_selectbox1"
    )
    hist_fig = create_histogram(
        df,
        selected_feature
    )
    st.plotly_chart(hist_fig, use_container_width=True)
    st.markdown("""
        ## Гистограммы и ящики с усами

        Гистограмма — это тип диаграммы, который используется для отображения распределения числовых данных. 
        Она помогает оценить плотность вероятности распределения данных. 
        Гистограммы идеально подходят для иллюстрации распределений признаков, таких как возраст студентов, количество пропущенных занятий или оценка за экзамен.

        Ящик с усами — это еще один вид графика для визуализации распределения числовых данных. 
        Он показывает медиану, первый и третий квартили, а также "усы", которые простираются до крайних точек данных, не считая выбросов. Ящики с усами особенно полезны для сравнения распределений между несколькими группами и выявления выбросов.
    """)

    display_box_plot(
        df,
        numerical_features,
        categorical_features
    )

    st.markdown("""
        ### Анализ числовых признаков

        Анализ гистограмм

        Распределение оценок (G1, G2, G3):
        * Все три гистограммы показывают нормальное распределение с некоторыми отклонениями. Например, в распределении G1 и G3 заметны выбросы оценок в нижней части шкалы, что может указывать на наличие учащихся с особыми образовательными потребностями или проблемами в учёбе.
        * Оценки G2 показывают более сглаженное распределение, что может быть связано с корректировкой учебных планов или усилением поддержки студентов после первого периода.

        Ящики с усами для оценок G1, G2, G3:
        * Ящики показывают, что медианные значения и квартили оценок схожи между различными периодами, хотя для G3 ящик чуть уже, что свидетельствует о меньшем разбросе оценок. Наличие выбросов в сторону нижних оценок подтверждает выводы, сделанные на основе гистограмм.
    """)
    plot_g123(df)

    st.markdown("""
        ## Ящики с усами для числовых признаков
        
        ### Анализ влияния различных факторов на оценки в финальном экзамене
        
        На представленных графиках исследуется взаимосвязь между определёнными факторами учащихся и их оценками за первый экзамен (G1). Рассмотрим, как разные аспекты жизни и поддержки влияют на успеваемость учеников.
        1. Поддержка в школе (schoolsup)
           * Наличие поддержки в школе: Ученики, получающие поддержку в школе (оранжевый), имеют заметно более низкие медианные оценки по сравнению с теми, кто такой поддержки не получает (синий). Это может указывать на то, что школьная поддержка чаще всего направляется студентам, испытывающим трудности с учёбой.
        
        2. Семейная поддержка (famsup)
           * Наличие семейной поддержки: Наличие или отсутствие семейной поддержки не показывает значительного различия в медианных оценках, что может свидетельствовать о разнообразии семейных обстоятельств и их сложном влиянии на академическую успеваемость.
        
        3. Платные дополнительные занятия (paid)
           *  Платные занятия: Ученики, посещающие платные занятия, в среднем показывают лучшие результаты, что может отражать высокую мотивацию или более высокие ресурсы их семей для инвестиций в образование.
        
        4. Внеклассные мероприятия (activities)
           * Участие во внеклассных мероприятиях: Учащиеся, участвующие во внеклассных мероприятиях, имеют схожие медианные оценки с теми, кто не участвует. Это может указывать на то, что внеклассные мероприятия не оказывают значительного влияния на академические оценки.
        
        5. Посещение детского сада (nursery)
           * Посещение детского сада: Студенты, которые посещали детский сад, показывают немного лучшие оценки, что может быть связано с ранним развитием учебных навыков.
        
        6. Желание получить высшее образование (higher)
           * Стремление к высшему образованию: Ожидаемо, что учащиеся, стремящиеся получить высшее образование, показывают лучшие результаты. Это подчеркивает влияние учебной мотивации на академическую успеваемость.
        
        7. Наличие интернета дома (internet)
           * Интернет дома: Учащиеся с доступом к интернету дома показывают выше средние оценки, что подчеркивает роль интернета в обеспечении образовательных ресурсов и поддержки.
        
        8. Романтические отношения (romantic)
           * Романтические отношения: Наличие романтических отношений ассоциируется с немного более низкими оценками. Это может отражать отвлечение внимания от учебы на личные отношения.
        
        Общий вывод:
        * Анализ показывает, что различные формы поддержки и условия жизни учащихся имеют значительное влияние на их академические достижения. Важно, чтобы образовательные учреждения и родители учитывали эти факторы при планировании образовательной поддержки, чтобы максимизировать академический потенциал каждого студента.
        """)

    plot_custom_boxplots(df)

    st.markdown("""
        ### Анализ ящиков с усами
        Возраст студентов:
        * Распределение возраста показывает, что большинство студентов находятся в возрасте от 16 до 18 лет. Медиана возраста приходится на 17 лет. Наличие немногочисленных выбросов в сторону более старших возрастов (20 и 22 года) может указывать на задержку в обучении или поздний переход в старшие классы.
        
        Пропуски занятий:
        * Медианное значение пропусков близко к 4, но распределение имеет длинный правый "хвост", что свидетельствует о наличии студентов с очень высоким уровнем пропусков, достигающих 75 дней. Это может быть связано с различными социальными, медицинскими или личными обстоятельствами.
        
        Итоговые оценки (G3):
        * Итоговые оценки студентов варьируются от 0 до 20, при этом медиана находится около 11, что указывает на средний уровень успеваемости. Ширина ящика свидетельствует о значительном разбросе результатов, что может отражать различия в учебных способностях, мотивации и доступных ресурсах среди студентов.
    """)

    plot_custom_visualizations(df)

    tab1, tab2 = st.tabs(["Простые графики", "Показать отчет о данных"])
    with tab1:
        st.subheader("Распределение студентов")
        st.subheader("Столбчатые диаграммы для категориальных признаков")
        selected_category_simple_histograms = st.selectbox(
            'Категория для анализа',
            categorical_features,
            key='category_get_simple_histograms'
        )
        st.plotly_chart(get_simple_histograms(df, selected_category_simple_histograms), use_container_width=True)

    with tab2:
        if st.button("Сформировать отчёт", use_container_width=True, type='primary'):
            st_profile_report(get_profile_report(df))

    st.markdown("""
        ## Играет ли пол какую-либо существенную роль в оценках учащихся?
        Из ряда графиков видно, что в каждом из трех экзаменов мужчины в среднем получают более высокие оценки, чем женщины. Это может быть связано с более высокой успеваемостью мужчин в академической среде, что поддерживается множеством исследований, указывающих на лучшую успеваемость мужчин в школьных предметах.
    """)
    plot_gender_exam_comparison(df)
    st.markdown("""
        ## Анализ графика итоговых оценок по полу в разных школах
        
        На этом графике представлены итоговые оценки, разделенные по полу учащихся и по школам (GP и MS). Видно, что:
        * Школа GP: В этой школе женщины и мужчины показывают примерно одинаковые результаты, что может свидетельствовать о равных возможностях для обоих полов в данной образовательной среде.
        * Школа MS: Здесь также наблюдается схожий уровень оценок между полами, хотя стоит отметить, что разница между школами в целом минимальна, что говорит о стандартизации учебных программ или методов оценки.
    """)
    plot_final_grade_comparison(df)

    @st.cache_data
    def plot_father_education_comparison(df):
        # Описание уровня образования отца
        education_descriptions = {
            0: "Нет",
            1: "Начальное (4 класса)",
            2: "С 5 по 9 класс",
            3: "Среднее образование",
            4: "Высшее образование"
        }

        # Список для оценок
        grades = ['G1', 'G2', 'G3']

        # Создание сетки графиков 1x3
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Создание графиков
        for i, grade in enumerate(grades):
            ax = sns.barplot(x='Fedu', y=grade, data=df, ci=None, ax=axs[i], palette="pastel")
            ax.set_xticklabels([education_descriptions.get(item, item) for item in sorted(df['Fedu'].unique())],
                               rotation=45, fontsize=12)
            ax.set_xlabel("Уровень образования отца", fontsize=14)
            ax.set_ylabel('Средний балл', fontsize=14)
            ax.set_title(f'Средний балл за {grade}', fontsize=16)

        # Подгонка макета
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("""
        ### Анализ графиков оценок по уровню образования отца
        
        Графики оценок G1, G2 и G3 в зависимости от уровня образования отца показывают интересную динамику:
        * Более высокие оценки у студентов с необразованными отцами: Это контринтуитивное наблюдение может отражать компенсаторные усилия со стороны студентов, которые стремятся компенсировать недостаток образовательных ресурсов в семье через усиленную самостоятельную работу или использование внешних ресурсов. Такой подход может быть мотивирован желанием изменить семейные традиции и достичь лучших жизненных результатов.
        * Общий тренд: На всех уровнях оценок наблюдается снижение средних значений от G1 к G3, что может указывать на увеличение сложности материала или усталость студентов к концу учебного периода.
    """)
    plot_father_education_comparison(df)

    st.header("Корреляционный анализ")
    st.markdown("""
        Матрица корреляции позволяет определить связи между признаками. Значения в матрице колеблются от -1 до 1, где:
        
        - 1 означает положительную линейную корреляцию,
        - -1 означает отрицательную линейную корреляцию,
        - 0 означает отсутствие линейной корреляции.
    """)
    fig1, fig2 = create_correlation_matrix(df, numerical_features)
    st.plotly_chart(fig1, use_container_width=True)

    markdown_col1, markdown_col2 = st.columns(2)
    markdown_col1.markdown("""
        Корреляционная матрица представляет связь между различными числовыми параметрами. В данном случае: 
        - **G1 (Оценка за первый период) и G2 (Оценка за второй период):** Имеют наибольшую положительную корреляцию с G3, с коэффициентами 0.8 и 0.9 соответственно. Это указывает на то, что хорошие оценки в начале учебного года сильно коррелируют с хорошими итоговыми оценками.
        - **Образование родителей (Medu, Fedu):** Положительная корреляция с G3 (0.22 для образования матери и 0.15 для образования отца). Это означает, что более высокое образование родителей может способствовать лучшим итоговым оценкам.
        - **Отсутствие неудач (failures):** Сильная отрицательная корреляция с G3 (-0.36). Студенты с большим количеством академических неудач имеют худшие итоговые оценки.
        - **Возраст (age):** Небольшая отрицательная корреляция (-0.16). Это может означать, что старшие студенты немного хуже справляются с экзаменами.
        - **Время на дорогу до школы (traveltime):** Небольшая отрицательная корреляция (-0.12). Большее время на дорогу может отрицательно сказаться на оценках.
        - **Проведение времени с друзьями (goout), потребление алкоголя (Dalc, Walc):** Негативная корреляция с G3 (-0.13 для goout, -0.05 для Dalc и Walc). Эти факторы могут отвлекать студентов от учебы и негативно влиять на их итоговые оценки.
        Из анализа видно, что на итоговые оценки студентов значительное влияние оказывают их промежуточные оценки, количество академических неудач, образовательный уровень родителей и время, затрачиваемое на учебу. Другие факторы, такие как возраст, время на дорогу до школы и социальные активности, также играют роль, но в меньшей степени.
    """)
    markdown_col2.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        """
        ## Точечные диаграммы для пар числовых признаков
        """
    )
    selected_features = st.multiselect(
        'Выберите признаки',
        numerical_features,
        default=numerical_features[::3],
        key='pairplot_vars'
    )

    # Опциональный выбор категориальной переменной для цветовой дифференциации
    hue_option = st.selectbox(
        'Выберите признак для цветового кодирования (hue)',
        ['None'] + categorical_features,
        index=2,
        key='pairplot_hue'
    )
    if hue_option == 'None':
        hue_option = None
    if selected_features:
        st.pyplot(create_pairplot(df, selected_features, hue=hue_option))
    else:
        st.error("Пожалуйста, выберите хотя бы один признак для создания pairplot.")
