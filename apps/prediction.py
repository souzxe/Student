import joblib
from pathlib import Path
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from streamlit_option_menu import option_menu
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures


@st.cache_data
def plot_regression_results(y_test, y_pred):
    # Функция для визуализации результатов линейной регрессии.
    # Построение графика фактических значений против предсказанных линейной регрессией
    fig = go.Figure()

    # Добавление истинных значений
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            name='Predicted vs Actual',
            marker=dict(color='blue', size=10, opacity=0.5)
        )
    )

    # Линия идеального прогноза
    fig.add_trace(
        go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            name='Ideal Fit',
            line=dict(color='red', width=2)
        )
    )

    fig.update_layout(
        title='Actual vs. Predicted Values',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        legend_title='Type',
        xaxis=dict(showline=True),
        yaxis=dict(showline=True)
    )

    # Построение гистограммы ошибок
    errors = y_test - y_pred
    fig_errors = go.Figure()
    fig_errors.add_trace(
        go.Histogram(
            x=errors,
            nbinsx=50,
        )
    )
    fig_errors.update_layout(
        title='Distribution of Prediction Errors',
        xaxis_title='Prediction Error',
        yaxis_title='Frequency',
        xaxis=dict(showline=True),
        yaxis=dict(showline=True),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig_errors, use_container_width=True)


@st.cache_data
def prepare_data(X, y):
    model = sm.OLS(y, X).fit()
    y_pred = model.predict(X)
    return model, y_pred


def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r_squared = r2_score(y_true, y_pred)
    return rmse, mae, r_squared


def app(df, current_dir: Path):
    st.title("Прогнозирование итоговых оценок студентов")

    c1, c2 = st.columns(2)
    c1.markdown("""
    ### Описание страницы
    Страница прогнозирования позволяет пользователям ввести данные о студенте и получить прогнозируемую итоговую оценку (G3). На этой странице представлены интерактивные элементы для ввода параметров, таких как возраст студента, образование родителей, время на дорогу до школы и другие значимые характеристики. После ввода данных и нажатия кнопки "Прогнозировать" система использует модель машинного обучения для предсказания итоговой оценки студента. Эта функциональность позволяет образовательным учреждениям и родителям получить представление о потенциальных академических результатах студента на основе различных факторов.
    """)
    c2.image(str(current_dir / "images" / "main3.png"), width=150, use_column_width='auto')

    df_encoded = df.copy(deep=True)
    st.markdown(
        """
        # Подготовка набора данных
        Перед подачей наших данных в модель машинного обучения нам сначала нужно подготовить данные. Это включает в себя кодирование всех категориальных признаков (либо LabelEncoding, либо OneHotEncoding), поскольку модель ожидает, что признаки будут представлены в числовой форме. Также для лучшей производительности мы выполним масштабирование признаков, то есть приведение всех признаков к одному масштабу с помощью StandardScaler, предоставленного в библиотеке scikit-learn.
        """
    )
    st.title("Преобразование категориальных переменных")

    st.markdown("""
        ### Преобразование категориальных переменных
        OrdinalEncoder является одним из методов преобразования категориальных признаков в числовые значения в машинном обучении. Этот метод присваивает каждой категории уникальный целочисленный код, который соответствует ее позиции в заданном порядке или алфавитном порядке.
        
        Процесс работы OrdinalEncoder:
        * Определение уникальных категорий: Прежде всего, определяются уникальные категории в исходном категориальном признаке.
        * Упорядочивание категорий: Категории могут быть упорядочены вручную или в соответствии с их алфавитным порядком.
        * Присвоение кодов: Каждой категории присваивается уникальный целочисленный код, начиная с 0 и увеличиваясь на единицу для каждой следующей категории.
        * Преобразование категориальных значений в числовые: Каждое категориальное значение заменяется соответствующим ему целочисленным кодом.
        
        Преимущества OrdinalEncoder:
        * Простота: Метод легко реализуется и применяется к категориальным признакам.
        * Сохранение информации о порядке: При наличии упорядоченности категорий метод сохраняет эту информацию, что может быть важно для некоторых моделей машинного обучения.

        #### Общие формулы и принципы

        В процессе предобработки данных могут использоваться следующие операции и принципы:

        - **Нормализация:** приведение всех числовых переменных к единому масштабу, чтобы улучшить сходимость алгоритма. Обычно используется Min-Max scaling или Z-score стандартизация.
        - **One-hot Encoding:** преобразование категориальных переменных в бинарные векторы; применяется, когда порядок категорий не имеет значения.
        - **Отбор признаков:** удаление нерелевантных или малоинформативных признаков для упрощения модели и улучшения её обобщающей способности.

        Эти методы помогают подготовить данные для эффективного обучения моделей машинного обучения.
        """)

    categorical_features = df.select_dtypes(include='category').columns.tolist()
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Первоначально мы должны решить, какие переменные мы будем использовать. Мы не делаем различий между школами. Поэтому вариативная школа будет исключена.
    df = df.drop(labels=["school"], axis=1)

    categories_ordenc = {
        "sex": [["F", "M"]],
        "address": [["R", "U"]],
        "famsize": [["LE3", "GT3"]],
        "Pstatus": [['A', 'T']],
        "schoolsup": [["no", "yes"]],
        "famsup": [["no", "yes"]],
        "paid": [["no", "yes"]],
        "activities": [["no", "yes"]],
        "nursery": [["no", "yes"]],
        "higher": [["no", "yes"]],
        "internet": [["no", "yes"]],
        "romantic": [["no", "yes"]],
        "Fjob": [["other", "teacher", "health", "services", 'at_home']],
        "Mjob": [["other", "teacher", "health", "services", 'at_home']],
        "reason": [["other", "home", "reputation", "course"]],
        "guardian": [["other", "mother", "father"]]
    }
    df_encoded = df.copy(deep=True)
    df_encoded = df_encoded.reindex(sorted(df_encoded.columns), axis=1)

    def encode_features(df, categories_ordenc):
        encoders = {}
        for column, categories in categories_ordenc.items():
            encoder = OrdinalEncoder(categories=categories, dtype=np.int64)
            df[column] = encoder.fit_transform(df[column].values.reshape(-1, 1))
            encoders[column] = encoder
        return df, encoders

    df_encoded, encoders = encode_features(df_encoded, categories_ordenc)

    st.markdown(
        """
        ## Кодирование признаков (Feature Encoding)
        """
    )
    feature_encoding_tab1, feature_encoding_tab2 = st.tabs([
        "Данные до Feature Encoding",
        "Данные после Feature Encoding"
    ])
    with feature_encoding_tab1:
        st.dataframe(df.head())
    with feature_encoding_tab2:
        st.dataframe(df_encoded.head())

    st.subheader('Разделение данных')
    st.markdown("""
        #### Подготовка данных
        
        1. Выбор переменных: Основываясь на предварительном анализе, можно выбрать следующие переменные, которые, как предполагается, могут влиять на итоговую оценку студента:
           - Школьная поддержка (`schoolsup`)
           - Семейная поддержка (`famsup`)
           - Платные дополнительные занятия (`paid`)
           - Внеклассные мероприятия (`activities`)
           - Посещение детского сада (`nursery`)
           - Желание получить высшее образование (`higher`)
           - Наличие интернета дома (`internet`)
           - Романтические отношения (`romantic`)
        
        2. Разделение данных: Данные разделяются на обучающую и тестовую выборки в соотношении 80% на 20% соответственно, чтобы обеспечить достаточный объем данных для обучения модели и адекватно проверить её на новых данных.
    """)
    X = df_encoded.drop('G3', axis=1).copy(deep=True)
    Y = df_encoded['G3']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    st.write("Размер тренировочных данных:", X_train.shape, y_train.shape)
    st.write("Размер тестовых данных:", X_test.shape, y_test.shape)

    tab1, tab2 = st.tabs(["Тренировочные данные", "Тестовые данные"])

    with tab1:
        st.subheader("Тренировочные данные")
        st.markdown("""
          **Описание:** Тренировочные данные используются для подгонки модели и оценки её параметров.
          Эти данные получены путем исключения из исходного датасета столбцов с целевой переменной 'G3'.

          **Данные тренировочного набора (X_train)**.
          Обучающий набор данных содержит информацию о признаках, используемых для обучения модели.
          """)
        st.dataframe(X_train.head(15), use_container_width=True)
        st.markdown("""
          **Целевая переменная (y_train)**.
          Целевая переменная содержит значения цены, которые модель должна научиться прогнозировать.
          В качестве целевой переменной для тренировочного набора используются исключительно значения столбца 'G3'.
          """)
        st.dataframe(pd.DataFrame(y_train.head(15)).T)

    st.header('Выбор типа математической модели прогноза')
    st.markdown(r"""
          ## Множественная линейная регрессия
          Множественная линейная регрессия позволяет оценивать зависимость одной зависимой переменной от двух или более независимых переменных. Это делает её отличным инструментом для анализа и прогнозирования, где несколько факторов влияют на интересующий результат.

          ### Формула множественной линейной регрессии

          Формула множественной линейной регрессии выглядит следующим образом:

          $$
          y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \ldots + \beta_nx_n + \varepsilon
          $$

          Где:
          - $ y $: Зависимая переменная (предсказываемая переменная). Это та переменная, значение которой мы пытаемся предсказать на основе независимых переменных.
          - $ \beta_0 $: Константа (интерцепт), представляющая собой значение $ y $, когда все независимые переменные равны нулю.
          - $ \beta_1, \beta_2, \ldots, \beta_n $: Коэффициенты независимых переменных, которые измеряют изменение зависимой переменной при изменении соответствующих независимых переменных.
          - $ x_1, x_2, \ldots, x_n $: Независимые переменные, используемые для предсказания значения $ y $.
          - $ \varepsilon $: Ошибка модели, описывающая разницу между наблюдаемыми значениями и значениями, предсказанными моделью.

          ### Описание параметров

          - **Зависимая переменная ( $ y $ )**: Это переменная, которую вы пытаетесь предсказать. Например, количество прокатов велосипедов может быть зависимой переменной, которую мы хотим предсказать на основе погоды, времени года и других условий.

          - **Константа ( $ \beta_0 $ )**: Это значение зависимой переменной, когда все входные (независимые) переменные равны нулю. В реальности это значение может не иметь физического смысла, особенно если ноль не является допустимым значением для независимых переменных.

          - **Коэффициенты ( $ \beta_1, \beta_2, \ldots, \beta_n $ )**: Эти значения указывают, насколько изменится зависимая переменная при изменении соответствующей независимой переменной на одну единицу, при условии что все остальные переменные остаются неизменными. Они являются ключевыми в понимании влияния каждой независимой переменной на зависимую переменную.

          - **Независимые переменные ( $ x_1, x_2, \ldots, x_n $ )**: Это переменные или факторы, которые предположительно влияют на зависимую переменную. В контексте вашего приложения это могут быть погода, день недели, сезон и другие.

          - **Ошибка модели ( $ \varepsilon $ )**: Ошибка модели показывает, насколько далеко наши предсказания от фактических значений. Это может быть вызвано неполным объяснением всех влияющих факторов или случайными изменениями, которые невозможно предсказать с помощью модели.
      """)

    linear_model, y_pred_train_linear = prepare_data(sm.add_constant(X_train), y_train)
    y_pred_train_linear = linear_model.predict(sm.add_constant(X_train))

    # Извлечение данных о параметрах модели
    st.subheader('Результаты модели')
    st.text(str(linear_model.summary())[:950])
    st.subheader('Коэффициенты модели')
    summary_data = linear_model.summary().tables[1]
    info = pd.DataFrame(summary_data.data[1:], columns=summary_data.data[0])
    st.dataframe(info, use_container_width=True, hide_index=True)

    st.markdown("""
          ## Анализ точности построения модели
          ## Прогноз на зависимые данные
      """)
    rmse, mae, r_squared = calculate_metrics(y_train, y_pred_train_linear)
    st.info(f"""
          ### Результаты прогноза на зависимые данные
          - **Среднеквадратическая ошибка (RMSE):** {rmse:.2f}%
          - **Средняя абсолютная ошибка (MAE):** {mae:.2f}
          - **Коэффициент детерминации (R²):** {r_squared:.3f}
      """)
    plot_regression_results(y_train, y_pred_train_linear)

    y_pred_test_linear = linear_model.predict(sm.add_constant(X_test))
    mape, mse, r_squared = calculate_metrics(y_test, y_pred_test_linear.values)
    st.info(f"""
          ### Результаты прогноза на нeзависимые данные
         - **Среднеквадратическая ошибка (RMSE):** {rmse:.4f}
         - **Средняя абсолютная ошибка (MAE):** {mae:.4f}
         - **Коэффициент детерминации (R²):** {r_squared:.3f}
     """)
    plot_regression_results(y_test, y_pred_test_linear)

    st.markdown("""
        ## Анализ результатов модели линейной регрессии
        ### Общий обзор результатов
        Модель линейной регрессии была обучена для прогнозирования итоговых оценок учеников (G3), исходя из различных доступных данных об учащихся. Результаты анализа показывают следующее:
        
        * R-squared (R²) для модели на тренировочной выборке составляет 0.861, что указывает на то, что модель объясняет 86.1% вариативности зависимой переменной с помощью выбранных предикторов. Это свидетельствует о высокой объяснительной способности модели.
        * Adjusted R-squared на уровне 0.845 подтверждает, что после коррекции на количество используемых переменных и размер выборки, модель все еще остается эффективной.
        * F-statistic и соответствующее P-value (Prob (F-statistic)) подтверждают статистическую значимость всей модели в целом.
        
        Важные предикторы
        * Среди значимых предикторов наибольшее влияние на итоговую оценку оказали:
        * G2 (коэффициент = 0.9567, P < 0.001): Оценки за второй период являются мощным предиктором итоговой оценки, что логично, учитывая последовательность учебного процесса.
        * G1 (коэффициент = 0.2022, P = 0.002): Оценки за первый период также значимо влияют на итоговую оценку.
        * absences (коэффициент = 0.0505, P < 0.001): Число пропусков занятий также оказывает влияние, хотя и менее значительное по сравнению с оценками за учебные периоды.
        
        Прогнозирование и оценка качества
        * Тестирование модели на тестовой выборке показало следующие результаты:
            * RMSE: 2.295
            * MAE: 1.554
            * R²: 0.743
        Эти метрики свидетельствуют о том, что модель достаточно хорошо справляется с прогнозированием итоговых оценок, хотя точность на тестовой выборке немного ниже, чем на тренировочной.
        
        ### Визуализация результатов
        * Сравнение истинных и предсказанных значений: График показывает, что предсказания модели в основном соответствуют истинным значениям, хотя наблюдаются некоторые отклонения, особенно для более высоких значений.
        * Распределение ошибок модели: График ошибок показывает, что большинство ошибок сосредоточено около нуля, что характеризует хорошее качество предсказаний модели, но присутствует некоторая асимметрия, указывающая на систематические ошибки в некоторых случаях.
        
        ### Выводы
        Модель линейной регрессии эффективно использовалась для анализа влияния различных факторов на итоговые оценки студентов. Она показала хорошие результаты как в теоретическом анализе, так и на практике, особенно в части предсказания. Полученные данные могут быть использованы для дальнейшего улучшения учебных программ и подходов к образованию.
     """)

    with st.form("prediction_form"):
        st.subheader('Введите параметры для прогноза')

        col1, col2 = st.columns(2)

        with col1:
            sex = option_menu(
                "Пол",
                options=encoders['sex'].categories[0],
                menu_icon="gender-ambiguous",
                icons=["gender-female", "gender-male"],
                default_index=1
            )
            address = option_menu(
                "Тип адреса",
                options=encoders['address'].categories[0],
                menu_icon="geo-alt",
                icons=["house", "building"],
                default_index=1
            )
            famsize = option_menu(
                "Размер семьи",
                options=encoders['famsize'].categories[0],
                menu_icon="people",
                icons=["person", "people"],
                default_index=1
            )
            Pstatus = option_menu(
                "Статус совместного проживания родителей",
                options=encoders['Pstatus'].categories[0],
                menu_icon="house",
                icons=["house-door", "house-door-fill"],
                default_index=1
            )
            schoolsup = option_menu(
                "Дополнительная образовательная поддержка",
                options=encoders['schoolsup'].categories[0],
                menu_icon="book",
                icons=["book", "book-fill"],
                default_index=1
            )
            famsup = option_menu(
                "Семейная образовательная поддержка",
                options=encoders['famsup'].categories[0],
                menu_icon="people",
                icons=["person", "people"],
                default_index=1
            )

        with col2:
            paid = option_menu(
                "Дополнительные платные занятия",
                options=encoders['paid'].categories[0],
                menu_icon="cash",
                icons=["cash", "cash-coin"],
                default_index=1
            )
            activities = option_menu(
                "Внеучебные активности",
                options=encoders['activities'].categories[0],
                menu_icon="activity",
                icons=["activity", "book"],
                default_index=1
            )
            nursery = option_menu(
                "Посещал ли детский сад",
                options=encoders['nursery'].categories[0],
                menu_icon="house-check",
                icons=["circle", "check2-circle"],
                default_index=1
            )
            higher = option_menu(
                "Хочет ли получить высшее образование",
                options=encoders['higher'].categories[0],
                menu_icon="mortarboard",
                icons=["mortarboard", "mortarboard-fill"],
                default_index=1
            )
            internet = option_menu(
                "Наличие интернета дома",
                options=encoders['internet'].categories[0],
                menu_icon="wifi",
                icons=["wifi-off", "wifi"],
                default_index=1
            )
            romantic = option_menu(
                "Наличие романтических отношений",
                options=encoders['romantic'].categories[0],
                menu_icon="heart",
                icons=["heart", "heart-fill"],
                default_index=0
            )

        col3, col4 = st.columns(2)

        with col3:
            guardian = option_menu(
                "Опекун",
                options=encoders['guardian'].categories[0],
                menu_icon="person",
                icons=["person", "person-badge", "person-check"],
                default_index=0
            )
            age = st.number_input("Возраст", min_value=df['age'].min(), max_value=100, value=18)
            Mjob = st.selectbox("Работа матери", options=encoders['Mjob'].categories[0])
            Fjob = st.selectbox("Работа отца", options=encoders['Fjob'].categories[0])
            traveltime = st.slider("Время в пути до школы", min_value=int(df['traveltime'].min()), max_value=int(df['traveltime'].max()), value=int(df['traveltime'].mean()))
            studytime = st.slider("Время на учёбу помимо школы в неделю", min_value=int(df['studytime'].min()), max_value=int(df['studytime'].max()), value=int(df['studytime'].mean()))
            failures = st.slider("Количество внеучебных неудач", min_value=int(df['failures'].min()), max_value=int(df['failures'].max()), value=int(df['failures'].mean()))
            absences = st.number_input("Количество пропущенных занятий", min_value=0, max_value=365, value=int(df['absences'].mean()))
            g1 = st.number_input("Оценки за первый период", min_value=0, max_value=int(df['G1'].max() * 2), value=19)

        with col4:
            reason = st.selectbox("Причина выбора школы", options=encoders['reason'].categories[0])
            famrel = st.slider("Качество семейных отношений", min_value=int(df['famrel'].min()), max_value=int(df['famrel'].max()), value=int(df['famrel'].mean()))
            freetime = st.slider("Свободное время после школы", min_value=int(df['freetime'].min()), max_value=int(df['freetime'].max()), value=3)
            Medu = st.slider("Образование матери", min_value=0, max_value=4, value=4)
            Fedu = st.slider("Образование отца", min_value=0, max_value=4, value=4)
            goout = st.slider("Проведение времени с друзьями", min_value=int(df['goout'].min()), max_value=int(df['goout'].max()), value=int(df['goout'].mean()))
            Dalc = st.slider("Потребление алкоголя в будни", min_value=int(df['Dalc'].min()), max_value=int(df['Dalc'].max()), value=int(df['Dalc'].mean()))
            Walc = st.slider("Потребление алкоголя в выходные", min_value=int(df['Walc'].min()), max_value=int(df['Walc'].max()), value=int(df['Walc'].mean()))
            health = st.slider("Текущее состояние здоровья", min_value=int(df['health'].min()), max_value=int(df['health'].max()), value=int(df['health'].mean()))
            g2 = st.number_input("Оценки за второй период", min_value=0, max_value=int(df['G2'].max() * 2), value=19)

        if st.form_submit_button("Прогнозировать", type='primary', use_container_width=True):
            input_df = {
                'const': 1,
                "G1": g1,
                "G2": g2,
                "age": age,
                "Fedu": Fedu,
                "Medu": Medu,
                "Mjob": Mjob,
                "Fjob": Fjob,
                "reason": reason,
                "traveltime": traveltime,
                "studytime": studytime,
                "failures": failures,
                "famrel": famrel,
                "freetime": freetime,
                "goout": goout,
                "Dalc": Dalc,
                "Walc": Walc,
                "health": health,
                "absences": absences,
                "sex": sex,
                "address": address,
                "famsize": famsize,
                "Pstatus": Pstatus,
                "schoolsup": schoolsup,
                "famsup": famsup,
                "guardian": guardian,
                "paid": paid,
                "activities": activities,
                "nursery": nursery,
                "higher": higher,
                "internet": internet,
                "romantic": romantic,
            }
            for column in categories_ordenc.keys():
                input_df[column] = encoders[column].transform([[input_df[column]]])[0][0]

            input_df = pd.DataFrame([input_df])
            input_df = input_df.reindex(sorted(sm.add_constant(X_test).columns), axis=1)

            prediction = max(linear_model.predict(input_df)[0], 0)

            st.success("Прогноз успешно выполнен!")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prediction),
                number={"valueformat": ".0f"},
                gauge={
                    "axis": {"range": [0, 20]},
                    "bar": {"color": 'darkgreen'},
                },
                title={"text": "<b>Прогнозируемая оценка за финальный экзамен</b>"}
            ))
            fig.update_layout(paper_bgcolor="#f0f2f6", font={'color': "darkblue", 'size': 30, 'family': "Arial"})
            st.plotly_chart(fig, use_container_width=True)
