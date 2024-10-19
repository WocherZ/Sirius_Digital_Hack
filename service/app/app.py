import requests
import streamlit as st
import pandas as pd
import openpyxl
import time

from PIL import Image

img = Image.open(r"..\assets\logo.png")

st.image(img, width=400)

page = st.sidebar.selectbox("Выберите страницу", 
                            ["Анализ ответов", "Анализ сотрудников"])

if page == "Анализ ответов":
    st.header("Анализ ответов")
    leaving_reasons = st.text_input("Какая у вас причина уйти из компании?", 
                                    placeholder = "Увольняюсь по причине ...")
    staying_possibility = st.text_input("Рассматриваете ли вы возможность остаться?", 
                                        placeholder = "Да/Нет, потому что ...")
    returning_possibility = st.text_input("Рассматриваете ли вы возможность вернуться?", 
                                          placeholder = "Да/Нет, потому что ...")
    recommendation_possibility = status = st.radio("Рекомендовали ли бы вы нашу компанию остальным?", 
                                                   ("Да","Нет"))
    
    if (recommendation_possibility == "Да"):
        reason_to_recomend = st.text_input("Что вас больше привлекает в нашей компании?",
                                           placeholder = "Печеньки и ...")
        
    elif(recommendation_possibility == "Нет"):
        reason_to_recomend = st.text_input("Что вас больше отталкивает в нашей компании?",
                                           placeholder = "Далеко до офиса ...")
    
    # Кнопка для запуска вычислений
    if st.button("Проанализировать"):
        response = requests.post("http://localhost:8000/single_clustering/", 
                                             data=leaving_reasons)
        st.write("Причина ухода:", response.json()["clusters"])

        payload = {"data": staying_possibility,
                    "question": "Рассматриваете ли вы возможность остаться?"
                    }
        response = requests.post("http://localhost:8000/single_sentiment_analysis/", 
                                    data=payload)
        st.write("Возможность остаться", response.json()["analysis"])
        payload = {"data": returning_possibility,
                    "question": "Рассматриваете ли вы возможность вернуться?"
                    }
        response = requests.post("http://localhost:8000/single_sentiment_analysis/", 
                                    data=payload)
        st.write("Возможность возвращения", response.json()["analysis"])

        st.write("Рекомендации другим:", recommendation_possibility)

        payload = {"data": recommendation_possibility,
                    "question": 'Рекомендовали ли бы вы нашу компанию остальным?'
                    }
        response = requests.post("http://localhost:8000/single_sentiment_analysis/", 
                                    data=payload)
        st.write("Причина рекомендаций:", response.json()["analysis"])


elif page == "Анализ сотрудников":
    st.subheader("Загрузка и анализ XLSX файла")

    @st.cache_data
    def load_and_process_data(uploaded_file):
        total_size = uploaded_file.seek(0, 2)
        uploaded_file.seek(0)

        chunk_size = 1024 * 1024
        chunks = []
        bytes_read = 0

        with st.spinner("Загрузка файла..."):
            progress_bar = st.progress(0)
            for chunk in iter(lambda: uploaded_file.read(chunk_size), b''):
                chunks.append(chunk)
                bytes_read += len(chunk)
                progress = bytes_read / total_size
                progress_bar.progress(progress)

        content = b''.join(chunks)
        df = pd.read_excel(content)

    uploaded_file = st.file_uploader("Загрузите XLSX файл", type=["xlsx"])

    analysis_type = st.sidebar.selectbox("Выберите тип анализа",
                                         ["Анализ одного сотрудника", "Анализ результатов опроса"])

    if uploaded_file is not None and analysis_type:
        try:
            df = load_and_process_data(uploaded_file)
            empoyees_data = pd.read_excel(uploaded_file, sheet_name='ответы сотрудников')
            hr_data = pd.read_excel(uploaded_file, sheet_name='ответы hr ')
            
            data_columns = {
                'Причина увольнения': {
                    'Сотрудники': "Комментарий к вопросу  1. Какие причины (факторы) сформировали ваше решение уйти из компании (выберите не более 3-х).",
                },
                'Возможность остаться': {
                    'Сотрудники': "Комментарий к вопросу 2 Рассматриваете ли вы возможность остаться в компании/перевестись внутри отрасли?",
                    'HR': "Комментарий к вопросу 2 Рассматриваете ли вы возможность остаться в компании/перевестись внутри отрасли? Были ли попытки руководителя сохранить вас в компании?"
                },
                'Возможность возвращения': {
                    'Сотрудники': "Комментарий к вопросу 3 Рассматриваете ли вы возможность возвращения в компанию?",
                    'HR': "Комментарий к вопросу 3 Рассматриваете ли вы возможность возвращения в компанию, если нет, то почему?"
                },
                'Рекомендация': {
                    'Сотрудники': "Комментарий к вопросу 4 Готовы ли вы рекомендовать компанию как работодателя?",
                    'HR': "Комментарий к вопросу 4 Готовы ли вы рекомендовать компанию как работодателя, если нет, то почему?"
                }
            }

            question = st.selectbox("Вопросы: ",
                                    ['Возможность остаться',
                                     'Возможность возвращения',
                                     'Рекомендация',
                                     'Причина увольнения'])

            colum_employer = empoyees_data[data_columns[question]['Сотрудники']]
            colum_hr = None
            
            if question != 'Причина увольнения':
                colum_hr = hr_data[data_columns[question]['HR']]

            merged_df = pd.DataFrame({
                "Сотрудники": colum_employer,
                "HR": colum_hr
            })
            merged_df.dropna(subset=['Сотрудники', 'HR'], inplace=True)


            def clean_df(df, column_name, min_length=2):
                df = df[df[column_name].str.len() >= min_length]
                return df


            merged_df = clean_df(merged_df, 'Сотрудники')
            merged_df = clean_df(merged_df, 'HR')
            merged_df_json = merged_df.to_json(orient='split')
            st.success("Файл успешно загружен!")

            # First question
            # def recommendation_analysis(merged_df):
            #     st.title("рекомендация")

            # Second question
            # def returning_analysis(merged_df):
            #     st.title("возврат")

            # Third question
            # def leaving_analysis(merged_df):
            #     st.title("уволен")

            # Fourth question
            # def staying_analysis(merged_df):
            #     st.title("остаться")

            if (question == 'Причина увольнения'):
                if analysis_type == 'Анализ одного сотрудника':
                    response = requests.post("http://localhost:8000/single_clustering/", 
                                             json=merged_df_json)
                else:
                    response = requests.post("http://localhost:8000/multi_clustering/", 
                                             json=merged_df_json)
                st.write(response.json()["clusters"])

            if (question in ['Возможность остаться', 
                             'Возможность возвращения', 
                             'Рекомендация']):
                if analysis_type == 'Анализ одного сотрудника':
                    payload = {"data": merged_df_json,
                               "question": question
                               }
                    response = requests.post("http://localhost:8000/single_sentiment_analysis/", 
                                             data=payload)
                    st.write(response.json()["analysis"])
                else:
                    for row in merged_df.iterrows():
                        payload = {"data": row,
                                   "question": question
                                   }
                        response = requests.post("http://localhost:8000/multi_sentiment_analysis/", 
                                                 json=payload)

                    st.write(response.json()["analysis"])
                    st.pyplot(response.json()["graph"])

        except Exception as e:
            st.error(f"Произошла ошибка: {e}")
