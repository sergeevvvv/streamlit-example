
import streamlit as st

st.header('Descartes: перевод математических формул, записанных словами в LaTeX')
st.markdown('Данная сборка модели обучена на арифметику, включая корни, логарифмы, степени, а также пределы и интегралы из математического анализа, функции суммы и произведения. Модель обучена на примерах минимальной вложенности на русском языке на латинском и греческих алфавитах не включая цифры.')

text = st.text_input('Введите формулу словами', value="два икс плюс три")

def translate(string):
    return string

if st.button('Показать формулу'):
    ans = translate(text)
    st.text('Формула в "сыром" LaTeX:')
    st.text(ans)
    
    st.text('Зарендеренная формула:')
    st.latex(ans)

st.text('Баги и идеи (github): https://github.com/sergeevvvv/MathMT/issues')
st.text('Вячеслав Сергеев. sergeev46v@gmail.com')
