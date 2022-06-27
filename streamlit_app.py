import streamlit as st

st.header('Descartes: перевод математических формул, записанных словами в LaTeX')
st.text('Данная сборка модели обучена на арифметику, включая корни, логарифмы, степени, а также пределы и интегралы из математического анализа, функции суммы и произведения. Модель обучена на примерах минимальной вложенности на русском языке на латинском и греческих алфавитах не включая цифры.')

text = st.text_input(label, value="два икс плюс три", max_chars=None, key=None, type="default", help=None, autocomplete=None, on_change=None, args=None, kwargs=None, *, placeholder=None, disabled=False)

def translate(string):
    return '$x^2$'

if st.button('Показать формулу'):
    ans = translate(text)
    st.text('Формула:')
    st.text(ans)
    
    st.text('Ее символьный вид:')
    st.latex(r'$'ans'$')

st.text('Баги и идеи на github:')
