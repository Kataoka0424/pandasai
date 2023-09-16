import pandas as pd
import streamlit as st
import matplotlib
import japanize_matplotlib
from PIL import Image
import os
import time

from pandasai.llm import OpenAI
from pandasai import SmartDataframe

from bs4 import BeautifulSoup

#pip install pandas streamlit pandasai beautifulsoup4 matplotlib japanize-matplotlib

st.set_page_config(page_title='pandasai', layout='centered')
st.title('pandasai')

OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']

#encoding設定
selected_encoding = st.selectbox(
    'encodingを選択', 
    ['utf-8', 'shift_jis', 'cp932'],
    key='encoding')

uploaded_file = st.file_uploader('files of csv', type='csv')

df = pd.DataFrame()
if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding=selected_encoding)

    st.write(f'row: {len(df)}')
    with st.expander('df', expanded=False):
        st.write(df)
    
# LLMのインスタンス化
llm = OpenAI(api_token=OPENAI_API_KEY)
# プロンプト入力
prompt = st.text_area('enter your prompt:')


if prompt:
    st.markdown('##### response')

    file_path = './exports/charts/temp_chart.png'

    if os.path.isfile(file_path): #ファイルの存在確認
        #グラフ画像のファイル削除
        os.remove(file_path)

    #クラス　自然言語受け付け可
    sdf = SmartDataframe(df, config={"llm": llm})
    response_data = sdf.chat(prompt)

    message = st.chat_message("assistant")
    message.write(response_data)

    #ファイルが作成されるまでの時間を作る
    time.sleep(15)

    if os.path.isfile(file_path): #ファイルの存在確認
        
        image = Image.open(file_path)
        st.image(image, caption='by PandasAI')

    with st.expander('code', expanded=False):
        st.code(sdf.last_code_generated)
   
else:
    st.warning('plase enter a prompt.')

# st.write(sdf[sdf['商品名'] == 'PUREDRIVE']['金額'].sum())


