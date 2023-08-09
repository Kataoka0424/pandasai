import pandas as pd
import streamlit as st
import os
import datetime
import openpyxl
import matplotlib
import japanize_matplotlib

from pandasai import PandasAI 
from pandasai.llm.openai import OpenAI

st.markdown('### pandasai')

# os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']

llm = OpenAI(api_token=OPENAI_API_KEY)
pandas_ai = PandasAI(llm)

# matplotlib.use('TkAgg')
# matplotlib.use('AGG')

@st.cache_data(ttl=datetime.timedelta(minutes=20))
def make_data(df):
    # df = pd.read_excel(
    # file, sheet_name='受注委託移動在庫生産照会', \
    #     usecols=[1, 3, 6, 8, 10, 14, 15, 16, 21, 28, 30, 31, 42, 46, 50, 51]) #index　ナンバー不要　index_col=0

    # *** 出荷月、受注月列の追加***
    df['伝票番号2'] = df['伝票番号'].apply(lambda x: str(x)[:-3])
    df['出荷月'] = df['出荷日'].dt.month
    df['受注月'] = df['受注日'].dt.month
    df['商品コード2'] = df['商　品　名'].map(lambda x: x.split()[0]) #品番
    df['商品コード3'] = df['商　品　名'].map(lambda x: str(x)[0:2]) #頭品番
    df['張地'] = df['商　品　名'].map(lambda x: x.split()[2] if len(x.split()) >= 4 else '') 

    # ***INT型への変更***
    df[['数量', '単価', '金額', '出荷倉庫', '原価金額', '出荷月', '受注月']] = df[['数量', '単価', '金額', '出荷倉庫', '原価金額', '出荷月', '受注月']].fillna(0).astype('int64')
    #fillna　０で空欄を埋める

    #LD分類列の追加
    livings = ['クッション', 'リビングチェア', 'リビングテーブル']
    dinings = ['ダイニングテーブル', 'ダイニングチェア', 'ベンチ']
    others = ['キャビネット類', 'その他テーブル', '雑品・特注品', 'その他椅子', 'デスク', '小物・その他']
    # 新しい列を追加してリスト名を入力
    def categorize_item(item):
        if item in livings:
            return 'livings'
        elif item in dinings:
            return 'dinings'
        elif item in others:
            return 'others'
        else:
            return 'unknown'
    
    df['LD分類'] = df['商品分類名2'].apply(categorize_item)

    df = df.drop(['伝票番号', '商　品　名'], axis=1)

    df = df[['伝票番号2', '注文No', '得意先名', '受注日', '出荷日', 'シリーズ名', '商品コード2', '商品コード3', \
             '塗色CD', '張地', '数量', '単価', '金額', '商品分類名2', 'LD分類', '出荷倉庫', '原価単価', '原価金額',\
            '営業担当者名', '受注月', '出荷月']]

    return df

uploaded_file_now = st.sidebar.file_uploader('今期ファイル', type='xlsx', key='now')
uploaded_file_last = st.sidebar.file_uploader('前期ファイル', type='xlsx', key='last')
uploaded_file = pd.DataFrame()

# uploaded_file は読み込んだデータフレーム
if uploaded_file_now and uploaded_file_last:
    df_now = pd.read_excel(
    uploaded_file_now, sheet_name='受注委託移動在庫生産照会', \
        usecols=[1, 3, 6, 8, 10, 14, 15, 16, 21, 28, 30, 31, 42, 46, 50, 51])
    df_last = pd.read_excel(
    uploaded_file_last, sheet_name='受注委託移動在庫生産照会', \
        usecols=[1, 3, 6, 8, 10, 14, 15, 16, 21, 28, 30, 31, 42, 46, 50, 51])
    uploaded_file = pd.concat([df_last, df_now], axis=0)

elif uploaded_file_now:
    uploaded_file =  pd.read_excel(
    uploaded_file_now, sheet_name='受注委託移動在庫生産照会', \
        usecols=[1, 3, 6, 8, 10, 14, 15, 16, 21, 28, 30, 31, 42, 46, 50, 51])
    
elif uploaded_file_last:
    uploaded_file = pd.read_excel(
    uploaded_file_last, sheet_name='受注委託移動在庫生産照会', \
        usecols=[1, 3, 6, 8, 10, 14, 15, 16, 21, 28, 30, 31, 42, 46, 50, 51])

#処理の開始
if not uploaded_file.empty:
    df = make_data(uploaded_file)

    st.write('columns')
    st.write(df.head(1))

    st.write('exsamples of prompts')

    prompt_dict = {
        'ひな形なし': '下記の条件を考慮して計算の上表示してください。#表示内容は数字#条件\
        - 受注日の2021年10月1日から2022年9月30日までを79期とします。- 受注日の2022年10月1日から2023年9月30日までを80期とします。',
        '表: 月毎': '下記の条件を考慮して表示してください。#表示内容は表形式#条件- 得意先名は「オツタカ」を含んだ名称\
        - 受注日の2021年10月1日から2022年9月30日までを79期とします。- 受注日の2022年10月1日から2023年9月30日までを80期とします。\
        - 月毎の金額の合計',
        'グラフ: 月毎': '下記の条件を考慮して表示してください。#表示内容は折れ線グラフ\
        #streamlit用のコードでグラフを作成してください#st.pyplot(fig)で表示してください#条件- \
        - 得意先名は「オツタカ」を含む\
        - 受注日の2021年10月1日から2022年9月30日までを79期とします。- 受注日の2022年10月1日から2023年9月30日までを80期とします。\
        - 月毎の金額の合計',
        '棒グラフ: 得意先別の金額合計': '下記の条件を考慮して表示してください。#表示内容は棒グラフ\
        #streamlit用のコードでグラフを作成してください#st.pyplot(fig)で表示してください#条件\
        - 得意先名は「オツタカ」を含む\
        - 受注日の2021年10月1日から2022年9月30日までを79期とします。- 受注日の2022年10月1日から2023年9月30日までを80期とします。\
        - ７月の金額の合計'
          }
    
    slct_prompt = st.selectbox(
        'select a example', 
        prompt_dict.keys())
    
    prompt = st.text_area(
        'enter your prpmpt:', 
        value=prompt_dict[slct_prompt]
        )

    if st.button('genarate'):
        if prompt:
            st.write(pandas_ai.run(df, prompt=prompt))
        else:
            st.warning('plase enter a prompt.')

