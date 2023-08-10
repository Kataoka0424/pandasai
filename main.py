import pandas as pd
import streamlit as st
import os
import datetime
import openpyxl
import matplotlib
import japanize_matplotlib

from pandasai import PandasAI 
from pandasai.llm.openai import OpenAI

st.set_page_config(page_title='pandasai', layout='centered')
st.markdown('### pandasai')

# os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']

llm = OpenAI(api_token=OPENAI_API_KEY)
pandas_ai = PandasAI(llm)

# matplotlib.use('TkAgg')
# matplotlib.use('AGG')

cols = [
    '伝票番号2', '注文No', '得意先名', '受注日', '出荷日', 'シリーズ名', '商品コード2', '商品コード3',\
    '塗色CD', '張地', '数量', '単価', '金額', '商品分類名2', 'LD分類', '出荷倉庫', '原価単価', '原価金額',\
    '営業担当者名', '受注月', '出荷月'
]


@st.cache_data(ttl=datetime.timedelta(minutes=20))
def make_data(df):

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

    df = df[cols]

    return df

#プロンプト
prompt_dict = {
    'ひな形なし': '下記の条件を考慮して計算の上表示してください。#表示内容は数字#条件\
    - 受注日の2021年10月1日から2022年9月30日までを79期とします。- 受注日の2022年10月1日から2023年9月30日までを80期とします。',
    '表: 月毎': '下記の条件を考慮して表示してください。#表示内容は表形式#条件- 得意先名は「オツタカ」を含んだ名称\
    - 受注日の2021年10月1日から2022年9月30日までを79期とします。- 受注日の2022年10月1日から2023年9月30日までを80期とします。\
    - 月毎の金額の合計',
    'グラフ: 月毎': '下記の条件を考慮して表示してください。#表示内容は折れ線グラフ\
    #streamlit用のコードでグラフを作成#st.pyplot(fig)で表示#条件 \
    - 得意先名は「オツタカ」を含む\
    - 受注日の2021年10月1日から2022年9月30日までを79期とします。- 受注日の2022年10月1日から2023年9月30日までを80期とします。\
    - 月毎の金額の合計',
    '棒グラフ: 得意先別の金額合計': '下記の条件を考慮して表示してください。#表示内容は棒グラフ\
    #streamlit用のコードでグラフを作成#st.pyplot(fig)で表示#条件\
    - 得意先名は「オツタカ」を含む\
    - 受注日の2021年10月1日から2022年9月30日までを79期とします。- 受注日の2022年10月1日から2023年9月30日までを80期とします。\
    - ７月の金額の合計'
    }

col1, col2 = st.columns([1, 1])
with col1:
    with st.expander('columns', expanded=False):
        df_cols = pd.DataFrame(cols)
        st.table(df_cols)

with col2:
    slct_prompt = st.selectbox(
        'select an example of prompt', 
        prompt_dict.keys(),
        key='selectbox'
    )


prompt = st.text_area(
    'enter your prompt:', 
    value=prompt_dict[slct_prompt]
)


uploaded_files = st.file_uploader('files of xlsx', type='xlsx', key='uploaded', accept_multiple_files=True)

# uploaded_file は読み込んだデータフレーム
df = pd.DataFrame()
if st.button('start chat', key='set_files'):
    filenames = [file.name for file in uploaded_files]

    for filename in filenames:
        #uploaded_files内でのindexを取得
        filename_index = filenames.index(filename)
        
        df_add = pd.read_excel(
            uploaded_files[filename_index], sheet_name='受注委託移動在庫生産照会', \
            usecols=[1, 3, 6, 8, 10, 14, 15, 16, 21, 28, 30, 31, 42, 46, 50, 51])
        
        df = pd.concat([df, df_add], axis=0, join='outer')


    #処理の開始

    df2 = make_data(df)

    with st.expander('dataframe', expanded=False):
        st.write(df2.head(1))

    min_date = df2['受注日'].min()
    max_date = df2['受注日'].max()
    st.write(f'【dataframe length】{len(df2)} 【min_date】{min_date} 【max_date】{max_date}')


    if prompt:
        st.markdown('##### response')
        with st.expander('content', expanded=True):
            st.write(pandas_ai.run(df2, prompt=prompt))
    else:
        st.warning('plase enter a prompt.')


