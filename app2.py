from matplotlib import pyplot as plt
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import streamlit as st
from pandas.api.types import is_integer_dtype, is_float_dtype, is_object_dtype
from sklearn.compose import ColumnTransformer

def main():
    st.title('K-Means Clustering')
    st.info('This is a simple example of K-Means clustering using Streamlit and Scikit-learn')

    #1. csv file upload
    file=st.file_uploader('Upload CSV', type=['csv'])
    if file is not None:
        #2.데이터 불러오기
        df=pd.read_csv(file)
        st.dataframe(df.head(5))
        #3. 유저가 컬럼을 선택할 수 있도록한다. -> 선택한 컬럼을 X로 지정
        st.info('K-Means Clustering에 사용할 컬럼을 선택해주세요.')
        selected_columns=st.multiselect('컬럼 선택', df.columns)

        #중요: Nan값이 있는지 확인
        st.info('선택한 컬럼에 NaN값이 있는지 확인 후 삭제하겠습니다.')
        st.dataframe(df.isna().sum())
        df.dropna()


        df_new=pd.DataFrame()
        #4. 각 컬럼이 문자열인지 숫자열인지 확인
        for column in selected_columns:
            if is_integer_dtype(df[column]):
                df_new[column]=df[column]
            # 숫자면 그대로 사용
            elif is_float_dtype(df[column]):
                df_new[column]=df[column]
            # 문자열이면 one-hot encoding
            elif is_object_dtype(df[column]):
                #유니크값 확인 1,2면 레이블인코딩
                if len(df[column].unique()) <= 2:
                    encoder=LabelEncoder()
                    encoder.fit(df[column])
                    encoder_result=encoder.transform(df[column])
                    df_new[column]=encoder_result
                else:
                    #3개 이상이면 원핫 인코딩 
                    ct = ColumnTransformer([('encoder',OneHotEncoder(),[0])] , passthrough=True)
                    column_names=sorted(df[column].unique())
                    df_new[column_names]=ct.fit_transform(df[column])
                    #해당 컬럼에 들어있던 유니크한 값들로 새로운 컬럼을 만들어준다.
                    
            else:
                st.error(f'{column}은 숫자 또는 문자열이어야 합니다.')
                return
            
        st.info('K-Means Clustering을 아래 DataFream으로 시작합니다.')    
        st.dataframe(df_new.head(5))

        st.subheader('최적의 k값을 찾기위해 WCSS를 계산합니다.')
        st.text(f'데이터의 개수는 {df_new.shape[0]}개 입니다.')

        # 데이터가 10개 미만인데 10개 그룹으로 나눠달라하면 에러남.
        # 해당 데이터의 최대 개수로 K값을 설정
        if df_new.shape[0] < 10:
            max_k=st.slider('K값을 선택해주세요.', min_value=2, max_value=df_new.shape[0])
        else:
            max_k=st.slider('K값을 선택해주세요.', min_value=2, max_value=10)

        #누르면 계산하는 버튼
        if st.button('WCSS 계산하기'):
            wcss=[]
            for k in range (1, max_k+1):
                kmeans=KMeans(n_clusters=k,random_state=4)
                kmeans.fit(df_new)
                wcss.append(kmeans.inertia_)

            fig1=plt.figure()
            plt.plot(range(1,max_k+1),wcss)
            st.pyplot(fig1)

            st.text('위 그래프를 보고 최적의 K값을 선택하세요.')

            k=st.number_input('K값을 입력해주세요.', min_value=2, max_value=max_k)
            kmeans=KMeans(n_clusters=k, random_state=4)
            df['Group']=kmeans.fit_predict(df_new)
            st.dataframe(df.head(5))

            




if __name__ == '__main__':
    main()