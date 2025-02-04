import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import streamlit as st

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

        X=df[selected_columns]

        #4. 숫자 컬럼인지, 문자열 컬럼인지 확인하고 인코딩작업을 해줘야한다.
        num_columns=df[selected_columns].select_dtypes(include='number').columns
        str_columns=df[selected_columns].select_dtypes(include='object').columns
        #4-1. 문자열 컬럼 인코딩
        if len(str_columns) > 0:
            encoder = OneHotEncoder(sparse=False, drop="first")  # drop="first"로 더미 트랩 방지 가능
            encoded = encoder.fit_transform(df[str_columns])  

            # 변환된 데이터를 데이터프레임으로 변환
            dfOneHot = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(str_columns))

            # 기존 데이터프레임과 결합
            df = pd.concat([df, dfOneHot], axis=1)

            # 문자열 컬럼 삭제
            df.drop(columns=str_columns, inplace=True)

        #5. 스케일링은 모델이 알아서 해준다.
        #6. 숫자, 문자열 컬럼을 X로 지정
        X = df[num_columns.tolist() + dfOneHot.columns.tolist()]

        
        
        


            
            
        
        




if __name__ == '__main__':
    main()