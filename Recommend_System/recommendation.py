#recommend.py 코드
import pandas as pd
#timestamp에 기록하기 위해서
import datetime as pydatetime
import numpy as np
from scipy.sparse.linalg import svds

class recommend:
    def init(self, userID):
        self.uId = userID;
        self.pivot_table=pd.Dataframe()
        
    def add(self, movieTitle, score):
        movies_df = pd.read_csv('./ml-latest-small/movie_cert.csv')
        rating_df = pd.read_csv('./ml-latest-small/rating_cert.csv')
        
        #movileTitle이 movie_cert에 없다면 에러 출력후 리턴
        temp = movies_df[movies_df['title']==movieTitle]
        if(len(temp)<1 or len(temp)>2):
            print("해당 영화가 없거나 에러가 존재합니다. 다시 입력하세요")
            return 
        
        movieId =temp.iloc[0]['movieId']
        #현재의 timestamp
        curTime = int(pydatatime.datatime.now().timestamp())
        
        #rating_cert.csv에 데이터 추가
        toAppend = {"userId": self.uId, "movieId" : movieId,
                   "rating":score, "timestamp":curTime}
        
        rating_df =rating_df.append(toAppend, ignore_index = True)
        
        rating_df.to_csv('./ml-latest-small/rating_cert.csv',index =False)
        
    def prediction(self, num_recommendations):
        # movie_predict 테이블을 만들기 위한 과정
        movies_df = pd.read_csv('./ml-latest-small/movie_cert.csv')
        ratings_df = pd.read_csv('./ml-latest-small/rating_cert.csv')
        
        pivot_df = ratings_df.pivot(index ='userId',columns='movieId',values='rating').fillna(0)

        R = pivot_df.values
        
        #user_ratings_mean은 사용자의 평균 평점
        user_ratings_mean = np.mean(R,axis =1)
        
        #R_demeaned : 사용자-영화에대한 사용자의 평균 평점
        R_demeaned = R - user_ratings_mean.reshape(-1,1)
        
        #U 행렬, sigma 행렬, V전치행렬을 반환
        #이때 spicy에 있는 svd를 이용
        U,sigma,Vt = svds(R_demeaned,k=50)
        
        sigma = np.diag(sigma)
        
        all_user_predicted_ratings = np.dot(np.dot(U,sigma),Vt)+user_ratings_mean.reshape(-1,1)
        
        preds_df = pd.DataFrame(all_user_predicted_ratings, columns=pivot_df.columns)
        
        # 모델 부분.
        user_row_number = list(pivot_df.index).index(self.uId)
        
        #앞에서 만든 prediction행렬을 사용자 인덱스에 따라 평점이 높은 순으로 영화를 정렬
        sorted_user_predicitons = preds_df.iloc[user_row_number].sort_values(ascending=False)
        
        # rating_cert에서 user_id에 해당하는 정보(movieID, rating)를 얻는다.
        new_data = pivot_df.loc[[self.uId]]
        watched_movies=[]
        for c in pivot_df.columns.to_list():
            if new_data.iloc[0][c] != 0:
                watched_movies.append(c)
        
        recommendations = pd.merge(sorted_user_predictions,movies_df, on='movieId')
        
        recommendations = recommendations[~recommendations['movieId'].isin(watched_movies)]
        
        return recommendations['title'][:num_recommendations].to_list()
        
        
        