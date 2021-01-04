#main.py 코드

import recommendation as myRecommend

print("***********영화 추천 프로그램**************")
while True :
    userID =int(input("로그인(-1 to end program): "))
    if userID ==-1 :
        break;
    print()
    
    rec = myRecommend.recommend(userID)
    
    while True:
        movieTitle = str(input("영화 제목 : "))
        if movieTitle =="":
            print("-----------------")
            break
        score = float(input(movieTitle+"의 평점 :"))
        rec.add(movieTitle, score)
        print();
        
    recommended = rec.prediction(10)

    print("< (user",userID,")님께 추천 드리는 영화 목록 >")
    for movie in recommended :
        print(movie)
        
    print("\n")