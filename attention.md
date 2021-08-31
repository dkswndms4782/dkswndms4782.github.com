<img src="https://render.githubusercontent.com/render/math?math="> 

> ### 기존 RNN의 문제점

- 첫째, 하나의 고정된 크기의 벡터에 모든 정보를 압축하려고 하니까 정보 손실이 발생한다. 


- 둘째, RNN의 고질적인 문제인 기울기 소실(Vanishing Gradient) 문제가 존재한다. 

---

# # Attention Mechanism

> ### Attention의 아이디어

- 디코더에서 출력 단어를 예측하는 매 시점(time step)마다, 인코더에서 전체 입력 문장을 다시 한 번 참고한다. 


- 단, 전체 입력 문장을 전부 다 동일한 비율로 참고하는 것이 아니라, 해당 시점에서 예측해야할 단어와 연관이 있는 입력 단어 부분을 좀 더 집중(attention)해서 보게 됩니다.

> ### Attention 함수(Attention Function)

<img src="https://user-images.githubusercontent.com/59716219/131542659-8a615bfc-1058-4a02-8b0c-1912f861c7cf.png" width="400" height="250">

**Attention(Q,K,V) = Attention Value**

-  어텐션 함수는 주어진 `쿼리(Query)`에 대해서 모든 `키(Key)`와의 유사도를 각각 구한다.  


-  이 구한 유사도를 키와 맵핑되어 있는 각각의 `값(Value)`을 모두 더해서 리턴합니다. 이를 `어텐션 값(Attention Value)`라고 한다.  
-  seq2seq + 어텐션 모델에서 Q,K,V에 해당되는 각의 Query, Keys, Values는 각각 다음과 같다.  
   - `Q = Query : t 시점의 디코더 셀에서의 은닉 상태`
   - `K = Keys : 모든 시점의 인코더 셀의 은닉 상태들`
   - `V = Values : 모든 시점의 인코더 셀의 은닉 상태들`

# # 닷-프로덕트 어텐션(Dot-Product Attention)

<img src="https://user-images.githubusercontent.com/59716219/131544271-33086b05-599c-4293-8fad-82d8d455e51e.png" width="550" height="400">

- softmax를 통해 나온 결과값은  I, am, a, student 단어 각각이 출력 단어를 예측할 때 얼마나 도움이 되는지의 정도를 수치화한 값이다.


-  입력 단어가 디코더의 예측에 도움이 되는 정도가 수치화하여 측정되면 이를 하나의 정보로 담아서 디코더로 전송한다.(위 그림에서는 초록색 삼각형)


