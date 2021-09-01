> ### 기존 RNN의 문제점

- 첫째, 하나의 고정된 크기의 벡터에 모든 정보를 압축하려고 하니까 정보 손실이 발생한다. 


- 둘째, RNN의 고질적인 문제인 기울기 소실(Vanishing Gradient) 문제가 존재한다. 

---

# # Attention Mechanism

> ### Attention의 아이디어

- 디코더에서 출력 단어를 예측하는 매 시점(time step)마다, 인코더에서 전체 입력 문장을 다시 한 번 참고한다. 


- 단, 전체 입력 문장을 전부 다 동일한 비율로 참고하는 것이 아니라, 해당 시점에서 예측해야할 단어와 연관이 있는 입력 단어 부분을 좀 더 집중(attention)해서 보게 됩니다.

   ---

> ### Attention 함수(Attention Function)

<img src="https://user-images.githubusercontent.com/59716219/131542659-8a615bfc-1058-4a02-8b0c-1912f861c7cf.png" width="400" height="250">

**Attention(Q,K,V) = Attention Value**

-  어텐션 함수는 주어진 `쿼리(Query)`에 대해서 모든 `키(Key)`와의 유사도를 각각 구한다.  


-  이 구한 유사도를 키와 맵핑되어 있는 각각의 `값(Value)`을 모두 더해서 리턴합니다. 이를 `어텐션 값(Attention Value)`라고 한다.  
-  seq2seq + 어텐션 모델에서 Q,K,V에 해당되는 각의 Query, Keys, Values는 각각 다음과 같다.  
   - `Q = Query : t 시점의 디코더 셀에서의 은닉 상태`
   - `K = Keys : 모든 시점의 인코더 셀의 은닉 상태들`
   - `V = Values : 모든 시점의 인코더 셀의 은닉 상태들`

---

# # 닷-프로덕트 어텐션(Dot-Product Attention)(Luong Attention)

<img src="https://user-images.githubusercontent.com/59716219/131544271-33086b05-599c-4293-8fad-82d8d455e51e.png" width="550" height="400">

- softmax를 통해 나온 결과값은  I, am, a, student 단어 각각이 출력 단어를 예측할 때 얼마나 도움이 되는지의 정도를 수치화한 값이다.


-  입력 단어가 디코더의 예측에 도움이 되는 정도가 수치화하여 측정되면 이를 하나의 정보로 담아서 디코더로 전송한다.(위 그림에서는 초록색 삼각형)

   ----
   
> ## 1)어텐션 스코어(Attention Score)를 구한다.

<img src="https://user-images.githubusercontent.com/59716219/131546588-24b8d4fb-e073-48e6-b652-9cf55ec01af5.png" width="500" height="300">

- <img src="https://render.githubusercontent.com/render/math?math=h_1, h_2, ... , h_N"> : 인코더의 각 시점의 은닉 상태
- <img src="https://render.githubusercontent.com/render/math?math=s_t"> : 디코더의 시점 t에서의 은닉 상태
- (인코더의 은닉 상태의 차원 == 디코더의 은닉 상태 차원)이라고 가정
- `Attention Score` : 인코더의 모든 은닉 상태들이 디코더의 현재 은닉 상태 <img src="https://render.githubusercontent.com/render/math?math=s_t">와 얼마나 유사한지를 판단하는 스코어값

   <img src="https://user-images.githubusercontent.com/59716219/131548585-0894a604-1e06-4f07-af7e-e5df7da55e9b.png" width="200" height="140">

- 현재 : dot product attention
- 현재 어텐션 스코어를 구하기 위한 식 :  <img src="https://render.githubusercontent.com/render/math?math=score(s_t, h_i)  = s_t^Th_i">
- 모든 어텐션 스코어 값은 스칼라
- <img src="https://render.githubusercontent.com/render/math?math=e^t"> : 어텐션 스코어의 모음값
- <img src="https://render.githubusercontent.com/render/math?math=e^t = [s_t^Th_1, ... , s_t^Th_N]">

   ---
   
> ## 2)소프트맥스(softmax) 함수를 통해 어텐션 분포(Attention Distribution)를 구한다.

<img src="https://user-images.githubusercontent.com/59716219/131549587-d4ea8cca-3ff7-42ab-be3d-b87466a8ee14.png" width="450" height="250">

- `어텐션 분포(Attention Distribution)` : <img src="https://render.githubusercontent.com/render/math?math=e^t">에 softmax함수를 적용한 것
- `어텐션 가중치(Attention Weight)` : Attention Distribution의 각각의 값
- <img src="https://render.githubusercontent.com/render/math?math={\alpha}^t = softmax(e^t)"> : 어텐션 분포

   ---

> ## 3) 각 인코더의 어텐션 가중치와 은닉 상태를 가중합하여 어텐션 값(Attention Value)을 구한다.

<img src="https://user-images.githubusercontent.com/59716219/131550703-4cdfe353-15ec-4e3e-9997-479a814b3531.png" width="450" height="300">

- Attention Value를 구하기 위해 가중합을 한다.(각 인코더의 은닉 상태와 어텐션 가중치값들을 곱하고, 최종적으로 모두 더함)
- <img src="https://render.githubusercontent.com/render/math?math=a_t"> : Attention Value
- <img src="https://render.githubusercontent.com/render/math?math=a_t = \sum_{i=1}^N\alpha_i^th_i">
- Attention Value는 인코더의 컨텍스트 벡터라고도 한다. 

   ---

> ## 4) 어텐션 값과 디코더의 t 시점의 은닉 상태를 연결한다.(Concatenate)

<img src="https://user-images.githubusercontent.com/59716219/131589352-e2435cd2-dc6a-41bd-9bf5-e0f5a2795ae2.png" width="450" height="250">

- 어텐션 메커니즘은 <img src="https://render.githubusercontent.com/render/math?math=a_t">를 <img src="https://render.githubusercontent.com/render/math?math=s_t">와 결합(concatenate)함
- 이 concatenate한 것을 <img src="https://render.githubusercontent.com/render/math?math=v_t">라고 함. 
- <img src="https://render.githubusercontent.com/render/math?math=v_t">를 <img src="https://render.githubusercontent.com/render/math?math=\hat y">예측 연산의 입력으로 사용하므로서 인코더로부터 얻은 정보를 활용하여 결과값을 예측하게 됨. 

   ---

> ## 5) 출력층 연산의 입력이 되는 <img src="https://render.githubusercontent.com/render/math?math=\tilde{s_t}">를 계산한다. 

<img src="https://user-images.githubusercontent.com/59716219/131594499-5e5d9294-780f-4038-a7fa-274683323041.png" width="350" height="180">


- <img src="https://render.githubusercontent.com/render/math?math=v_t">를 출력층으로 보내 전 신경망 연산을 한번 더 추가함.
- 중치 행렬과 곱한 후에 하이퍼볼릭탄젠트 함수를 지나도록 하여 출력층 연산을 위한 새로운 벡터인 <img src="https://render.githubusercontent.com/render/math?math=\tilde{s_t}">를 얻는다.(이는 출력층의 입력이 된다)
- <img src="https://render.githubusercontent.com/render/math?math=W_c"> : 학습 가능한 가중치 행렬
- <img src="https://render.githubusercontent.com/render/math?math=b_c"> : 편향(그림에서 생략)
- <img src="https://render.githubusercontent.com/render/math?math=\tilde{s_t} = tanh(W_c[a_t"> ;<img src="https://render.githubusercontent.com/render/math?math=s_t]"> + <img src="https://render.githubusercontent.com/render/math?math=b_c)">

   ---

> ## 6) <img src="https://render.githubusercontent.com/render/math?math=\tilde{s_t}">를 출력층의 입력으로 사용합니다.

- <img src="https://render.githubusercontent.com/render/math?math=\hat{y_t} = Softmax(W_y\tilde{s_t}"> + <img src="https://render.githubusercontent.com/render/math?math=b_y)">

---

# # 다양한 종류의 어텐션(Attention)

- 위에서 배운 어텐션 닷-프로덕트 어텐션인 이유? : 어텐션 스코어를 구하는 방법이 내적이었기 때문.
- 다양한 어텐션 스코어를 구하는 방법이 있다.

<img src="https://user-images.githubusercontent.com/59716219/131595196-d8a57080-4c27-4eb4-b48c-799197a42730.png" width="530" height="350">

- <img src="https://render.githubusercontent.com/render/math?math=s_t"> : Query
- <img src="https://render.githubusercontent.com/render/math?math=h_i"> : Keys
- <img src="https://render.githubusercontent.com/render/math?math=W_a, W_b"> : 학습 가능한 가중치 행렬

---

# # Luong Attention & Bahdanau Attention 차이점 

- `Query`
   - Loung : t시점의 디코더 셀에서의 은닉 상태
   - Bahdanau : t-1시점의 디코더 셀에서의 은닉 상태

- `Attention Score`
   - Loung : <img src="https://render.githubusercontent.com/render/math?math=score(s_t,h_i) = s_t^Th_i">
   - Bahdanau : <img src="https://render.githubusercontent.com/render/math?math=score(s_{t-1}, h_i) = W_a^Ttanh(W_bs_{t-1} "> + <img src="https://render.githubusercontent.com/render/math?math=W_ch_i)">

- `Concatenate with Context Vector(Attention Value)`
   - Luong : 디코더의 hidden state vector(<img src="https://render.githubusercontent.com/render/math?math=s_t">) + context vector
   - Bahdanau : Embedding of Previous Decoder Output + context vector

---

# # 바다나우 어텐션(Bahdanau Attention)

- **Attention(Q,K,V) = Attention Value**
   - `Q = Query : t-1 시점의 디코더 셀에서의 은닉 상태`
   - `K = Keys : 모든 시점의 인코더 셀의 은닉 상태들`
   - `V = Values : 모든 시점의 인코더 셀의 은닉 상태들`


> ## 1)어텐션 스코어(Attention Score)를 구한다.

<img src="https://user-images.githubusercontent.com/59716219/131597689-1b7c42aa-ce15-4633-94dd-31892fbacc5c.png" width="500" height="300">

- t-1시점의 은닉상태 <img src="https://render.githubusercontent.com/render/math?math=s_{t-1}">을 사용
- <img src="https://render.githubusercontent.com/render/math?math=score(s_{t-1}, h_i) = W_a^Ttanh(W_bs_{t-1}"> + <img src="https://render.githubusercontent.com/render/math?math=W_ch_i)">
   - 이때 <img src="https://render.githubusercontent.com/render/math?math=W_a, W_b, W_c">는 학습 가능한 가중치 행렬
   - <img src="https://render.githubusercontent.com/render/math?math=s_{t-1}">와 <img src="https://render.githubusercontent.com/render/math?math=h_1, h_2, h_3, h_4">의 어텐션 스코어는 각각 구해야한다. 
   - 병렬 연산을 위해 <img src="https://render.githubusercontent.com/render/math?math=h_1, h_2, h_3, h_4">를 하나의 행렬 H로 두자
- <img src="https://render.githubusercontent.com/render/math?math=score(s_{t-1}, h_i) = W_a^Ttanh(W_bs_{t-1}"> + <img src="https://render.githubusercontent.com/render/math?math=W_cH)">

- <img src="https://render.githubusercontent.com/render/math?math=W_bs_{t-1}, W_cH">를 그림으로 보자

   <img src="https://user-images.githubusercontent.com/59716219/131598491-d260f7b0-db96-4b5d-b0da-db450f939f22.png" width="300" height="170">
   
- 이들을 더한 후, 하이퍼볼릭탄젠트 함수를 지나도록 한다. (현재까지 진행된 수식 <img src="https://render.githubusercontent.com/render/math?math=tanh(W_bs_{t-1}"> + <img src="https://render.githubusercontent.com/render/math?math=W_cH)">)
   
   <img src="https://user-images.githubusercontent.com/59716219/131598757-89dd12a7-11cd-474b-ba71-eea1eaa67202.png" width="300" height="100">
   
- <img src="https://render.githubusercontent.com/render/math?math=W_a^T">와 곱하여  <img src="https://render.githubusercontent.com/render/math?math=s_{t-1}, h_1, h_2, h_3, h_4">의 유사도가 기록된 어텐션 스코어 벡터  <img src="https://render.githubusercontent.com/render/math?math=e^t">를 얻는다.

   ---
   
> ## 2)소프트맥스(softmax) 함수를 통해 어텐션 분포(Attention Distribution)를 구한다.

<img src="https://user-images.githubusercontent.com/59716219/131599260-f3840f83-c138-4099-8698-6d80bf7e29b9.png" width="280" height="90">

- Attention Distribution : <img src="https://render.githubusercontent.com/render/math?math=e_t">에 소프트맥스 함수 적용한 것.
- Attention Distribution에서 각각의 값은 어텐션 가중치(Attention Weight)라고 함.

   ---

> ## 3) 각 인코더의 어텐션 가중치와 은닉 상태를 가중합하여 어텐션 값(Attention Value)을 구한다.

<img src="https://user-images.githubusercontent.com/59716219/131599530-df26b2d9-0393-4f39-ac10-7658c44857b9.png" width="250" height="100">

- 어텐션의 최종 결과값을 얻기 위해 각 인코더의 은닉 상태와 어텐션 가중치값들을 곱하고, 최종적으로 모두 더한다. 
- 가중합을 한다고 말할 수 있다. 
- Attention Value는 인코더의 문맥을 포함하고 있다고 하여 Context Vector라고도 부름.


   ---

> ## 4) 컨텍스트 벡터로부터 <img src="https://render.githubusercontent.com/render/math?math=s_t">를 구한다.

<img src="https://user-images.githubusercontent.com/59716219/131600470-03cf0f3c-79d3-4ca6-b60b-ffc51177140f.png" width="220" height="200"> <img src="https://user-images.githubusercontent.com/59716219/131600482-891b2388-df25-48b0-b52a-7318410b2321.png" width="300" height="200">

- 바다나우 어텐션 메커니즘에서는 context vector와 현재 시점의 입력이 단어의 임베딩 벡터를 연결(concatenate)하고, 현 시점의 새로운 입력으로 사용
- <img src="https://render.githubusercontent.com/render/math?math=s_{t-1}">과 현재 시점의 새로운 입력으로부터 <img src="https://render.githubusercontent.com/render/math?math=s_{t}">를 구한다.
- <img src="https://render.githubusercontent.com/render/math?math=s_{t}">는 출력층으로 전달되어 현재 시점의 예측값을 구하게 된다. 

---

# # 양방향 LSTM & Attention mechanism

> ## 1. IMDB 리뷰 데이터 전처리하기

```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocab_size)
# X_train의 최대 길이 : 2494
# X_train의 평균 길이 : 238

# padding
max_len = 500
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)
```

> ## 2. 바다나우 어텐션(Bahdanau Attention) 구현

```python
import tensorflow as tf
class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = Dense(units)
    self.W2 = Dense(units)
    self.V = Dense(1)

  def call(self, values, query): # 단, key와 value는 같음
    # query shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # score 계산을 위해 뒤에서 할 덧셈을 위해서 차원을 변경해줍니다.
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
```

> ## 3. 양방향 LSTM + 어텐션 메커니즘(BiLSTM with Attention Mechanism)

```python
# 입력층 설계
sequence_input = Input(shape=(max_len,), dtype='int32')
# 임베딩층 설계
embedded_sequences = Embedding(vocab_size, 128, input_length=max_len, mask_zero = True)(sequence_input)
# 양방향 LSTM 두 층을 사용할 것임.
# 첫번째 층이므로 두번째 층을 위에 쌓아야기때문에 return_sequences = True로 해줌
lstm = Bidirectional(LSTM(64, dropout=0.5, return_sequences = True))(embedded_sequences)
# 두번쨰 층. 상태를 리턴받아야 하므로 return_state = True로 해줌
lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional \
  (LSTM(64, dropout=0.5, return_sequences=True, return_state=True))(lstm)
# LSTM 상태 크기 : (None, 500, 128)// h, c의 상태 크기 : (None, 64)

# 양방향 LSTM을 사용할 경우, 순방향 LSTM과 역방향 LSTM은 각각의 은닉 상태와 셀 상태를 가진다. 양방향 LSTM의 은닉 상태와 셀 상태를 사용려면 두 방향의 LSTM의 상태들을 연결(concatenate)
state_h = Concatenate()([forward_h, backward_h]) # 은닉 상태
state_c = Concatenate()([forward_c, backward_c]) # 셀 상태

# 어텐션 메커니즘에서는 은닉 상태를 사용한다. 이를 입력으로 컨텍스트 벡터(context vector)를 얻는다.
attention = BahdanauAttention(64) # 가중치 크기 정의
context_vector, attention_weights = attention(lstm, state_h)

# 컨텍스트 벡터를 밀집층(dense layer)에 통과시키고, 이진 분류이므로 최종 출력층에 1개의 뉴런을 배치하고, 활성화 함수로 시그모이드 함수를 사용한다. 
dense1 = Dense(20, activation="relu")(context_vector)
dropout = Dropout(0.5)(dense1)
output = Dense(1, activation="sigmoid")(dropout)
model = Model(inputs=sequence_input, outputs=output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs = 3, batch_size = 256, validation_data=(X_test, y_test), verbose=1)

```
