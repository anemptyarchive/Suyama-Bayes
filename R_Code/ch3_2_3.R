
# 3.2.3 ポアソン分布の学習と予測 ---------------------------------------------------------

# 3.2.3項で利用するパッケージ
library(tidyverse)


### 尤度(ポアソン分布)の設定 -----

# 真のパラメータを指定
lambda_truth <- 4


# 作図用のxの値を作成
x_vec <- seq(0, 4 * lambda_truth)

# 尤度を計算:式(2.37)
model_df <- tibble(
  x = x_vec, # x軸の値
  ln_C_Poi = x * log(lambda_truth) - lgamma(x + 1), # 正規化項(対数)
  prob = exp(ln_C_Poi - lambda_truth) # 確率
  #prob = dpois(x = x, lambda = lambda_truth) # 確率
)

# 尤度を作図
ggplot(model_df, aes(x = x, y = prob)) + 
  geom_bar(stat = "identity", position = "dodge", fill = "blue") + # 尤度
  labs(title = "Poisson Distribution", 
       subtitle = paste0("lambda=", lambda_truth))


### データの生成 -----

# (観測)データ数を指定
N <- 50

# ポアソン分布に従うデータを生成
x_n <- rpois(n = N ,lambda = lambda_truth)

# 観測データを確認
table(x_n)


# 観測データのデータフレームを作成
data_df <- tibble(x_n = x_n)

# 観測データのヒストグラムを作図
ggplot() + 
  #geom_histogram(data = data_df, aes(x = x_n), binwidth = 1) + # 観測データ:(度数)
  geom_histogram(data = data_df, aes(x = x_n, y = ..density..), binwidth = 1) + # 観測データ:(相対度数)
  geom_bar(data = model_df, aes(x = x, y = prob), 
           stat = "identity", position = "dodge", 
           alpha = 0, color = "red", linetype = "dashed") + # 真の分布
  labs(title = "Poisson Distribution", 
       subtitle = paste0("N=", N, ", lambda=", lambda_truth), 
       x = "x")


### 事前分布(ガンマ分布)の設定 -----

# 事前分布のパラメータを指定
a <- 1
b <- 1


# 作図用のlambdaの値を作成
lambda_vec <- seq(0, 2 * lambda_truth, by = 0.005)

# 事前分布を計算:式(2.56)
prior_df <- tibble(
  lambda = lambda_vec, # x軸の値
  ln_C_Gam = a * log(b) - lgamma(a), # 正規化項(対数)
  density = exp(ln_C_Gam + (a - 1) * log(lambda) - b * lambda) # 確率密度
  #density = dgamma(x = lambda, shape = a, rate = b) # 確率密度
)

# 事前分布を作図
ggplot(prior_df, aes(x = lambda, y = density)) + 
  geom_line(color = "purple") + # 事前分布
  geom_vline(aes(xintercept = lambda_truth), 
             color = "red", linetype = "dashed") + # 真の値
  labs(title = "Gamma Distribution", 
       subtitle = paste0("a=", a, ", b=", b), 
       x = expression(lambda))


### 事後分布(ガンマ分布)の計算 -----

# 事後分布のパラメータを計算:式(3.38)
a_hat <- sum(x_n) + a
b_hat <- N + b


# 事後分布を計算:式(2.56)
posterior_df <- tibble(
  lambda = lambda_vec, # x軸の値
  ln_C_Gam = a_hat * log(b_hat) - lgamma(a_hat), # 正規化項(対数)
  density = exp(ln_C_Gam + (a_hat - 1) * log(lambda) - b_hat * lambda) # 確率密度
  #density = dgamma(x = lambda, shape = a_hat, rate = b_hat) # 確率密度
)

# 事後分布を作図
ggplot(posterior_df, aes(x = lambda, y = density)) + 
  geom_line(color = "purple") + # 事後分布
  geom_vline(aes(xintercept = lambda_truth), 
             color = "red", linetype = "dashed") + # 真の値
  labs(title = "Gamma Distribution", 
       subtitle = paste0("N=", N, ", a_hat=", a_hat, ", b_hat=", b_hat), 
       x = expression(lambda))


### 予測分布(負の二項分布)の計算 -----

# 予測分布のパラメータを計算:式(3.44')
r_hat <- a_hat
p_hat <- 1 / (b_hat + 1)
#r_hat <- sum(x_n) + a
#p_hat <- 1 / (N + 1 + b)


# 予測分布を計算:式(3.43)
predict_df <- tibble(
  x = x_vec, # x軸の値
  ln_C_NB = lgamma(x + r_hat) - lgamma(x + 1) - lgamma(r_hat), # 正規化項(対数)
  prob = exp(ln_C_NB + r_hat * log(1 - p_hat) + x * log(p_hat)) # 確率
  #prob = dnbinom(x = x, size = r_hat, prob = 1 - p_hat) # 確率
)

# 予測分布を作図
ggplot() + 
  geom_bar(data = predict_df, aes(x = x, y = prob), 
           stat = "identity", position = "dodge", fill = "purple") + # 予測分布
  geom_bar(data = model_df, aes(x = x, y = prob), 
           stat = "identity", position = "dodge", 
           alpha = 0, color = "red", linetype = "dashed") + # 真の分布
  labs(title = "Negative Binomial Distribution", 
       subtitle = paste0("N=", N, ", r_hat=", r_hat, ", p_hat=", round(p_hat, 3)))


# ・アニメーションによる推移の確認 ---------------------------------------------------------------

# 3.2.3項で利用するパッケージ
library(tidyverse)
library(gganimate)


### モデルの設定 -----

# 真のパラメータを指定
lambda_truth <- 4

# 事前分布のパラメータを指定
a <- 1
b <- 1


# 作図用のlambdaの値を作成
lambda_vec <- seq(0, 2 * lambda_truth, length.out = 1000)

# 事前分布(ガンマ分布)を計算
posterior_df <- tibble(
  lambda = lambda_vec, # x軸の値
  density = dgamma(x = lambda, shape = a, rate = b), # 確率密度
  label = as.factor(paste0("N=", 0, ", a=", a, ", b=", b))
)


# 初期値による予測分布のパラメータを計算:式(3.44)
r <- a
p <- 1 / (b + 1)


# 作図用のxの値を作成
x_vec <- seq(0, 4 * lambda_truth)

# 初期値による予測分布(負の二項分布)を計算
predict_df <- tibble(
  x = x_vec, # 作図用の値
  prob = dnbinom(x = x, size = r, prob = 1 - p), # 確率
  label = as.factor(paste0("N=", 0, ", r=", r, ", p=", round(p, 5))) # フレーム切替用のラベル
)


### 推論処理 -----

# データ数(試行回数)を指定
N <- 100

# 観測データの受け皿を作成
x_n <- rep(0, N)

# ベイズ推論
for(n in 1:N){
  
  # ポアソン分布に従うデータを生成
  x_n[n] <- rpois(n = 1 ,lambda = lambda_truth)
  
  # 事後分布のパラメータを更新:式(3.38)
  a <- sum(x_n[n] * 1) + a
  b <- 1 + b
  
  # 事後分布(ガンマ分布)を計算:式(2.56)
  tmp_posterior_df <- tibble(
    lambda = lambda_vec, # x軸の値
    density = dgamma(x = lambda, shape = a, rate = b), # 確率密度
    label = as.factor(paste0("N=", n, ", a_hat=", a, ", b_hat=", b)) # フレーム切替用のラベル
  )
  
  # 予測分布のパラメータを更新:式(3.44)
  r <- a
  p <- 1 / (b + 1)
  
  # 予測分布(負の二項分布)を計算:式(3.43)
  tmp_predict_df <- tibble(
    x = x_vec, # x軸の値
    prob = dnbinom(x = x, size = r, prob = 1 - p), # 確率
    label = as.factor(paste0("N=", n, ", r_hat=", r, ", p_hat=", round(p, 5))) # フレーム切替用のラベル
  )
  
  # n回目の結果を結合
  posterior_df <- rbind(posterior_df, tmp_posterior_df)
  predict_df <- rbind(predict_df, tmp_predict_df)
}

# 観測データを確認
table(x_n)


### 作図処理 -----

# 観測データのデータフレームを作成
data_df <- tibble(
  x_n = c(NA, x_n), 
  label = unique(posterior_df[["label"]]) # フレーム切替用のラベル
)

# 事後分布を作図
posterior_graph <- ggplot() + 
  geom_line(data = posterior_df, aes(x = lambda, y = density), 
            color = "purple") + # 事後分布
  geom_vline(aes(xintercept = lambda_truth), 
             color = "red", linetype = "dashed") + # 真の値
  geom_point(data = data_df, aes(x = x_n, y = 0), size = 5) + # 観測データ
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Gamma Distribution", 
       subtitle = "{current_frame}", 
       x = expression(lambda))

# gif画像を出力
gganimate::animate(posterior_graph, nframes = N + 1, fps = 10)


# 尤度をN+1フレーム分に複製
model_df <- tibble(
  x = rep(x_vec, times = N + 1), # x軸の値
  prob = rep(dpois(x = x_vec, lambda = lambda_truth), times = N + 1), # 確率
  label = predict_df[["label"]] # フレーム切替用のラベル
)

# 観測データのデータフレームを作成
data_df <- tibble(
  x_n = c(NA, x_n), 
  label = unique(predict_df[["label"]]) # フレーム切替用のラベル
)

# 予測分布を作図
predict_graph <- ggplot() + 
  geom_bar(data = predict_df, aes(x = x, y = prob), 
           stat = "identity", position = "dodge", fill = "purple") + # 予測分布
  geom_bar(data = model_df, aes(x = x, y = prob), 
           stat = "identity", position = "dodge", 
           alpha = 0, color = "red", linetype = "dashed") + # 真の分布
  geom_point(data = data_df, aes(x = x_n, y = 0), size = 5) + # 観測データ
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Negative Binomial Distribution", 
       subtitle = "{current_frame}")

# gif画像を出力
gganimate::animate(predict_graph, nframes = N + 1, fps = 10)


