
# 3.3.2 1次元ガウス分布の学習と予測：精度が未知の場合 ------------------------------------------------------------------

# 3.3.2項で利用するパッケージ
library(tidyverse)


### 尤度(ガウス分布)の設定 -----

# (既知の)パラメータを指定
mu <- 25

# 真の精度パラメータを指定
lambda_truth <- 0.01
sqrt(1 / lambda_truth) # 標準偏差


# 作図用のxの値を作成
x_vec <- seq(
  mu - 4 * sqrt(1 / lambda_truth), 
  mu + 4 * sqrt(1 / lambda_truth), 
  length.out = 1000
)

# 尤度を計算:式(2.64)
model_df <- tibble(
  x = x_vec, # x軸の値
  ln_C_N = - 0.5 * (log(2 * pi) - log(lambda_truth)), # 正規化項(対数)
  density = exp(ln_C_N - 0.5 * lambda_truth * (x - mu)^2) # 確率密度
  #C_N = 1 / sqrt(2 * pi / lambda_truth), # 正規化項
  #density = C_N * exp(- 0.5 * lambda_truth * (x - mu)^2) # 確率密度
  #density = dnorm(x, mean = mu, sd = sqrt(1 / lambda_truth)) # 確率密度
)

# 尤度を作図
ggplot(model_df, aes(x = x, y = density)) + 
  geom_line(color = "blue") + # 尤度
  labs(title = "Gaussian Distribution", 
       subtitle = paste0("mu=", round(mu, 2), ", lambda=",lambda_truth))


### データの生成 -----

# (観測)データ数を指定
N <- 50

# ガウス分布に従うデータを生成
x_n <- rnorm(n = N, mean = mu, sd = sqrt(1 / lambda_truth))

# 観測データを確認
summary(x_n)


# 観測データのデータフレームを作成
data_df <- tibble(x_n = x_n)

# 観測データのヒストグラムを作成
ggplot() + 
  #geom_histogram(data = data_df, aes(x = x_n), binwidth = 1) + # 観測データ:(度数)
  geom_histogram(data = data_df, aes(x = x_n, y = ..density..), binwidth = 1) + # 観測データ:(相対度数)
  geom_line(data = model_df, aes(x = x, y = density), 
            color = "red", linetype = "dashed") + # 真の分布
  labs(title = "Gaussian Distribution", 
       subtitle = paste0("N=", N, ", mu=", mu, ", lambda=", lambda_truth), 
       x = "x")


### 事前分布(ガンマ分布)の設定 -----

# lambdaの事前分布のパラメータを指定
a <- 1
b <- 1


# 作図用のlambdaの値を作成
lambda_vec <- seq(0, 4 * lambda_truth, length.out = 1000)

# lambdaの事前分布を計算:式(2.56)
prior_df <- tibble(
  lambda = lambda_vec, # x軸の値
  ln_C_Gam = a * log(b) - lgamma(a), # 正規化項(対数)
  density = exp(ln_C_Gam + (a - 1) * log(lambda) - b * lambda) # 確率密度
  #density = dgamma(x = lambda, shape = a, rate = b) # 確率密度
)

# lambdaの事前分布を作図
ggplot(prior_df, aes(x = lambda, y = density)) + 
  geom_line(color = "purple") + # lambdaの事前分布
  geom_vline(aes(xintercept = lambda_truth), 
             color = "red", linetype = "dashed") + # 真の値
  labs(title = "Gamma Distribution", 
       subtitle = paste0("a=", a, ", b=", b))


### 事後分布(ガンマ分布)の計算 -----

# lambdaの事後分布のパラメータを計算:式(3.69)
a_hat <- 0.5 * N + a
b_hat <- 0.5 * sum((x_n - mu)^2) + b


# lambdaの事後分布を計算:式(2.56)
posterior_df <- tibble(
  lambda = lambda_vec, # x軸の値
  ln_C_Gam = a_hat * log(b_hat) - lgamma(a_hat), # 正規化項(対数)
  density = exp(ln_C_Gam + (a_hat - 1) * log(lambda) - b_hat * lambda) # 確率密度
  #density = dgamma(x = lambda, shape = a_hat, rate = b_hat) # 確率密度
)

# lambdaの事後分布を作図
ggplot(posterior_df, aes(x = lambda, y = density)) + 
  geom_line(color = "purple") + # lambdaの事後分布
  geom_vline(aes(xintercept = lambda_truth), 
             color = "red", linetype = "dashed") + # 真の値
  labs(title = "Gamma Distribution", 
       subtitle = paste0("N=", N, ", a_hat=", a_hat, ", b_hat=", round(b_hat, 1)), 
       x = expression(lambda))


### 予測分布(スチューデントのt分布)の計算 -----

# 予測分布のパラメータを計算:式(3.79)
mu_st <- mu
lambda_st_hat <- a_hat / b_hat
nu_st_hat <- 2 * a_hat
#lambda_st_hat <- (N + 2 * a) / (sum((x_n - mu)^2) + 2 * b)
#nu_st_hat <- N + 2 * a


# 予測分布を計算:式(3.76)
predict_df <- tibble(
  x = x_vec, # x軸の値
  ln_C_St = lgamma(0.5 * (nu_st_hat + 1)) - lgamma(0.5 * nu_st_hat), # 正規化項(対数)
  ln_term1 = 0.5 * log(lambda_st_hat / pi / nu_st_hat), 
  ln_term2 = - 0.5 * (nu_st_hat + 1) * log(1 + lambda_st_hat / nu_st_hat * (x - mu_st)^2), 
  density = exp(ln_C_St + ln_term1 + ln_term2) # 確率密度
)

# 予測分布を作図
ggplot() + 
  geom_line(data = predict_df, aes(x = x, y = density), 
            color = "purple") + # 予測分布
  geom_line(data = model_df, aes(x = x, y = density), 
            color = "red", linetype = "dashed") + # 真の分布
  labs(title = "Student's t Distribution", 
       subtitle = paste0("N=", N, 
                         ", mu_s=", mu_st, 
                         ", lambda_s_hat=", round(lambda_st_hat, 5), 
                         ", nu_s_hat=", nu_st_hat))


# ・アニメーションによる推移の確認 ---------------------------------------------------------------

# 3.3.2項で利用するパッケージ
library(tidyverse)
library(gganimate)


### モデルの設定 -----

# (既知の)パラメータを指定
mu <- 25

# 真の精度パラメータを指定
lambda_truth <- 0.01


# lambdaの事前分布のパラメータを指定
a <- 1
b <- 1


# 作図用のlambdaの値を作成
lambda_vec <- seq(0, 4 * lambda_truth, length.out = 1000)

# lambdaの事前分布(ガンマ分布)を計算:式(2.56)
posterior_df <- tibble(
  lambda = lambda_vec, # x軸の値
  density = dgamma(x = lambda, shape = a, rate = b), # 確率密度
  label = as.factor(paste0("N=", 0, ", a=", a, ", b=", b)) # フレーム切替用のラベル
)


# 初期値による予測分布のパラメータを計算:式(3.79)
mu_st <- mu
lambda_st <- a / b
nu_st <- 2 * a

# 作図用のxの値を作成
x_vec <- seq(
  mu - 4 * sqrt(1 / lambda_truth), 
  mu + 4 * sqrt(1 / lambda_truth), 
  length.out = 1000
)

# 初期値による予測分布(スチューデントのt分布)を計算:式(3.76)
predict_df <- tibble(
  x = x_vec, # x軸の値
  ln_C_St = lgamma(0.5 * (nu_st + 1)) - lgamma(0.5 * nu_st), # 正規化項(対数)
  ln_term1 = 0.5 * log(lambda_st / pi / nu_st), 
  ln_term2 = - 0.5 * (nu_st + 1) * log(1 + lambda_st / nu_st * (x - mu_st)^2), 
  density = exp(ln_C_St + ln_term1 + ln_term2), # 確率密度
  label = as.factor(
    paste0("N=", 0, ", mu_s=", mu_st, ", lambda_s=", lambda_st, ", nu_s", nu_st)
  ) # フレーム切替用のラベル
)


### 推論処理 -----

# データ数(試行回数)を指定
N <- 100

# 観測データの受け皿を作成
x_n <- rep(0, N)

# ベイズ推論
for(n in 1:N){
  
  # ガウス分布に従うデータを生成
  x_n[n] <- rnorm(n = 1, mean = mu, sd = sqrt(1 / lambda_truth))
  
  # lambdaの事後分布のパラメータを更新:式(3.69)
  a <- 1 / 2 + a
  b <- 0.5 * (x_n[n] - mu)^2 + b
  
  # lambdaの事後分布(ガンマ分布)を計算:式(2.56)
  tmp_posterior_df <- tibble(
    lambda = lambda_vec, # x軸の値
    density = dgamma(x = lambda, shape = a, rate = b), # 確率密度
    label = as.factor(
      paste0("N=", n, ", a_hat=", a, ", b_hat=", round(b, 1))
    ) # フレーム切替用のラベル
  )
  
  # 予測分布のパラメータを更新:式(3.79)
  mu_st <- mu
  lambda_st <- a / b
  nu_st <- 2 * a
  
  # 予測分布(スチューデントのt分布)を計算:式(3.76)
  tmp_predict_df <- tibble(
    x = x_vec, # x軸の値
    ln_C_St = lgamma(0.5 * (nu_st + 1)) - lgamma(0.5 * nu_st), # 正規化項(対数)
    ln_term1 = 0.5 * log(lambda_st / pi / nu_st), 
    ln_term2 = - 0.5 * (nu_st + 1) * log(1 + lambda_st / nu_st * (x - mu_st)^2), 
    density = exp(ln_C_St + ln_term1 + ln_term2), # 確率密度
    label = as.factor(
      paste0(
        "N=", n, ", mu_s=", mu_st, ", lambda_s_hat=", round(lambda_st, 5), ", nu_s_hat=", nu_st
      )
    ) # フレーム切替用のラベル
  )
  
  # n回目の結果を結合
  posterior_df <- rbind(posterior_df, tmp_posterior_df)
  predict_df <- rbind(predict_df, tmp_predict_df)
}

# 観測データを確認
summary(x_n)


### 作図処理 -----

# 観測データフレームを作成
label_list <- unique(posterior_df[["label"]]) # ラベルを抽出
data_df <- tibble(lambda_n = NA, label = label_list[1]) # 初期値
for(n in 1:N) {
  # n回目までの観測データ
  tmp_df <- tibble(
    lambda_n = 1 / x_n[1:n]^2, # 2乗の逆数に変換
    label = label_list[n + 1] # フレーム切替用のラベル
  )
  
  # 結合
  data_df <- rbind(data_df, tmp_df)
}

# lambdaの事後分布を作図
posterior_graph <- ggplot(posterior_df, aes(x = lambda, y = density)) + 
  geom_line(color = "purple") + # lambdaの事後分布
  geom_vline(aes(xintercept = lambda_truth), 
             color = "red", linetype = "dashed") + # 真の値
  geom_point(data = data_df, aes(x = lambda_n, y = 0)) + # 観測データ
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Gamma Distribution", 
       subtitle = "{current_frame}", 
       x = expression(lambda))

# gif画像を出力
gganimate::animate(posterior_graph, nframes = N + 1, fps = 10)


# 尤度を計算:式(2.64)
model_df <- tibble(
  x = x_vec, # x軸の値
  density = dnorm(x, mean = mu, sd = sqrt(1 / lambda_truth)) # 確率密度
)

# 観測データフレームを作成
label_list <- unique(predict_df[["label"]]) # ラベルを抽出
data_df <- tibble(x_n = NA, label = label_list[1]) # 初期値
for(n in 1:N) {
  # n回目までの観測データ
  tmp_df <- tibble(
    x_n = x_n[1:n], 
    label = label_list[n + 1] # フレーム切替用のラベル
  )
  
  # 結合
  data_df <- rbind(data_df, tmp_df)
}

# 予測分布を作図
predict_graph <- ggplot() + 
  geom_line(data = predict_df, aes(x = x, y = density), 
            color = "purple") + # 予測分布
  geom_line(data = model_df, aes(x = x, y = density), 
            color = "red", linetype = "dashed") + # 真の分布
  geom_point(data = data_df, aes(x = x_n, y = 0)) + # 観測データ
  gganimate::transition_manual(label) + # フレーム
  ylim(c(0, 0.1)) + # y軸の表示範囲
  labs(title = "Student's t Distribution", 
       subtitle = "{current_frame}")

# gif画像を出力
gganimate::animate(predict_graph, nframes = N + 1, fps = 10)


