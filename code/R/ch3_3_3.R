
# 3.3.3 1次元ガウス分布の学習と予測：平均・精度が未知の場合 --------------------------------------------------------------

# 3.3.3で利用するパッケージ
library(tidyverse)


### 尤度(ガウス分布)の設定 -----

# 真の平均パラメータを指定
mu_truth <- 25

# 真の精度パラメータを指定
lambda_truth <- 0.01
sqrt(1 / lambda_truth) # 標準偏差


# 作図用のxの値を作成
x_vec <- seq(
  mu_truth - 4 * sqrt(1 / lambda_truth), 
  mu_truth + 4 * sqrt(1 / lambda_truth), 
  length.out = 1000
)

# 尤度を計算:式(2.64)
model_df <- tibble(
  x = x_vec, # x軸の値
  ln_C_N = - 0.5 * (log(2 * pi) - log(lambda_truth)), # 正規化項(対数)
  density = exp(ln_C_N - 0.5 * lambda_truth * (x - mu_truth)^2) # 確率密度
  #C_N = 1 / sqrt(2 * pi / lambda_truth), # 正規化項
  #density = C_N * exp(- 0.5 * lambda_truth * (x - mu_truth)^2) # 確率密度
  #density = dnorm(x = x, mean = mu_truth, sd = sqrt(1 / lambda_truth)) # 確率密度
)

# 尤度を作図
ggplot(model_df, aes(x = x, y = density)) + 
  geom_line(color = "blue") + # 尤度
  labs(title = "Gaussian Distribution", 
       subtitle = paste0("mu=", mu_truth, ", lambda=", lambda_truth))


### データの生成 -----

# (観測)データ数を指定
N <- 50

# ガウス分布に従うデータを生成
x_n <- rnorm(n = N, mean = mu_truth, sd = sqrt(1 / lambda_truth))

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
       subtitle = paste0("N=", N, ", mu=", mu_truth, ", lambda=", lambda_truth), 
       x = "x")


### 事前分布(ガウス・ガンマ分布)の設定 -----

# muの事前分布のパラメータを指定
m <- 0
beta <- 1


# lambdaの事前分布のパラメータを指定
a <- 1
b <- 1


# lambdaの期待値を計算:式(2.59)
E_lambda <- a / b

# 作図用のmuの値を作成
mu_vec <- seq(mu_truth - 50, mu_truth + 50, length.out = 1000)

# muの事前分布を計算:式(2.64)
prior_mu_df <- tibble(
  mu = mu_vec, # x軸の値
  ln_C_N = - 0.5 * (log(2 * pi) - log(beta * E_lambda)), # 正規化項(対数)
  density = exp(ln_C_N - 0.5 * beta * E_lambda * (mu - m)^2) # 確率密度
  #density = dnorm(x = mu, mean = m, sd = sqrt(1 / beta / E_lambda)) # 確率密度
)

# muの事前分布を作図
ggplot(prior_mu_df, aes(x = mu, y = density)) + 
  geom_line(color = "purple") + # muの事前分布
  geom_vline(aes(xintercept = mu_truth), 
             color = "red", linetype = "dashed") + # 真の値
  labs(title = "Gaussian Distribution", 
       subtitle = paste0("m=", m, ", beta=", beta, ', E[lambda]=', E_lambda), 
       x = expression(mu))


# 作図用のlambdaの値を作成
lambda_vec <- seq(0, 4 * lambda_truth, length.out = 1000)

# lambdaの事前分布を計算:式(2.56)
prior_lambda_df <- tibble(
  lambda = lambda_vec, # x軸の値
  ln_C_Gam = a * log(b) - lgamma(a), # 正規化項(対数)
  density = exp(ln_C_Gam + (a - 1) * log(lambda) - b * lambda) # 確率密度
  #density = dgamma(x = lambda, shape = a, rate = b) # 確率密度
)

# lambdaの事前分布を作図
ggplot(prior_lambda_df, aes(x = lambda, y = density)) + 
  geom_line(color = "purple") + # lambdaの事前分布
  geom_vline(aes(xintercept = lambda_truth), 
             color = "red", linetype = "dashed") + # 真の値
  labs(title = "Gamma Distribution", 
       subtitle = paste0("a=", a, ", b=", b), 
       x = expression(lambda))


### 事後分布(ガウス・ガンマ分布)の計算 -----

# muの事後分布のパラメータを計算:式(3.83)
beta_hat <- N + beta
m_hat <- (sum(x_n) + beta * m) / beta_hat


# lambdaの事後分布のパラメータを計算:式(3.88)
a_hat <- 0.5 * N + a
b_hat <- 0.5 * (sum(x_n^2) + beta * m^2 - beta_hat * m_hat^2) + b


# lambdaの期待値を計算:式(2.59)
E_lambda_hat <- a_hat / b_hat

# muの事後分布を計算:式(2.64)
posterior_mu_df <- tibble(
  mu = mu_vec, # x軸の値
  ln_C_N = - 0.5 * (log(2 * pi) - log(beta_hat * E_lambda_hat)), # 正規化項(対数)
  density = exp(ln_C_N - 0.5 * beta_hat * E_lambda_hat * (mu - m_hat)^2) # 確率密度
  #density = dnorm(mu, mean = m_hat, sd = sqrt(1 / beta_hat / E_lambda_hat)) # 確率密度
)

# muの事後分布を作図
ggplot(posterior_mu_df, aes(x = mu, y = density)) + 
  geom_line(color = "purple") + # muの事後分布
  geom_vline(aes(xintercept = mu_truth), 
             color = "red", linetype = "dashed") + # 真の値
  labs(title = "Gaussian Distribution", 
       subtitle = paste0("N=", N, 
                         ", m_hat=", round(m_hat, 1), 
                         ", beta_hat=", beta_hat, 
                         ", E[lambda]=", round(E_lambda_hat, 5)), 
       x = expression(mu))


# lambdaの事後分布を計算:式(2.56)
posterior_lambda_df <- tibble(
  lambda = lambda_vec, # x軸の値
  ln_C_Gam = a_hat * log(b_hat) - lgamma(a_hat), # 正規化項(対数)
  density = exp(ln_C_Gam + (a_hat - 1) * log(lambda) - b_hat * lambda) # 確率密度
  #density = dgamma(x = lambda, shape = a_hat, rate = b_hat) # 確率密度
)

# lambdaの事後分布を作図
ggplot(posterior_lambda_df, aes(x = lambda, y = density)) + 
  geom_line(color = "purple") + # lambdaの事後分布
  geom_vline(aes(xintercept = lambda_truth), 
             color = "red", linetype = "dashed") + # 真の値
  labs(title = "Gamma Distribution", 
       subtitle = paste0("N=", N, ", a_hat=", a_hat, ", b_hat=", round(b_hat, 1)), 
       x = expression(lambda))


### 予測分布(スチューデントのt分布)の計算 -----

# 予測分布のパラメータを計算:式(3.95')
mu_st_hat <- m_hat
lambda_st_hat <- beta_hat * a_hat / (1 + beta_hat) / b_hat
nu_st_hat <- 2 * a_hat
#mu_st_hat <- (sum(x_n) + beta * m) / (N + beta)
#numer_lambda <- (N + beta) * (N / 2 + a)
#denom_lambda <- (N + 1 + beta) * ((sum(x_n^2) + beta * m^2 - beta_hat * m_hat^2) / 2 + b)
#lambda_st_hat <- numer_lambda / denom_lambda
#nu_st_hat <- N + 2 * a


# 予測分布を計算:式(3.76)
predict_df <- tibble(
  x = x_vec, # x軸の値
  ln_C_St = lgamma(0.5 * (nu_st_hat + 1)) - lgamma(0.5 * nu_st_hat), # 正規化項(対数)
  ln_term1 = 0.5 * log(lambda_st_hat / pi / nu_st_hat), 
  ln_term2 = - 0.5 * (nu_st_hat + 1) * log(1 + lambda_st_hat / nu_st_hat * (x - mu_st_hat)^2), 
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
                         ", mu_s_hat=", round(mu_st_hat, 1), 
                         ", lambda_s_hat=", round(lambda_st_hat, 5), 
                         ", nu_s_hat=", nu_st_hat))


# ・アニメーションによる推移の確認 --------------------------------------------------------------

# 3.3.3項で利用するパッケージ
library(tidyverse)
library(gganimate)


### モデルの設定 -----

# 真の平均パラメータを指定
mu_truth <- 25

# 真の精度パラメータを指定
lambda_truth <- 0.01


# muの事前分布のパラメータを指定
m <- 0
beta <- 1


# lambdaの事前分布のパラメータを指定
a <- 1
b <- 1


# 作図用のmuの値を作成
mu_vec <- seq(mu_truth - 30, mu_truth + 30, length.out = 1000)

# muの事前分布(ガウス分布)を計算:式(2.64)
posterior_mu_df <- tibble(
  mu = mu_vec, # x軸の値
  density = dnorm(mu, mean = m, sd = sqrt(1 / (beta * a / b))), # 確率密度
  label = as.factor(
    paste0(
      "N=", 0, ", m=", m, ", beta=", beta, ", E[lambda]=", a / b
    )
  ) # フレーム切替用のラベル
)


# 作図用のlambdaの値を作成
lambda_vec <- seq(0, 4 * lambda_truth, length.out = 1000)

# lambdaの事前分布(ガンマ分布)を計算:式(2.56)
posterior_lambda_df <- tibble(
  lambda = lambda_vec, # x軸の値
  density = dgamma(x = lambda, shape = a, rate = b), # 確率密度
  label = as.factor(paste0("N=", 0, ", a=", a, ", b=", b)) # フレーム切替用のラベル
)


# 初期値による予測分布のパラメータを計算:式(3.95)
mu_st <- m
lambda_st <- beta * a / (1 + beta) / b
nu_st <- 2 * a

# 作図用のxの値を作成
x_vec <- seq(
  mu_truth - 4 * sqrt(1 / lambda_truth), 
  mu_truth + 4 * sqrt(1 / lambda_truth), 
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
    paste0(
      "N=", 0, ", nu_s=", nu_st, ", mu_s=",mu_st, ", lambda_s=", lambda_st
    )
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
  x_n[n] <- rnorm(n = 1, mean = mu_truth, sd = sqrt(1 / lambda_truth))
  
  # muの事後分布のパラメータを更新:式(3.83)
  beta_old <- beta
  beta <- 1 + beta
  m_old <- m
  m <- (x_n[n] + beta_old * m) / beta
  
  # lambdaの事後分布のパラメータを更新:式(3.88)
  a <- 1 / 2 + a
  b <- (x_n[n]^2 + beta_old * m_old^2 - beta * m^2) / 2 + b
  
  # muの事後分布(ガウス分布)を計算:式(2.64)
  tmp_posterior_mu_df <- tibble(
    mu = mu_vec, # x軸の値
    density = dnorm(mu, mean = m, sd = sqrt(1 / (beta * a / b))), # 確率密度
    label = as.factor(
      paste0(
        "N=", n, ", m_hat=", round(m, 1), 
        ", beta_hat=", beta, ", E[lambda_hat]=", round(a / b, 5)
      )
    ) # フレーム切替用のラベル
  )
  
  # lambdaの事後分布(ガンマ分布)を計算:式(2.56)
  tmp_posterior_lambda_df <- tibble(
    lambda = lambda_vec, # x軸の値
    density = dgamma(x = lambda, shape = a, rate = b), # 確率密度
    label = as.factor(
      paste0("N=", n, ", a_hat=", a, ", b_hat=", round(b, 1))
    ) # フレーム切替用のラベル
  )
  
  # 予測分布のパラメータを更新:式(3.95)
  mu_st <- m
  lambda_st <- beta * a / (1 + beta) / b
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
        "N=", n, 
        ", mu_s_hat=", round(mu_st, 1), 
        ", lambda_s_hat=", round(lambda_st, 5), 
        ", nu_s_hat=", nu_st
      )
    ) # フレーム切替用のラベル
  )
  
  # n回目の結果を結合
  posterior_mu_df <- rbind(posterior_mu_df, tmp_posterior_mu_df)
  posterior_lambda_df <- rbind(posterior_lambda_df, tmp_posterior_lambda_df)
  predict_df <- rbind(predict_df, tmp_predict_df)
}

# 観測データを確認
summary(x_n)


### 作図処理 -----

# 観測データフレームを作成
label_list <- unique(posterior_mu_df[["label"]]) # ラベルを抽出
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

# muの事後分布を作図
posterior_mu_graph <- ggplot() + 
  geom_line(data = posterior_mu_df, aes(x = mu, y = density), 
            color = "purple") + # muの事後分布
  geom_vline(aes(xintercept = mu_truth), 
             color = "red", linetype = "dashed") + # 真の値
  geom_point(data = data_df, aes(x = x_n, y = 0)) + # 観測データ
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Gaussian Distribution", 
       subtitle = "{current_frame}", 
       x = expression(mu))

# gif画像を出力
gganimate::animate(posterior_mu_graph, nframes = N + 1, fps = 10)


# 観測データフレームを作成
label_list <- unique(posterior_lambda_df[["label"]]) # ラベルを抽出
data_df <- tibble(lambda_n = NA, label = label_list[1]) # 初期値
for(n in 1:N) {
  # n回目までの観測データ
  tmp_df <- tibble(
    lambda_n = 1 / x_n[1:n]^2, # 2乗の平方根に変換
    label = label_list[n + 1] # フレーム切替用のラベル
  )
  
  # 結合
  data_df <- rbind(data_df, tmp_df)
}

# lambdaの事後分布を作図
posterior_lambda_graph <- ggplot() + 
  geom_line(data = posterior_lambda_df, aes(x = lambda, y = density), 
            color = "purple") + # lambdaの事後分布
  geom_vline(aes(xintercept = lambda_truth), 
             color = "red", linetype = "dashed") + # 真の値
  geom_point(data = data_df, aes(x = lambda_n, y = 0)) + # 観測データ
  gganimate::transition_manual(label) + # フレーム
  #xlim(c(0, 2 * max(lambda_vec))) + # x軸の表示範囲
  labs(title = "Gamma Distribution", 
       subtitle = "{current_frame}", 
       x = expression(lambda))

# gif画像を出力
gganimate::animate(posterior_lambda_graph, nframes = N + 1, fps = 10)


# 尤度を計算:式(2.64)
model_df <- tibble(
  x = x_vec, # x軸の値
  density = dnorm(x = x, mean = mu_truth, sd = sqrt(1 / lambda_truth)) # 確率密度
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


