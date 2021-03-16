
# 3.3.1 1次元ガウス分布の学習と予測：平均が未知の場合 ----------------------------------------------------

# 3.3.1項で利用するパッケージ
library(tidyverse)


### 尤度(ガウス分布)の設定 -----

# 真のパラメータを指定
mu_truth <- 25
lambda <- 0.01
print(sqrt(1 / lambda)) # 標準偏差

# 作図用のxの値を設定
x_line <- seq(
  mu_truth - 4 * sqrt(1 / lambda), 
  mu_truth + 4 * sqrt(1 / lambda), 
  length.out = 1000
)

# 尤度を計算:式(2.64)
model_df <- tibble(
  x = x_line, # x軸の値
  ln_C_N = - 0.5 * (log(2 * pi) - log(lambda)), # 正規化項(対数)
  density = exp(ln_C_N - 0.5 * lambda * (x - mu_truth)^2), # 確率密度
  #C_N = 1 / sqrt(2 * pi / lambda), # 正規化項
  #density = C_N * exp(- 0.5 * lambda * (x - mu_truth)^2) # 確率密度
  #density = dnorm(x = x, mean = mu_truth, sd = sqrt(1 / lambda)) # 確率密度
)

# 尤度を作図
ggplot(model_df, aes(x = x, y = density)) + 
  geom_line(color = "purple") + # 尤度
  labs(title = "Gaussian Distribution", 
       subtitle = paste0("mu=", round(mu_truth, 2), ", lamda=", lambda))


### 観測データの生成 -----

# (観測)データ数を指定
N <- 50

# ガウス分布に従うデータを生成
x_n <- rnorm(n = N, mean = mu_truth, sd = sqrt(1 / lambda))

# 観測データを確認
summary(x_n)

# 観測データのヒストグラムを作図
tibble(x = x_n) %>% 
  ggplot(aes(x = x)) + 
    geom_histogram(binwidth = 1) + # 観測データ
    labs(title = "Gaussian Distribution", 
         subtitle = paste0("N=", N, ", mu=", mu_truth, ", sigma=", round(sqrt(1 / lambda), 1)))


### 事前分布(ガウス分布)の設定 -----

# muの事前分布のパラメータを指定
m <- 0
lambda_mu <- 0.001
print(sqrt(1 / lambda_mu)) # 標準偏差

# 作図用のmuの値を設定
mu_line <- seq(mu_truth - 30, mu_truth + 30, length.out = 1000)

# muの事前分布を計算:式(2.64)
prior_df <- tibble(
  mu = mu_line, # x軸の値
  ln_C_N = - 0.5 * (log(2 * pi) - log(lambda_mu)), # 正規化項(対数)
  density = exp(ln_C_N - 0.5 * lambda_mu * (mu - m)^2) # 確率密度
  #density = dnorm(x = mu, mean = m, sd = sqrt(1 / lambda_mu)) # 確率密度
)

# muの事前分布を作図
ggplot(prior_df, aes(x = mu, y = density)) + 
  geom_line(color = "purple") + # muの事前分布
  labs(title = "Gaussian Distribution", 
       subtitle = paste0("m=", m, ", lambda_mu=", round(lambda_mu, 2)), 
       x = expression(mu))


### 事後分布(ガウス分布)の計算 -----

# muの事後分布のパラメータを計算:式(3.53),(3.54)
lambda_mu_hat <- N * lambda + lambda_mu
m_hat <- (lambda * sum(x_n) + lambda_mu * m) / lambda_mu_hat

# muの事後分布を計算:式(2.64)
posterior_df <- tibble(
  mu = mu_line, # x軸の値
  ln_C_N = - 0.5 * (log(2 * pi) - log(lambda_mu_hat)), # 正規化項(対数)
  density = exp(ln_C_N - 0.5 * lambda_mu_hat * (mu - m_hat)^2) # 確率密度
  #density = dnorm(x = mu, mean = m_hat, sd = sqrt(1 / lambda_mu_hat)) # 確率密度
)

# muの事後分布を作図
ggplot(posterior_df, aes(x = mu, y = density)) + 
  geom_line(color = "purple") + # muの事後分布
  geom_vline(aes(xintercept = mu_truth), 
             color = "red", linetype = "dashed") + # 真のパラメータ
  labs(title = "Gaussian Distribution", 
       subtitle = paste0("N=", N, ", m_hat=", round(m_hat, 1), ", lambda_mu_hat=", round(lambda_mu_hat, 2)), 
       x = expression(mu))


### 予測分布(ガウス分布)を計算 -----

# 予測分布のパラメータを計算:式(3.62')
lambda_star_hat <- lambda * lambda_mu_hat / (lambda + lambda_mu_hat)
mu_star_hat <- m_hat
#lambda_star_hat <- (N * lambda + lambda_mu) * lambda / ((N + 1) * lambda + lambda_mu)
#mu_star_hat <- (lambda * sum(x_n) + lambda_mu * m) / (N * lambda + lambda_mu)

# 予測分布を計算:式(2.64)
predict_df <- tibble(
  x = x_line, # x軸の値
  ln_C_N = - 0.5 * (log(2 * pi) - log(lambda_star_hat)), # 正規化項(対数)
  density = exp(ln_C_N - 0.5 * lambda_star_hat * (x - mu_star_hat)^2) # 確率密度
  #density = dnorm(x = x, mean = mu_star_hat, sd = sqrt(1 / lambda_star_hat)) # 確率密度
)

# 予測分布を作図
ggplot() + 
  geom_line(data = predict_df, aes(x = x, y = density), color = "purple") + # 予測分布
  geom_line(data = model_df, aes(x = x, y = density), color = "red", linetype = "dashed") + # 真の分布
  labs(title = "Gaussian Distribution", 
       subtitle = paste0("N=", N, ", mu_star_hat=", round(mu_star_hat, 1), 
                         ", sigma_star_hat=", round(sqrt(1 / lambda_star_hat), 1)))


# ・アニメーション ----------------------------------------------------

# 利用するパッケージ
library(tidyverse)
library(gganimate)


### 推論処理 -----

# 真のパラメータを指定
mu_truth <- 25
lambda <- 0.01

# muの事前分布のパラメータを指定
m <- 0
lambda_mu <- 0.001

# 作図用のmuの値を設定
mu_line <- seq(mu_truth - 50, mu_truth + 50, length.out = 1000)

# muの事前分布(ガウス分布)を計算:式(2.64)
posterior_df <- tibble(
  mu = mu_line, # x軸の値
  density = dnorm(x = mu, mean = m, sd = sqrt(1 / lambda_mu)), # 確率密度
  label = as.factor(
    paste0(
      "N=", 0, ", m=", m, ", lambda_mu=", round(lambda_mu, 2)
    )
  ) # パラメータ
)

# 初期値による予測分布のパラメータを計算:式(3.62)
lambda_star <- lambda * lambda_mu / (lambda + lambda_mu)
mu_star <- m

# 作図用のxの値を設定
x_line <- seq(
  mu_truth - 4 * sqrt(1 / lambda), 
  mu_truth + 4 * sqrt(1 / lambda), 
  length.out = 1000
)

# 初期値による予測分布(ガウス分布)を計算:式(2.64)
predict_df <- tibble(
  x = x_line, # x軸の値
  density = dnorm(x = x, mean = mu_star, sd = sqrt(1 / lambda_star)),  # 確率密度
  label = as.factor(
    paste0(
      "N=", 0, ", mu_star=", round(mu_star, 1), ", sigma_star=", round(sqrt(1 / lambda_star), 1)
    )
  ) # パラメータ
)

# データ数(試行回数)を指定
N <- 100

# 観測データの受け皿を初期化
x_n <- rep(0, N)

# ベイズ推論
for(n in 1:N){
  
  # ガウス分布に従うデータを生成
  x_n[n] <- rnorm(n = 1, mean = mu_truth, sd = sqrt(1 / lambda))
  
  # muの事後分布のパラメータを更新:式(3.53),(3.54)
  lambda_mu_old <- lambda_mu
  lambda_mu <- lambda + lambda_mu
  m <- (lambda * x_n[n] + lambda_mu_old * m) / lambda_mu
  
  # muの事後分布(ガウス分布)を計算:式(2.64)
  tmp_posterior_df <- tibble(
    mu = mu_line, # x軸の値
    density = dnorm(x = mu, mean = m, sd = sqrt(1 / lambda_mu)), # 確率密度
    label = as.factor(
      paste0(
        "N=", n, ", m_hat=", round(m, 1), ", lambda_mu_hat=", round(lambda_mu, 2)
      )
    ) # パラメータ
  )
  
  # 予測分布のパラメータを更新:式(3.62)
  lambda_star <- lambda * lambda_mu / (lambda + lambda_mu)
  mu_star <- m
  
  # 予測分布(ガウス分布)を計算:式(2.64)
  tmp_predict_df <- tibble(
    x = x_line, # x軸の値
    density = dnorm(x = x, mean = mu_star, sd = sqrt(1 / lambda_star)), # 確率密度
    label = as.factor(
      paste0(
        "N=", n, ", mu_star_hat=", round(mu_star, 1), ", sigma_star_hat=", round(sqrt(1 / lambda_star), 1)
      )
    ) # パラメータ
  )
  
  # 推論結果を結合
  posterior_df <- rbind(posterior_df, tmp_posterior_df)
  predict_df <- rbind(predict_df, tmp_predict_df)
}

# 観測データを確認
summary(x_n)


### 作図処理 -----

# muの事後分布を作図
posterior_graph <- ggplot(posterior_df, aes(x = mu, y = density)) + 
  geom_line(color = "purple") + # muの事後分布
  geom_vline(aes(xintercept = mu_truth), 
             color = "red", linetype = "dashed") + # 真のパラメータ
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Gaussian Distribution", 
       subtitle = "{current_frame}", 
       x = expression(mu))

# gif画像を出力
gganimate::animate(posterior_graph, nframes = N + 1, fps = 10)


# 尤度を計算
model_df <- tibble(
  x = x_line, # x軸の値
  density = dnorm(x, mean = mu_truth, sd = sqrt(1 / lambda)) # 確率密度
)

# 予測分布を作図
predict_graph <- ggplot() + 
  geom_line(data = predict_df, aes(x = x, y = density), 
            color = "purple") + # 予測分布
  geom_line(data = model_df, aes(x = x, y = density), 
            color = "red", linetype = "dashed") + # 真の分布
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Gaussian Distribution", 
       subtitle = "{current_frame}")

# gif画像を出力
gganimate::animate(predict_graph, nframes = N + 1, fps = 10)


