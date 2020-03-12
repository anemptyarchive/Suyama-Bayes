
# 3.3.1 1次元ガウス分布の学習と予測：平均が未知の場合 ----------------------------------------------------

# 3.3.1 事後分布 --------------------------------------------------------------


# 利用パッケージ
library(tidyverse)


## パラメータの初期値を指定
# 観測モデルのパラメータ
mu_truth <- 25
lambda <- 0.01

# 事前分布のパラメータ
m <- 20
lambda_mu <- 0.001

# 試行回数
N <- 50


# ガウス分布に従うデータを生成
x_n <- rnorm(n = N, mean = mu_truth, sd = sqrt(lambda^(-1)))

# 観測データを確認
summary(x_n)


# 事後分布のパラメータを計算
lambda_mu_hat <- N * lambda + lambda_mu
m_hat <- (lambda * sum(x_n) + lambda_mu * m) / lambda_mu_hat

# 事後分布を計算
posterior_df <- tibble(
  mu = seq(
    m_hat - 3 * sqrt(lambda_mu_hat^(-1)), 
    m_hat + 3 * sqrt(lambda_mu_hat^(-1)), 
    by = 0.1
  ),  # 作図用の値
  density = dnorm(mu, mean = m_hat, sd = sqrt(lambda_mu_hat^(-1))) # 確率密度
)

# 作図
ggplot(posterior_df, aes(mu, density)) + 
  geom_line(color = "#56256E") + # 折れ線グラフ
  geom_vline(aes(xintercept = mu_truth), 
             color = "red", linetype = "dashed") + # 垂直線
  labs(title = "Gaussian Distribution", 
       subtitle = paste0("N=", N, ", m_hat=", round(m_hat, 2), ", lambda_mu_hat=", lambda_mu_hat), 
       x = expression(mu)) # ラベル


# 3.3.1 予測分布 --------------------------------------------------------------


# 予測分布のパラメータを計算
lambda_hat <- lambda * lambda_mu_hat / (lambda + lambda_mu_hat)
mu_hat <- m_hat

#lambda_hat <- (N * lambda + lambda_mu) * lambda / ((N + 1) * lambda + lambda_mu)
#mu_hat <- (lambda * sum(x_n) + lambda_mu * m) / (N * lambda + lambda_mu)


# 予測分布の計算
predict_df <- tibble(
  x = seq(
    mu_hat - 3 * sqrt(lambda_hat^(-1)), 
    mu_hat + 3 * sqrt(lambda_hat^(-1)), 
    by = 0.1
  ),  # 作図用の値
  density = dnorm(x, mean = mu_hat, sd = sqrt(lambda_hat^(-1))) # 確率密度
)


# 作図
ggplot(predict_df, aes(x, density)) + 
  geom_line(color = "#56256E") + # 折れ線グラフ
  geom_vline(aes(xintercept = mu_truth), 
             color = "red", linetype = "dashed") + # 垂直線
  labs(title = "Gaussian Distribution", 
       subtitle = paste0("N=", N, ", m_hat=", round(mu_hat, 2), ", lambda_mu_hat=", round(lambda_hat, 2)), 
       x = expression(mu)) # ラベル


# 3.3.1 gif ----------------------------------------------------


# 利用パッケージ
library(tidyverse)
library(gganimate)


## パラメータの初期値を指定
# 観測モデルのパラメータ
mu_truth <- 25
lambda <- 0.01

# 事前分布のパラメータ
m <- 20
lambda_mu <- 0.001

# 試行回数
N <- 50


# 事前分布を計算
posterior_df <- tibble(
  mu = seq(
    mu_truth - 2 * sqrt(lambda^(-1)), 
    mu_truth + 2 * sqrt(lambda^(-1)), 
    by = 0.01
  ),  # 作図用の値
  density = dnorm(mu, mean = m, sd = sqrt(lambda_mu^(-1))), # 確率密度
  N = 0  # 試行回数
)


# 初期値による予測分布のパラメータを計算
lambda_star <- lambda * lambda_mu / (lambda + lambda_mu)
mu_star <- m

# 初期値による予測分布を計算
predict_df <- tibble(
  x = seq(
    mu_truth - 3 * sqrt(lambda^(-1)), 
    mu_truth + 3 * sqrt(lambda^(-1)), 
    by = 0.01
  ),  # 作図用の値
  density = dnorm(x, mean = mu_star, sd = sqrt(lambda_star^(-1))),  # 確率密度
  N = 0 # 試行回数
)


# パラメータを推定
x_n <- rep(0, N) # 受け皿
for(n in 1:N){
  
  # ガンマ分布に従うデータを生成
  x_n[n] <- rnorm(n = 1, mean = mu_truth, sd = sqrt(lambda^(-1)))
  
  
  # パラメータを更新
  lambda_mu_old <- lambda_mu
  lambda_mu <- 1 * lambda + lambda_mu
  m <- (lambda * x_n[n] + lambda_mu_old * m) / lambda_mu
  
  # 事後分布を計算
  tmp_posterior_df <- tibble(
    mu = seq(
      mu_truth - 2 * sqrt(lambda^(-1)), 
      mu_truth + 2 * sqrt(lambda^(-1)), 
      by = 0.01
    ),  # 作図用の値
    density = dnorm(mu, mean = m, sd = sqrt(lambda_mu^(-1))),  # 確率密度
    N = n  # 試行回数
  )
  
  
  # 予測分布のパラメータを計算
  lambda_star <- lambda * lambda_mu / (lambda + lambda_mu)
  mu_star <- m
  
  # 予測分布を計算
  tmp_predict_df <- tibble(
    x = seq(
      mu_truth - 3 * sqrt(lambda^(-1)), 
      mu_truth + 3 * sqrt(lambda^(-1)), 
      by = 0.01
    ),  # 作図用の値
    density = dnorm(x, mean = mu_star, sd = sqrt(lambda_star^(-1))),  # 確率密度
    N = n # 試行回数
  )
  
  
  # 推定結果を結合
  posterior_df <- rbind(posterior_df, tmp_posterior_df)
  predict_df <- rbind(predict_df, tmp_predict_df)
}

# 観測データを確認
summary(x_n)


## 事後分布
# 作図
posterior_graph <- ggplot(posterior_df, aes(mu, density)) + 
  geom_line(color = "#56256E") + # 折れ線グラフ
  geom_vline(aes(xintercept = mu_truth), 
             color = "red", linetype = "dashed") + # 垂直線
  transition_manual(N) + # フレーム
  labs(title = "Gaussian Distribution", 
       subtitle = "N= {current_frame}", 
       x = expression(mu)) # ラベル

# 描画
animate(posterior_graph)


## 予測分布
predict_graph <- ggplot(predict_df, aes(x, density)) + 
  geom_line(color = "#56256E") + # 折れ線グラフ
  geom_vline(aes(xintercept = mu_truth), 
             color = "red", linetype = "dashed") + # 垂直線
  transition_manual(N) + # フレーム
  labs(title = "Gaussian Distribution", 
       subtitle = "N= {current_frame}", 
       x = expression(x)) # ラベル

# 描画
animate(predict_graph)


