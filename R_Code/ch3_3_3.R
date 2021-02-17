
# 3.3.3 1次元ガウス分布の平均と精度の推定 --------------------------------------------------------------

# 3.3.3. 事後分布 --------------------------------------------------------------


# 利用パッケージ
library(tidyverse)


## パラメータの初期値を設定
# 観測モデルのパラメータ
mu_truth <- 25
lambda_truth <- 0.01

# 平均muの事前分布のパラメータ
m <- 20
beta <- 1

# 精度lambdaの事前分布のパラメータ
a <- 1
b <- 1

# 試行回数
N <- 50


# ガウス分布に従うデータを生成
x_n <- rnorm(n = N, mean = mu_truth, sd = sqrt(lambda_truth^(-1)))

# 観測データを確認
summary(x_n)


# 事後分布のパラメータを計算
beta_hat <- N + beta
m_hat <- (sum(x_n) + beta * m) / beta_hat
a_hat <- N / 2 + a
b_hat <- (sum(x_n^2) + beta * m^2 - beta_hat * m_hat^2) / 2 + b
lambda_bar <- a_hat / b_hat


# muの事後分布を計算
posterior_mu_df <- tibble(
  mu = seq(
    mu_truth - 3 * sqrt((beta_hat * lambda_bar)^(-1)), 
    mu_truth + 3 * sqrt((beta_hat * lambda_bar)^(-1)), 
    by = 0.01
  ),  # 作図用の値
  density = dnorm(mu, mean = m_hat, sd = sqrt((beta_hat * lambda_bar)^(-1))) # 確率密度
)

# 作図
ggplot(posterior_mu_df, aes(mu, density)) + 
  geom_line(color = "#56256E") + # 折れ線グラフ
  geom_vline(aes(xintercept = mu_truth), 
             color = "red", linetype = "dashed") + # 垂直線
  labs(title = "Gaussian Distribution", 
       subtitle = paste0("N=", N, ", m_hat=", round(m_hat, 2), ", beta_hat=", beta_hat), 
       x = expression(mu)) # ラベル


# lambdaの事後分布を計算
posterior_lambda_df <- tibble(
  lambda = seq(0, 3 * lambda_truth, by = 0.00001),  # 作図用の値
  C_G = a_hat * log(b_hat) - lgamma(a_hat),  # 正規化項(対数)
  density = exp(C_G + (a_hat - 1) * log(lambda) - b_hat * lambda) # 確率密度
)

# 作図
ggplot(posterior_lambda_df, aes(lambda, density)) + 
  geom_line(color = "#56256E") + # 折れ線グラフ
  geom_vline(aes(xintercept = lambda_truth), 
             color = "red", linetype = "dashed") + # 垂直線
  labs(title = "Gamma Distribution", 
       subtitle = paste0("N=", N, ", a_hat=", a_hat, ", b_hat=", round(b_hat, 2)), 
       x = expression(lambda)) # ラベル


# 3.3.3 予測分布 --------------------------------------------------------------


# 予測分布のパラメータを計算
mu_s_hat <- m_hat
lambda_s_hat <- beta_hat * a_hat / (1 + beta_hat) / b_hat
nu_s_hat <- 2 * a_hat

#mu_s_hat <- (sum(x_n) + beta * m) / (N + beta)
#tmp_lambda_numer <- (N + beta) * (N / 2 + a)
#tmp_lambda_denom <- (N + 1 + beta) * ((sum(x_n^2) + beta * m^2 - beta_hat * m_hat^2) / 2 + b)
#lambda_s_hat <- tmp_lambda_numer / tmp_lambda_denom
#nu_s_hat <- N + 2 * a

# 予測分布を計算
predict_df <- tibble(
  x = seq(-mu_s_hat, 3 * mu_s_hat, by = 0.001),  # 作図用の値
  C_St = lgamma((nu_s_hat + 1) / 2) - lgamma(nu_s_hat / 2),  # 正規化項(対数)
  term1 = log(lambda_s_hat / pi / nu_s_hat) / 2, 
  term2 = - (nu_s_hat + 1) / 2 * log(1 + lambda_s_hat / nu_s_hat * (x - mu_s_hat)^2), 
  density = exp(C_St + term1 + term2) # 確率密度
)


# 作図
ggplot(predict_df, aes(x, density)) + 
  geom_line(color = "#56256E") + # 折れ線グラフ
  labs(title = "Student's t Distribution", 
       subtitle = paste0("N=", N, ", mu=", round(mu_s_hat, 2), ", lambda=", round(lambda_s_hat, 2), ", nu=", nu_s_hat)) # ラベル



# 3.3.3. gif --------------------------------------------------------------


# 利用パッケージ
library(tidyverse)
library(gganimate)


## パラメータの初期値を設定
# 観測モデルのパラメータ
mu_truth <- 25
lambda_truth <- 0.01

# 平均muの事前分布のパラメータ
m <- 20
beta <- 1

# 精度lambdaの事前分布のパラメータ
a <- 1
b <- 1
lambda_bar <- a / b

# 試行回数
N <- 50


# muの事前分布を計算
posterior_mu_df <- tibble(
  mu = seq(
    mu_truth - 2 * sqrt(lambda_truth^(-1)), 
    mu_truth + 2 * sqrt(lambda_truth^(-1)), 
    by = 0.01
  ),  # 作図用の値
  density = dnorm(mu, mean = m, sd = sqrt(beta * lambda_bar)^(-1)), # 確率密度
  N = 0 # 試行回数
)


# lambdaの事前分布を計算
posterior_lambda_df <- tibble(
  lambda = seq(0, 4 * lambda_truth, by = 0.00001),  # 作図用の値
  C_G = a * log(b) - lgamma(a),  # 正規化項(対数)
  density = exp(C_G + (a - 1) * log(lambda) - b * lambda), # 確率密度
  N = 0 # 試行回数
)


# 初期値による予測分布のパラメータを計算
mu_s <- m
lambda_s <- beta * a / (1 + beta) / b
nu_s <- 2 * a

# 初期値による予測分布を計算
predict_df <- tibble(
  x = seq(-mu_truth, 3 * mu_truth, by = 0.001),  # 作図用の値
  C_St = lgamma((nu_s + 1) / 2) - lgamma(nu_s / 2),  # 正規化項(対数)
  term1 = log(lambda_s / pi / nu_s) / 2, 
  term2 = - (nu_s + 1) / 2 * log(1 + lambda_s / nu_s * (x - mu_s)^2), 
  density = exp(C_St + term1 + term2), # 確率密度
  N = 0 # 試行回数
)


# パラメータを推定
x_n <- rep(0, N)
for(n in 1:N){
  
  # ガウス分布に従うデータを生成
  x_n[n] <- rnorm(n = 1, mean = mu_truth, sd = sqrt(lambda_truth^(-1)))
  
  # パラメータを更新
  beta_old <- beta
  beta <- 1 + beta
  m_old <- m
  m <- (x_n[n] + beta_old * m) / beta
  a <- 1 / 2 + a
  b <- (x_n[n]^2 + beta_old * m_old^2 - beta * m^2) / 2 + b
  lambda_bar <- a / b
  
  
  # muの事後分布を計算
  tmp_posterior_mu_df <- tibble(
    mu = seq(
      mu_truth - 2 * sqrt(lambda_truth^(-1)), 
      mu_truth + 2 * sqrt(lambda_truth^(-1)), 
      by = 0.01
    ),  # 作図用のmu
    density = dnorm(mu, mean = m, sd = sqrt(beta * lambda_bar)^(-1)), # 確率密度
    N = n # 試行回数
  )
  
  # lambdaの事後分布を計算
  tmp_posterior_lambda_df <- tibble(
    lambda = seq(0, 4 * lambda_truth, by = 0.00001),  # 作図用の値
    C_G = a * log(b) - lgamma(a),  # 正規化項(対数)
    density = exp(C_G + (a - 1) * log(lambda) - b * lambda), # 確率密度
    N = n # 試行回数
  )
  
  
  # 予測分布のパラメータを更新
  mu_s <- m
  lambda_s <- beta * a / (1 + beta) / b
  nu_s <- 2 * a
  
  # 予測分布を計算
  tmp_predict_df <- tibble(
    x = seq(-mu_truth, 3 * mu_truth, by = 0.001),  # 作図用の値
    C_St = lgamma((nu_s + 1) / 2) - lgamma(nu_s / 2),  # 正規化項(対数)
    term1 = log(lambda_s / pi / nu_s) / 2, 
    term2 = - (nu_s + 1) / 2 * log(1 + lambda_s / nu_s * (x - mu_s)^2), 
    density = exp(C_St + term1 + term2), # 確率密度
    N = n # 試行回数
  )
  
  # 推定結果を結合
  posterior_mu_df <- rbind(posterior_mu_df, tmp_posterior_mu_df)
  posterior_lambda_df <- rbind(posterior_lambda_df, tmp_posterior_lambda_df)
  predict_df <- rbind(predict_df, tmp_predict_df)
}

# 観測データを確認
summary(x_n)


## muの事後分布
# 作図
posterior_mu_graph <- ggplot(posterior_mu_df, aes(mu, density)) + 
  geom_line(color = "#56256E") + # 折れ線グラフ
  geom_vline(aes(xintercept = mu_truth), 
             color = "red", linetype = "dashed") + # 垂直線
  transition_manual(N) + # フレーム
  labs(title = "Gaussian Distribution", 
       subtitle = "N= {current_frame}", 
       x = expression(mu)) # ラベル

# 描画
animate(posterior_mu_graph)


## lambdaの事後分布
# 作図
posterior_lambda_graph <- ggplot(posterior_lambda_df, aes(lambda, density)) + 
  geom_line(color = "#56256E") + # 折れ線グラフ
  geom_vline(aes(xintercept = lambda_truth), 
             color = "red", linetype = "dashed") + # 垂直線
  transition_manual(N) + # フレーム
  labs(title = "Gamma Distribution", 
       subtitle = "N= {current_frame}", 
       x = expression(lambda)) # ラベル

# 描画
animate(posterior_lambda_graph)


## 予測分布
# 作図
predict_graph <- ggplot(predict_df, aes(x, density)) + 
  geom_line(color = "#56256E") + # 折れ線グラフ
  transition_manual(N) + # フレーム
  labs(title = "Student's t Distribution", 
       subtitle = "N= {current_frame}") # ラベル


# 描画
animate(predict_graph)


